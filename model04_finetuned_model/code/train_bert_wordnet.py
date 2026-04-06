from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from bert_wordnet_model import BertWordNetClassifier, BertWordNetConfig


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "bert_wordnet"

TRAIN_SPLIT = PROCESSED_ROOT / "hc3_wordnet_train.csv"
VAL_SPLIT = PROCESSED_ROOT / "hc3_wordnet_val.csv"
TEST_SPLIT = PROCESSED_ROOT / "hc3_wordnet_test.csv"

FEATURE_COLUMNS = [
    "wn_avg_synset_count",
    "wn_avg_synonym_count",
    "wn_coverage_ratio",
    "wn_avg_hypernym_depth",
]

REQUIRED_MODEL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
]
OPTIONAL_WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
]


class BertWordNetDataset(Dataset):
    """Dataset wrapper that keeps the text, label, and scaled WordNet features together."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        scaler: StandardScaler,
        max_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = scaler.transform(self.df[FEATURE_COLUMNS].to_numpy())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "example_id": row["example_id"],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "wordnet_features": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT + WordNet feature fusion classifier.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, default=TRAIN_SPLIT)
    parser.add_argument("--val-path", type=Path, default=VAL_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Defaults to 0 for conservative Windows stability.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for this training run. A timestamped name is generated if omitted.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a short GPU sanity-check with small train/validation subsets.",
    )
    parser.add_argument(
        "--sanity-train-samples",
        type=int,
        default=256,
        help="Number of training examples to use in --sanity-check mode.",
    )
    parser.add_argument(
        "--sanity-val-samples",
        type=int,
        default=128,
        help="Number of validation examples to use in --sanity-check mode.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_sanity_check_overrides(args: argparse.Namespace) -> None:
    if not args.sanity_check:
        return

    args.epochs = 1
    args.max_length = 128


def limit_dataframe_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    if limit <= 0:
        raise ValueError(f"Sample limit must be positive, got {limit}.")
    return df.head(limit).reset_index(drop=True)


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    mode = "sanity" if args.sanity_check else "formal"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{timestamp}"


def resolve_model_source(model_name: str, model_path: Path | None) -> str:
    if model_path is None:
        return model_name

    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(
            "The provided --model-path does not exist or is not a directory: "
            f"{model_path}\n"
            "Required local BERT files:\n"
            "- config.json\n"
            "- tokenizer_config.json\n"
            "- vocab.txt\n"
            "- one of: model.safetensors or pytorch_model.bin"
        )

    missing_files = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).exists()]
    has_weights = any((model_path / name).exists() for name in OPTIONAL_WEIGHT_FILES)

    if missing_files or not has_weights:
        missing_message = "\n".join(f"- {name}" for name in missing_files) if missing_files else "- none"
        weight_message = "- model.safetensors or pytorch_model.bin" if not has_weights else "- present"
        raise FileNotFoundError(
            "The provided --model-path is missing required BERT files: "
            f"{model_path}\n"
            "Missing required config/tokenizer files:\n"
            f"{missing_message}\n"
            "Weight file check:\n"
            f"{weight_message}"
        )

    return str(model_path)


def create_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].to_numpy())
    return scaler


def create_data_loader(
    df: pd.DataFrame,
    tokenizer,
    scaler: StandardScaler,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = BertWordNetDataset(df=df, tokenizer=tokenizer, scaler=scaler, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def format_elapsed_time(start_time: float) -> str:
    elapsed_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def run_epoch(model, data_loader, optimizer, device, epoch_number: int, total_epochs: int) -> float:
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()
    total_batches = len(data_loader)

    print(f"Starting training epoch {epoch_number}/{total_epochs} with {total_batches} batches")

    for batch_index, batch in enumerate(data_loader, start=1):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            wordnet_features=batch["wordnet_features"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        total_loss += loss_value

        print(
            "Train epoch {epoch}/{total_epochs} | batch {batch}/{total_batches} | "
            "loss={loss:.6f} | elapsed={elapsed}".format(
                epoch=epoch_number,
                total_epochs=total_epochs,
                batch=batch_index,
                total_batches=total_batches,
                loss=loss_value,
                elapsed=format_elapsed_time(epoch_start_time),
            )
        )

    return total_loss / max(len(data_loader), 1)


def evaluate_model(
    model,
    data_loader,
    device,
    epoch_number: int,
    total_epochs: int,
) -> tuple[dict[str, float], list[dict[str, int | str]]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    prediction_rows: list[dict[str, int | str]] = []
    eval_start_time = time.time()
    total_batches = len(data_loader)

    print(f"Starting validation for epoch {epoch_number}/{total_epochs} with {total_batches} batches")

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                wordnet_features=batch["wordnet_features"].to(device),
            )
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()

            print(
                "Validation epoch {epoch}/{total_epochs} | batch {batch}/{total_batches} | "
                "elapsed={elapsed}".format(
                    epoch=epoch_number,
                    total_epochs=total_epochs,
                    batch=batch_index,
                    total_batches=total_batches,
                    elapsed=format_elapsed_time(eval_start_time),
                )
            )

            all_predictions.extend(predictions)
            all_labels.extend(labels)

            for example_id, label, prediction in zip(batch["example_id"], labels, predictions):
                prediction_rows.append(
                    {
                        "example_id": example_id,
                        "label": label,
                        "prediction": prediction,
                    }
                )

    return compute_metrics(all_labels, all_predictions), prediction_rows


def save_training_artifacts(
    output_dir: Path,
    model,
    tokenizer,
    scaler: StandardScaler,
    metrics: dict[str, float],
    config: BertWordNetConfig,
    prediction_rows: list[dict[str, int | str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    scaler_state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_columns": FEATURE_COLUMNS,
    }
    (output_dir / "scaler.json").write_text(json.dumps(scaler_state, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "val_predictions.csv", index=False)


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    completed_epochs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_name": run_name,
        "mode": "sanity-check" if args.sanity_check else "formal",
        "device": str(device),
        "requested_epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "model_path": str(args.model_path) if args.model_path is not None else None,
        "sanity_train_samples": args.sanity_train_samples if args.sanity_check else None,
        "sanity_val_samples": args.sanity_val_samples if args.sanity_check else None,
    }
    (output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_epoch_metrics(output_dir: Path, epoch_history: list[dict[str, float | int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "epoch_metrics.json").write_text(json.dumps(epoch_history, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_sanity_check_overrides(args)
    set_seed(args.seed)
    run_name = resolve_run_name(args)
    run_output_dir = (args.output_dir / run_name).resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train_path)
    val_df = load_split(args.val_path)

    if args.sanity_check:
        train_df = limit_dataframe_rows(train_df, args.sanity_train_samples)
        val_df = limit_dataframe_rows(val_df, args.sanity_val_samples)
        print("Sanity-check mode enabled")
        print(
            json.dumps(
                {
                    "run_name": run_name,
                    "train_samples": len(train_df),
                    "val_samples": len(val_df),
                    "epochs": args.epochs,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                },
                indent=2,
            )
        )

    model_source = resolve_model_source(args.model_name, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    scaler = create_scaler(train_df)

    train_loader = create_data_loader(
        train_df,
        tokenizer=tokenizer,
        scaler=scaler,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_data_loader(
        val_df,
        tokenizer=tokenizer,
        scaler=scaler,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertWordNetConfig(model_name=model_source)
    model = BertWordNetClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_history: list[dict[str, float | int]] = []
    completed_epochs = 0

    print(f"Run name: {run_name}")
    print(f"Run output directory: {run_output_dir}")
    print(f"Device: {device}")
    print(f"Requested epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"DataLoader num_workers: {args.num_workers}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    best_metrics: dict[str, float] | None = None
    best_predictions: list[dict[str, int | str]] = []
    best_state_dict = None

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch + 1, args.epochs)
        val_metrics, prediction_rows = evaluate_model(model, val_loader, device, epoch + 1, args.epochs)
        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch + 1
        epoch_history.append(dict(val_metrics))
        completed_epochs = epoch + 1
        save_epoch_metrics(run_output_dir, epoch_history)
        save_run_metadata(run_output_dir, args, device, run_name, completed_epochs)

        print(
            "Epoch {epoch}/{total} | train_loss={train_loss:.6f} | "
            "val_accuracy={accuracy:.6f} | val_precision={precision:.6f} | "
            "val_recall={recall:.6f} | val_f1={f1:.6f}".format(
                epoch=epoch + 1,
                total=args.epochs,
                train_loss=train_loss,
                accuracy=val_metrics["accuracy"],
                precision=val_metrics["precision"],
                recall=val_metrics["recall"],
                f1=val_metrics["f1"],
            )
        )
        print(json.dumps(val_metrics, indent=2))

        if best_metrics is None or val_metrics["f1"] > best_metrics["f1"]:
            best_metrics = val_metrics
            best_predictions = prediction_rows
            best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_metrics is None or best_state_dict is None:
        raise RuntimeError("Training did not produce any metrics.")

    model.load_state_dict(best_state_dict)
    save_training_artifacts(
        output_dir=run_output_dir,
        model=model,
        tokenizer=tokenizer,
        scaler=scaler,
        metrics=best_metrics,
        config=config,
        prediction_rows=best_predictions,
    )
    save_epoch_metrics(run_output_dir, epoch_history)
    save_run_metadata(run_output_dir, args, device, run_name, completed_epochs)

    print()
    print(f"Saved training artifacts to: {run_output_dir}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()

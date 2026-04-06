from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from bert_wordnet_model import BertWordNetClassifier, BertWordNetConfig


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "bert_wordnet"

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
    def __init__(self, df: pd.DataFrame, tokenizer, scaler: StandardScaler, max_length: int) -> None:
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
    parser = argparse.ArgumentParser(description="Evaluate a trained BERT + WordNet model.")
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=TEST_SPLIT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def resolve_model_source(model_dir: Path, model_path: Path | None) -> str:
    if model_path is None:
        return str(model_dir / "tokenizer")

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


def load_scaler(model_dir: Path) -> StandardScaler:
    scaler_state = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = torch.tensor(scaler_state["mean"], dtype=torch.float64).numpy()
    scaler.scale_ = torch.tensor(scaler_state["scale"], dtype=torch.float64).numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler_state["feature_columns"])
    return scaler


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main() -> None:
    args = parse_args()

    config_dict = json.loads((args.model_dir / "model_config.json").read_text(encoding="utf-8"))
    config = BertWordNetConfig(**config_dict)

    model_source = resolve_model_source(args.model_dir, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    scaler = load_scaler(args.model_dir)
    test_df = pd.read_csv(args.test_path)
    dataset = BertWordNetDataset(test_df, tokenizer=tokenizer, scaler=scaler, max_length=args.max_length)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertWordNetClassifier(config).to(device)
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location=device))
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []
    prediction_rows: list[dict[str, int | str]] = []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                wordnet_features=batch["wordnet_features"].to(device),
            )
            predictions = torch.argmax(outputs["logits"], dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()

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

    metrics = compute_metrics(all_labels, all_predictions)
    (args.model_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(args.model_dir / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to: {args.model_dir / 'test_metrics.json'}")
    print(f"Saved test predictions to: {args.model_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()

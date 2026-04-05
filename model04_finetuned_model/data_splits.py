from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

INPUT_DATASET = PROCESSED_ROOT / "unified_binary_classification_wordnet.csv"
TRAIN_OUTPUT = PROCESSED_ROOT / "hc3_wordnet_train.csv"
VAL_OUTPUT = PROCESSED_ROOT / "hc3_wordnet_val.csv"
TEST_OUTPUT = PROCESSED_ROOT / "hc3_wordnet_test.csv"
SPLIT_SUMMARY_OUTPUT = PROCESSED_ROOT / "hc3_wordnet_split_summary.json"

MODEL_COLUMNS = [
    "example_id",
    "text",
    "label",
    "wn_avg_synset_count",
    "wn_avg_synonym_count",
    "wn_coverage_ratio",
    "wn_avg_hypernym_depth",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_wordnet_dataset(input_path: Path = INPUT_DATASET) -> pd.DataFrame:
    return pd.read_csv(input_path)


def prepare_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[
        [
            "text",
            "label",
            "wn_avg_synset_count",
            "wn_avg_synonym_count",
            "wn_coverage_ratio",
            "wn_avg_hypernym_depth",
        ]
    ].copy()

    model_df["text"] = model_df["text"].fillna("").astype(str)
    model_df["label"] = model_df["label"].astype(int)
    model_df.insert(0, "example_id", [f"hc3_ex_{i:06d}" for i in range(len(model_df))])
    return model_df


def make_splits(
    model_df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        model_df,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=random_state,
        stratify=model_df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return (
        train_df.sort_values("example_id").reset_index(drop=True),
        val_df.sort_values("example_id").reset_index(drop=True),
        test_df.sort_values("example_id").reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Path]:
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    return {
        "train": TRAIN_OUTPUT,
        "val": VAL_OUTPUT,
        "test": TEST_OUTPUT,
    }


def save_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path = SPLIT_SUMMARY_OUTPUT,
) -> Path:
    summary = {
        "random_seed": RANDOM_SEED,
        "row_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_counts": {
            "train": train_df["label"].value_counts().sort_index().to_dict(),
            "val": val_df["label"].value_counts().sort_index().to_dict(),
            "test": test_df["label"].value_counts().sort_index().to_dict(),
        },
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def build_and_save_splits(input_path: Path = INPUT_DATASET) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_wordnet_dataset(input_path)
    model_df = prepare_modeling_dataframe(df)
    train_df, val_df, test_df = make_splits(model_df)
    save_splits(train_df, val_df, test_df)
    save_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


def main() -> None:
    train_df, val_df, test_df = build_and_save_splits()

    print("Saved split files:")
    print(f"- train: {TRAIN_OUTPUT}")
    print(f"- val:   {VAL_OUTPUT}")
    print(f"- test:  {TEST_OUTPUT}")
    print(f"- summary: {SPLIT_SUMMARY_OUTPUT}")
    print()
    print("Row counts")
    print(f"- train: {len(train_df)}")
    print(f"- val:   {len(val_df)}")
    print(f"- test:  {len(test_df)}")
    print()
    print("Train label counts")
    print(train_df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

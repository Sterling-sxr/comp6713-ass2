from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
INPUT_DATASET = PROCESSED_ROOT / "unified_binary_classification.csv"
OUTPUT_DATASET = PROCESSED_ROOT / "unified_binary_classification_wordnet.csv"

FEATURE_COLUMNS = [
    "wn_avg_synset_count",
    "wn_avg_synonym_count",
    "wn_coverage_ratio",
    "wn_avg_hypernym_depth",
]

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LEMMATIZER = WordNetLemmatizer()
CONTENT_TAG_PREFIXES = ("NN", "VB", "JJ", "RB")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def is_content_pos(tag: str) -> bool:
    return tag.startswith(CONTENT_TAG_PREFIXES)


def penn_to_wordnet_pos(tag: str) -> str | None:
    if tag.startswith("NN"):
        return wn.NOUN
    if tag.startswith("VB"):
        return wn.VERB
    if tag.startswith("JJ"):
        return wn.ADJ
    if tag.startswith("RB"):
        return wn.ADV
    return None


@lru_cache(maxsize=50000)
def get_wordnet_stats(lemma: str, wn_pos: str) -> tuple[int, int, int]:
    synsets = wn.synsets(lemma, pos=wn_pos)
    synset_count = len(synsets)

    if not synsets:
        return 0, 0, 0

    synonym_count = len({item.name() for synset in synsets for item in synset.lemmas()})
    hypernym_depth = synsets[0].max_depth()
    return synset_count, synonym_count, hypernym_depth


def extract_content_words(text: str) -> list[tuple[str, str]]:
    tokens = tokenize_text(text)
    if not tokens:
        return []

    tagged_tokens = pos_tag(tokens)
    content_words: list[tuple[str, str]] = []

    for token, tag in tagged_tokens:
        if not is_content_pos(tag):
            continue

        wn_pos = penn_to_wordnet_pos(tag)
        if wn_pos is None:
            continue

        lemma = LEMMATIZER.lemmatize(token, pos=wn_pos)
        if lemma:
            content_words.append((lemma, wn_pos))

    return content_words


def compute_wordnet_features(text: str) -> dict[str, float]:
    content_words = extract_content_words(text)
    total_content_words = len(content_words)

    if total_content_words == 0:
        return {
            "wn_avg_synset_count": 0.0,
            "wn_avg_synonym_count": 0.0,
            "wn_coverage_ratio": 0.0,
            "wn_avg_hypernym_depth": 0.0,
        }

    synset_counts: list[int] = []
    synonym_counts: list[int] = []
    hypernym_depths: list[int] = []
    covered_count = 0

    for lemma, wn_pos in content_words:
        synset_count, synonym_count, hypernym_depth = get_wordnet_stats(lemma, wn_pos)
        synset_counts.append(synset_count)
        synonym_counts.append(synonym_count)

        if synset_count > 0:
            covered_count += 1
            hypernym_depths.append(hypernym_depth)

    return {
        "wn_avg_synset_count": sum(synset_counts) / total_content_words,
        "wn_avg_synonym_count": sum(synonym_counts) / total_content_words,
        "wn_coverage_ratio": covered_count / total_content_words,
        "wn_avg_hypernym_depth": (
            sum(hypernym_depths) / len(hypernym_depths) if hypernym_depths else 0.0
        ),
    }


def load_processed_dataset(input_path: Path = INPUT_DATASET) -> pd.DataFrame:
    return pd.read_csv(input_path)


def add_wordnet_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    feature_df = df[text_column].fillna("").astype(str).apply(compute_wordnet_features).apply(pd.Series)
    return pd.concat([df.copy(), feature_df], axis=1)


def save_enriched_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_DATASET) -> Path:
    df.to_csv(output_path, index=False)
    return output_path


def build_wordnet_feature_dataset(
    input_path: Path = INPUT_DATASET,
    output_path: Path = OUTPUT_DATASET,
) -> pd.DataFrame:
    df = load_processed_dataset(input_path)
    enriched_df = add_wordnet_features(df)
    save_enriched_dataset(enriched_df, output_path)
    return enriched_df


def main() -> None:
    print(f"Loading processed dataset from: {INPUT_DATASET}")
    enriched_df = build_wordnet_feature_dataset()
    print(f"Saved enriched dataset to: {OUTPUT_DATASET}")
    print()
    print(enriched_df[["record_id", "label", *FEATURE_COLUMNS]].head(5).to_string(index=False))
    print()
    print("Missing value check")
    print(enriched_df[FEATURE_COLUMNS].isna().sum().to_string())
    print()
    print("Feature summary")
    print(enriched_df[FEATURE_COLUMNS].describe().to_string())


if __name__ == "__main__":
    main()

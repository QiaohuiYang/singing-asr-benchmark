"""
Data integration pipeline for the Singing ASR Benchmark.

This script:
1. Loads per-recording ASR metrics.
2. Loads acoustic features.
3. Validates identifiers and input columns.
4. Pairs speech and singing recordings by utterance.
5. Computes singing-related ASR degradation (ΔWER and ΔCER).
6. Retains singing-specific acoustic variables for exploratory analysis.
7. Saves the final paired analysis table.

Inputs:
    results/metrics.csv
    results/acoustic_features.csv

Output:
    results/final_analysis.csv

Usage:
    python analyze_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

METRICS_FILE = (
    RESULTS_DIR
    / "metrics.csv"
)

FEATURES_FILE = (
    RESULTS_DIR
    / "acoustic_features.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_analysis.csv"
)

EXPECTED_RECORDINGS = 36
EXPECTED_PAIRS = 18


# ============================================================
# Data loading
# ============================================================

def load_input_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ASR metrics and acoustic features."""

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found:\n{METRICS_FILE}\n\n"
            "Please run evaluate.py first."
        )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Acoustic feature file not found:\n{FEATURES_FILE}\n\n"
            "Please run extract_features.py first."
        )

    metrics = pd.read_csv(
        METRICS_FILE
    )

    features = pd.read_csv(
        FEATURES_FILE
    )

    return metrics, features


# ============================================================
# Validation
# ============================================================

def validate_columns(
    metrics: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Validate required columns in both input datasets."""

    required_metrics = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
        "WER",
        "CER",
    ]

    required_features = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
        "duration_sec",
        "mean_f0_hz",
        "median_f0_hz",
        "std_f0_hz",
        "f0_range_hz",
        "mean_rms",
    ]

    missing_metrics = [
        column
        for column in required_metrics
        if column not in metrics.columns
    ]

    missing_features = [
        column
        for column in required_features
        if column not in features.columns
    ]

    if missing_metrics:
        raise ValueError(
            "Missing columns in metrics.csv:\n"
            + "\n".join(missing_metrics)
        )

    if missing_features:
        raise ValueError(
            "Missing columns in acoustic_features.csv:\n"
            + "\n".join(missing_features)
        )


def validate_recordings(
    metrics: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Check expected recording counts."""

    if len(metrics) != EXPECTED_RECORDINGS:
        print(
            "WARNING: "
            f"metrics.csv contains {len(metrics)} rows; "
            f"expected {EXPECTED_RECORDINGS}."
        )

    if len(features) != EXPECTED_RECORDINGS:
        print(
            "WARNING: "
            f"acoustic_features.csv contains {len(features)} rows; "
            f"expected {EXPECTED_RECORDINGS}."
        )


def validate_unique_keys(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Check that each recording is uniquely identified by:
    utterance_id + language + mode + length_group.
    """

    key_columns = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
    ]

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = (
            df.loc[
                duplicate_mask,
                key_columns
            ]
            .drop_duplicates()
        )

        raise ValueError(
            f"Duplicate recording keys found in "
            f"{dataset_name}:\n"
            f"{duplicates.to_string(index=False)}"
        )


# ============================================================
# Merge input datasets
# ============================================================

def merge_datasets(
    metrics: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ASR metrics with acoustic features."""

    merge_keys = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
    ]

    merged = pd.merge(
        metrics,
        features,
        on=merge_keys,
        how="inner",
        suffixes=(
            "",
            "_feature",
        ),
        validate="one_to_one",
    )

    return merged


# ============================================================
# Create paired dataset
# ============================================================

def create_paired_dataset(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair speech and singing recordings by utterance.

    Each final row corresponds to one utterance and contains
    speech metrics, singing metrics, and singing acoustic features.
    """

    speech = (
        merged[
            merged["mode"]
            .astype(str)
            .str.lower()
            == "speech"
        ]
        .copy()
    )

    singing = (
        merged[
            merged["mode"]
            .astype(str)
            .str.lower()
            == "singing"
        ]
        .copy()
    )

    print(
        f"Speech recordings:  {len(speech)}"
    )

    print(
        f"Singing recordings: {len(singing)}"
    )

    # --------------------------------------------------------
    # Rename speech-specific columns
    # --------------------------------------------------------

    speech = speech.rename(
        columns={
            "WER": "WER_speech",
            "CER": "CER_speech",

            "substitutions":
                "substitutions_speech",

            "deletions":
                "deletions_speech",

            "insertions":
                "insertions_speech",

            "duration_sec":
                "duration_speech",

            "mean_f0_hz":
                "mean_f0_speech",

            "median_f0_hz":
                "median_f0_speech",

            "std_f0_hz":
                "std_f0_speech",

            "f0_range_hz":
                "f0_range_speech",

            "mean_rms":
                "rms_speech",
        }
    )

    # --------------------------------------------------------
    # Rename singing-specific columns
    # --------------------------------------------------------

    singing = singing.rename(
        columns={
            "WER": "WER_singing",
            "CER": "CER_singing",

            "substitutions":
                "substitutions_singing",

            "deletions":
                "deletions_singing",

            "insertions":
                "insertions_singing",

            "duration_sec":
                "duration_singing",

            "mean_f0_hz":
                "mean_f0_singing",

            "median_f0_hz":
                "median_f0_singing",

            "std_f0_hz":
                "std_f0_singing",

            "f0_range_hz":
                "f0_range_singing",

            "mean_rms":
                "rms_singing",
        }
    )

    # --------------------------------------------------------
    # Pair speech and singing
    # --------------------------------------------------------

    pair_keys = [
        "utterance_id",
        "language",
        "length_group",
    ]

    paired = pd.merge(
        speech,
        singing,
        on=pair_keys,
        how="inner",
        suffixes=(
            "_speech",
            "_singing",
        ),
        validate="one_to_one",
    )

    return paired


# ============================================================
# Calculate degradation metrics
# ============================================================

def calculate_degradation_metrics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate WER/CER degradation from speech to singing."""

    paired["delta_WER"] = (
        paired["WER_singing"]
        - paired["WER_speech"]
    )

    paired["delta_CER"] = (
        paired["CER_singing"]
        - paired["CER_speech"]
    )

    # Relative WER change is undefined when speech WER = 0.
    # We therefore return NaN in those cases.
    paired["relative_WER_change"] = np.where(
        paired["WER_speech"] > 0,
        (
            paired["WER_singing"]
            - paired["WER_speech"]
        )
        / paired["WER_speech"],
        np.nan,
    )

    return paired


# ============================================================
# Add singing-specific acoustic variables
# ============================================================

def add_singing_acoustic_variables(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create simplified names for singing acoustic variables.

    These names are used by the downstream exploratory analysis.
    """

    paired["singing_duration"] = (
        paired["duration_singing"]
    )

    paired["singing_mean_f0"] = (
        paired["mean_f0_singing"]
    )

    paired["singing_f0_std"] = (
        paired["std_f0_singing"]
    )

    paired["singing_f0_range"] = (
        paired["f0_range_singing"]
    )

    paired["singing_rms"] = (
        paired["rms_singing"]
    )

    return paired


# ============================================================
# Select final columns
# ============================================================

def create_final_table(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Select the compact final analysis table."""

    final_columns = [
        "utterance_id",
        "language",
        "length_group",

        "WER_speech",
        "WER_singing",
        "delta_WER",

        "CER_speech",
        "CER_singing",
        "delta_CER",

        "substitutions_speech",
        "deletions_speech",
        "insertions_speech",

        "substitutions_singing",
        "deletions_singing",
        "insertions_singing",

        "duration_speech",
        "duration_singing",

        "singing_mean_f0",
        "singing_f0_std",
        "singing_f0_range",
        "singing_rms",
    ]

    missing_columns = [
        column
        for column in final_columns
        if column not in paired.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns needed for final analysis:\n"
            + "\n".join(missing_columns)
        )

    return paired[
        final_columns
    ].copy()


# ============================================================
# Save final dataset
# ============================================================

def save_final_table(
    final_df: pd.DataFrame,
) -> None:
    """Save the final paired analysis table."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Print summaries
# ============================================================

def print_paired_results(
    final_df: pd.DataFrame,
) -> None:
    """Print key paired-analysis results."""

    print("\n" + "=" * 70)
    print("PAIRED ANALYSIS")
    print("=" * 70)

    display_columns = [
        "utterance_id",
        "language",
        "length_group",
        "WER_speech",
        "WER_singing",
        "delta_WER",
    ]

    print(
        final_df[
            display_columns
        ]
        .round(3)
        .to_string(index=False)
    )


def print_overall_results(
    final_df: pd.DataFrame,
) -> None:
    """Print overall descriptive statistics."""

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Number of paired utterances: "
        f"{len(final_df)}"
    )

    print(
        f"Mean speech WER:  "
        f"{final_df['WER_speech'].mean():.3f}"
    )

    print(
        f"Mean singing WER: "
        f"{final_df['WER_singing'].mean():.3f}"
    )

    print(
        f"Mean ΔWER:         "
        f"{final_df['delta_WER'].mean():.3f}"
    )

    print(
        f"Median ΔWER:       "
        f"{final_df['delta_WER'].median():.3f}"
    )


def print_language_results(
    final_df: pd.DataFrame,
) -> None:
    """Print descriptive results by language."""

    print("\n" + "=" * 70)
    print("RESULTS BY LANGUAGE")
    print("=" * 70)

    language_summary = (
        final_df
        .groupby("language")
        .agg(
            n=(
                "delta_WER",
                "size",
            ),
            mean_WER_speech=(
                "WER_speech",
                "mean",
            ),
            mean_WER_singing=(
                "WER_singing",
                "mean",
            ),
            mean_delta_WER=(
                "delta_WER",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        language_summary
        .round(3)
        .to_string(index=False)
    )


def print_length_results(
    final_df: pd.DataFrame,
) -> None:
    """Print descriptive results by utterance length."""

    print("\n" + "=" * 70)
    print("RESULTS BY LENGTH")
    print("=" * 70)

    length_order = [
        "short",
        "medium",
        "long",
    ]

    length_summary = (
        final_df
        .assign(
            length_clean=(
                final_df["length_group"]
                .astype(str)
                .str.lower()
            )
        )
        .groupby("length_clean")
        .agg(
            n=(
                "delta_WER",
                "size",
            ),
            mean_WER_speech=(
                "WER_speech",
                "mean",
            ),
            mean_WER_singing=(
                "WER_singing",
                "mean",
            ),
            mean_delta_WER=(
                "delta_WER",
                "mean",
            ),
        )
        .reindex(length_order)
        .reset_index()
    )

    print(
        length_summary
        .round(3)
        .to_string(index=False)
    )


def print_effect_direction(
    final_df: pd.DataFrame,
) -> None:
    """Print the number of pairs moving in each direction."""

    delta = final_df["delta_WER"]

    num_higher = (
        (delta > 0).sum()
    )

    num_same = (
        (delta == 0).sum()
    )

    num_lower = (
        (delta < 0).sum()
    )

    print("\n" + "=" * 70)
    print("DIRECTION OF SINGING EFFECT")
    print("=" * 70)

    print(
        f"Singing WER > Speech WER: "
        f"{num_higher}"
    )

    print(
        f"Singing WER = Speech WER: "
        f"{num_same}"
    )

    print(
        f"Singing WER < Speech WER: "
        f"{num_lower}"
    )


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    """Run the complete data integration pipeline."""

    print("=" * 70)
    print("SINGING ASR BENCHMARK")
    print("Data Integration")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    metrics, features = (
        load_input_files()
    )

    print("\nInput data")
    print("-" * 70)

    print(
        f"Metrics rows:   "
        f"{len(metrics)}"
    )

    print(
        f"Features rows:  "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_columns(
        metrics,
        features,
    )

    validate_recordings(
        metrics,
        features,
    )

    validate_unique_keys(
        metrics,
        "metrics.csv",
    )

    validate_unique_keys(
        features,
        "acoustic_features.csv",
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_datasets(
        metrics,
        features,
    )

    print(
        f"Merged rows:    "
        f"{len(merged)}"
    )

    if len(merged) != EXPECTED_RECORDINGS:

        print(
            "WARNING: "
            f"Expected {EXPECTED_RECORDINGS} merged "
            f"recordings."
        )

    # --------------------------------------------------------
    # Pair
    # --------------------------------------------------------

    paired = create_paired_dataset(
        merged
    )

    if len(paired) != EXPECTED_PAIRS:

        print(
            "WARNING: "
            f"Expected {EXPECTED_PAIRS} paired utterances, "
            f"but created {len(paired)}."
        )

    # --------------------------------------------------------
    # Calculate degradation
    # --------------------------------------------------------

    paired = calculate_degradation_metrics(
        paired
    )

    # --------------------------------------------------------
    # Add acoustic variables
    # --------------------------------------------------------

    paired = add_singing_acoustic_variables(
        paired
    )

    # --------------------------------------------------------
    # Final compact table
    # --------------------------------------------------------

    final_df = create_final_table(
        paired
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_final_table(
        final_df
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_paired_results(
        final_df
    )

    print_overall_results(
        final_df
    )

    print_language_results(
        final_df
    )

    print_length_results(
        final_df
    )

    print_effect_direction(
        final_df
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA INTEGRATION COMPLETE")
    print("=" * 70)

    print(
        f"Final paired analysis: "
        f"{len(final_df)} rows"
    )

    print(
        f"Saved to:\n"
        f"  {OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
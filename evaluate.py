"""
Evaluation pipeline for the Singing ASR Benchmark.

This script:
1. Loads Whisper transcriptions.
2. Computes WER and CER for every recording.
3. Decomposes word-level errors into substitutions, deletions,
   and insertions.
4. Produces per-recording metrics and condition-level summaries.

Input:
    results/transcriptions.csv

Outputs:
    results/metrics.csv
    results/summary_by_condition.csv

Usage:
    python evaluate.py
"""

from pathlib import Path

import pandas as pd
from jiwer import cer, process_words, wer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "transcriptions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "summary_by_condition.csv"
)

EXPECTED_RECORDINGS = 36


# ============================================================
# Utility functions
# ============================================================

def load_transcriptions() -> pd.DataFrame:
    """Load and validate the transcription results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find input file:\n{INPUT_FILE}\n\n"
            "Please run run_whisper.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
        "reference",
        "prediction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    return df


def calculate_record_metrics(
    row: pd.Series,
) -> dict:
    """
    Calculate WER, CER, and word-level error counts
    for one recording.
    """

    reference = str(row["reference"])
    prediction = str(row["prediction"])

    word_error_rate = wer(
        reference,
        prediction,
    )

    character_error_rate = cer(
        reference,
        prediction,
    )

    detail = process_words(
        reference,
        prediction,
    )

    return {
        "utterance_id": row["utterance_id"],
        "language": row["language"],
        "language_code": row.get(
            "language_code",
            "",
        ),
        "mode": row["mode"],
        "length_group": row["length_group"],
        "filename": row.get(
            "filename",
            "",
        ),
        "reference": reference,
        "prediction": prediction,
        "WER": word_error_rate,
        "CER": character_error_rate,
        "substitutions": detail.substitutions,
        "deletions": detail.deletions,
        "insertions": detail.insertions,
    }


def evaluate_all(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics for every recording."""

    results = []

    for _, row in df.iterrows():

        metrics = calculate_record_metrics(row)

        results.append(metrics)

    return pd.DataFrame(results)


def create_condition_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary statistics grouped by language,
    mode, and utterance length.
    """

    summary_df = (
        metrics_df
        .groupby(
            [
                "language",
                "mode",
                "length_group",
            ],
            as_index=False,
        )
        .agg(
            n=("WER", "size"),
            mean_WER=("WER", "mean"),
            median_WER=("WER", "median"),
            mean_CER=("CER", "mean"),
            median_CER=("CER", "median"),
            mean_substitutions=(
                "substitutions",
                "mean",
            ),
            mean_deletions=(
                "deletions",
                "mean",
            ),
            mean_insertions=(
                "insertions",
                "mean",
            ),
        )
    )

    length_order = {
        "short": 0,
        "medium": 1,
        "long": 2,
    }

    summary_df["length_order"] = (
        summary_df["length_group"]
        .astype(str)
        .str.lower()
        .map(length_order)
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "language",
                "mode",
                "length_order",
            ]
        )
        .drop(
            columns=["length_order"]
        )
        .reset_index(drop=True)
    )

    return summary_df


def save_results(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Save evaluation outputs to the results directory."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Main evaluation pipeline
# ============================================================

def main() -> None:
    """Run the complete evaluation pipeline."""

    print("=" * 70)
    print("SINGING ASR BENCHMARK")
    print("Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    df = load_transcriptions()

    print("\nInput data check")
    print("-" * 70)
    print(
        f"Number of recordings: {len(df)}"
    )

    if len(df) != EXPECTED_RECORDINGS:

        print(
            "WARNING: "
            f"Expected {EXPECTED_RECORDINGS} recordings, "
            f"but found {len(df)}."
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics_df = evaluate_all(df)

    # --------------------------------------------------------
    # Create condition-level summary
    # --------------------------------------------------------

    summary_df = create_condition_summary(
        metrics_df
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_results(
        metrics_df,
        summary_df,
    )

    # --------------------------------------------------------
    # Display per-recording results
    # --------------------------------------------------------

    display_columns = [
        "utterance_id",
        "language",
        "mode",
        "length_group",
        "WER",
        "CER",
        "substitutions",
        "deletions",
        "insertions",
    ]

    print("\n" + "=" * 70)
    print("PER-RECORDING METRICS")
    print("=" * 70)

    print(
        metrics_df[display_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Display condition summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY BY CONDITION")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final output information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Per-recording metrics:\n"
        f"  {METRICS_FILE}"
    )

    print(
        f"\nCondition summary:\n"
        f"  {SUMMARY_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
from jiwer import wer, cer, process_words


# ============================================================
# 1. File paths
# ============================================================

INPUT_FILE = Path("results/transcriptions.csv")
OUTPUT_DIR = Path("results")

METRICS_FILE = OUTPUT_DIR / "metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary_by_condition.csv"


# ============================================================
# 2. Load transcription results
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("INPUT DATA CHECK")
print("=" * 70)

print(f"Number of recordings: {len(df)}")

if len(df) != 36:
    print("WARNING: Expected 36 recordings.")


required_columns = [
    "utterance_id",
    "language",
    "mode",
    "length_group",
    "reference",
    "prediction",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. Calculate metrics for every recording
# ============================================================

results = []

for _, row in df.iterrows():

    reference = str(row["reference"])
    prediction = str(row["prediction"])

    # WER and CER
    word_error_rate = wer(reference, prediction)
    char_error_rate = cer(reference, prediction)

    # Detailed word-level error breakdown
    detail = process_words(
        reference,
        prediction
    )

    results.append({
        "utterance_id": row["utterance_id"],
        "language": row["language"],
        "language_code": row.get("language_code", ""),
        "mode": row["mode"],
        "length_group": row["length_group"],
        "filename": row.get("filename", ""),
        "reference": reference,
        "prediction": prediction,

        "WER": word_error_rate,
        "CER": char_error_rate,

        "substitutions": detail.substitutions,
        "deletions": detail.deletions,
        "insertions": detail.insertions,
    })


metrics_df = pd.DataFrame(results)


# ============================================================
# 4. Save per-recording metrics
# ============================================================

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Create condition-level summary
# ============================================================

summary_df = (
    metrics_df
    .groupby(
        ["language", "mode", "length_group"],
        as_index=False
    )
    .agg(
        n=("WER", "size"),
        mean_WER=("WER", "mean"),
        median_WER=("WER", "median"),
        mean_CER=("CER", "mean"),
        median_CER=("CER", "median"),
        mean_substitutions=("substitutions", "mean"),
        mean_deletions=("deletions", "mean"),
        mean_insertions=("insertions", "mean"),
    )
)


# ============================================================
# 6. Sort summary nicely
# ============================================================

length_order = {
    "short": 0,
    "medium": 1,
    "long": 2,
}

summary_df["length_order"] = (
    summary_df["length_group"].map(length_order)
)

summary_df = (
    summary_df
    .sort_values(
        ["language", "mode", "length_order"]
    )
    .drop(columns=["length_order"])
)


# ============================================================
# 7. Save summary
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. Print results
# ============================================================

print("\n" + "=" * 70)
print("PER-RECORDING METRICS")
print("=" * 70)

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

print(
    metrics_df[display_columns]
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("SUMMARY BY CONDITION")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)


# ============================================================
# 9. Final file locations
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Per-recording metrics:")
print(f"  {METRICS_FILE}")

print(f"\nCondition summary:")
print(f"  {SUMMARY_FILE}")
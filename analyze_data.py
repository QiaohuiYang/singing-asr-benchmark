from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. File paths
# ============================================================

METRICS_FILE = Path("results/metrics.csv")
FEATURES_FILE = Path("results/acoustic_features.csv")

OUTPUT_FILE = Path("results/final_analysis.csv")


# ============================================================
# 2. Load data
# ============================================================

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find: {METRICS_FILE}"
    )

if not FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find: {FEATURES_FILE}"
    )


metrics = pd.read_csv(METRICS_FILE)
features = pd.read_csv(FEATURES_FILE)


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print(f"Metrics rows:   {len(metrics)}")
print(f"Features rows:  {len(features)}")


# ============================================================
# 3. Check identifiers
# ============================================================

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
    col for col in required_metrics
    if col not in metrics.columns
]

missing_features = [
    col for col in required_features
    if col not in features.columns
]

if missing_metrics:
    raise ValueError(
        f"Missing columns in metrics.csv: {missing_metrics}"
    )

if missing_features:
    raise ValueError(
        f"Missing columns in acoustic_features.csv: {missing_features}"
    )


# ============================================================
# 4. Merge metrics and acoustic features
# ============================================================

merged = pd.merge(
    metrics,
    features,
    on=[
        "utterance_id",
        "language",
        "mode",
        "length_group",
    ],
    how="inner",
    suffixes=("", "_feature"),
)


print(f"Merged rows:   {len(merged)}")


if len(merged) != 36:
    print(
        "WARNING: Expected 36 merged recordings."
    )


# ============================================================
# 5. Separate speech and singing
# ============================================================

speech = (
    merged[
        merged["mode"].str.lower() == "speech"
    ]
    .copy()
)

singing = (
    merged[
        merged["mode"].str.lower() == "singing"
    ]
    .copy()
)


print(f"Speech recordings:   {len(speech)}")
print(f"Singing recordings:  {len(singing)}")


# ============================================================
# 6. Rename mode-specific columns
# ============================================================

speech = speech.rename(
    columns={
        "WER": "WER_speech",
        "CER": "CER_speech",
        "substitutions": "substitutions_speech",
        "deletions": "deletions_speech",
        "insertions": "insertions_speech",
        "duration_sec": "duration_speech",
        "mean_f0_hz": "mean_f0_speech",
        "median_f0_hz": "median_f0_speech",
        "std_f0_hz": "std_f0_speech",
        "f0_range_hz": "f0_range_speech",
        "mean_rms": "rms_speech",
    }
)

singing = singing.rename(
    columns={
        "WER": "WER_singing",
        "CER": "CER_singing",
        "substitutions": "substitutions_singing",
        "deletions": "deletions_singing",
        "insertions": "insertions_singing",
        "duration_sec": "duration_singing",
        "mean_f0_hz": "mean_f0_singing",
        "median_f0_hz": "median_f0_singing",
        "std_f0_hz": "std_f0_singing",
        "f0_range_hz": "f0_range_singing",
        "mean_rms": "rms_singing",
    }
)


# ============================================================
# 7. Create paired dataset
# ============================================================

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
    suffixes=("_speech", "_singing"),
)


# ============================================================
# 8. Calculate degradation metrics
# ============================================================

paired["delta_WER"] = (
    paired["WER_singing"]
    - paired["WER_speech"]
)

paired["delta_CER"] = (
    paired["CER_singing"]
    - paired["CER_speech"]
)


paired["relative_WER_change"] = np.where(
    paired["WER_speech"] > 0,
    (
        paired["WER_singing"]
        - paired["WER_speech"]
    ) / paired["WER_speech"],
    np.nan,
)


# ============================================================
# 9. Calculate singing-specific acoustic variables
# ============================================================

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


# ============================================================
# 10. Keep clean final columns
# ============================================================

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


final_df = paired[final_columns].copy()


# ============================================================
# 11. Add human-readable language labels
# ============================================================

# Keep existing language column unchanged.
# This section is intentionally simple so the dataset
# remains easy to interpret.


# ============================================================
# 12. Save final analysis table
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. Print key results
# ============================================================

print("\n" + "=" * 70)
print("PAIRED ANALYSIS")
print("=" * 70)

print(
    final_df[
        [
            "utterance_id",
            "language",
            "length_group",
            "WER_speech",
            "WER_singing",
            "delta_WER",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 14. Overall descriptive statistics
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Mean speech WER:   "
    f"{final_df['WER_speech'].mean():.3f}"
)

print(
    f"Mean singing WER:  "
    f"{final_df['WER_singing'].mean():.3f}"
)

print(
    f"Mean ΔWER:          "
    f"{final_df['delta_WER'].mean():.3f}"
)


# ============================================================
# 15. Results by language
# ============================================================

print("\n" + "=" * 70)
print("RESULTS BY LANGUAGE")
print("=" * 70)

language_summary = (
    final_df
    .groupby("language")
    .agg(
        n=("delta_WER", "size"),
        mean_WER_speech=("WER_speech", "mean"),
        mean_WER_singing=("WER_singing", "mean"),
        mean_delta_WER=("delta_WER", "mean"),
    )
    .reset_index()
)

print(
    language_summary
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 16. Results by length
# ============================================================

print("\n" + "=" * 70)
print("RESULTS BY LENGTH")
print("=" * 70)

length_order = ["short", "medium", "long"]

length_summary = (
    final_df
    .groupby("length_group")
    .agg(
        n=("delta_WER", "size"),
        mean_WER_speech=("WER_speech", "mean"),
        mean_WER_singing=("WER_singing", "mean"),
        mean_delta_WER=("delta_WER", "mean"),
    )
    .reindex(length_order)
    .reset_index()
)

print(
    length_summary
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 17. Count direction of singing effect
# ============================================================

num_higher = (
    (final_df["delta_WER"] > 0)
    .sum()
)

num_same = (
    (final_df["delta_WER"] == 0)
    .sum()
)

num_lower = (
    (final_df["delta_WER"] < 0)
    .sum()
)

print("\n" + "=" * 70)
print("DIRECTION OF SINGING EFFECT")
print("=" * 70)

print(f"Singing WER > Speech WER: {num_higher}")
print(f"Singing WER = Speech WER: {num_same}")
print(f"Singing WER < Speech WER: {num_lower}")


# ============================================================
# 18. Final output
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Saved final analysis to:")
print(f"  {OUTPUT_FILE}")
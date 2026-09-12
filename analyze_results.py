from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_ANALYSIS = RESULTS_DIR / "final_analysis.csv"
METRICS_FILE = RESULTS_DIR / "metrics.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata_ground_truth.csv"


# ============================================================
# 2. Load data
# ============================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

if not FINAL_ANALYSIS.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{FINAL_ANALYSIS}\n\n"
        "Please make sure final_analysis.csv is inside results/."
    )

df = pd.read_csv(FINAL_ANALYSIS)

print(f"Final analysis rows: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. Required columns
# ============================================================

required_columns = [
    "utterance_id",
    "language",
    "length_group",
    "WER_speech",
    "WER_singing",
    "CER_speech",
    "CER_singing",
    "delta_WER",
    "delta_CER",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# 4. Standardize categorical values
# ============================================================

df["language_clean"] = (
    df["language"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["length_clean"] = (
    df["length_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 5. Derived variables
# ============================================================

df["delta_WER"] = (
    df["WER_singing"] - df["WER_speech"]
)

df["delta_CER"] = (
    df["CER_singing"] - df["CER_speech"]
)


# ============================================================
# 6. Overall summary
# ============================================================

summary_rows = []

for metric_name, speech_col, singing_col, delta_col in [
    ("WER", "WER_speech", "WER_singing", "delta_WER"),
    ("CER", "CER_speech", "CER_singing", "delta_CER"),
]:

    summary_rows.append({
        "metric": metric_name,
        "n_pairs": len(df),

        "speech_mean": df[speech_col].mean(),
        "singing_mean": df[singing_col].mean(),
        "mean_delta": df[delta_col].mean(),

        "speech_median": df[speech_col].median(),
        "singing_median": df[singing_col].median(),
        "median_delta": df[delta_col].median(),

        "speech_std": df[speech_col].std(),
        "singing_std": df[singing_col].std(),
    })

summary_df = pd.DataFrame(summary_rows)

summary_file = RESULTS_DIR / "overall_summary.csv"
summary_df.to_csv(summary_file, index=False)

print("\nOverall summary:")
print(summary_df.round(4))


# ============================================================
# 7. Wilcoxon signed-rank tests
# ============================================================

def wilcoxon_report(speech, singing):
    """
    Paired Wilcoxon signed-rank test.

    Returns:
        n_pairs
        n_nonzero
        statistic
        p_value
        effect_r
    """

    speech = np.asarray(speech, dtype=float)
    singing = np.asarray(singing, dtype=float)

    diff = singing - speech

    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) == 0:
        return {
            "n_pairs": len(diff),
            "n_nonzero": 0,
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_r": 0.0,
        }

    result = wilcoxon(
        speech,
        singing,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    n = len(nonzero_diff)

    mean_w = n * (n + 1) / 4
    sd_w = np.sqrt(
        n * (n + 1) * (2 * n + 1) / 24
    )

    z = (
        (result.statistic - mean_w) / sd_w
        if sd_w > 0
        else 0
    )

    effect_r = abs(z) / np.sqrt(n)

    return {
        "n_pairs": len(diff),
        "n_nonzero": n,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "effect_r": effect_r,
    }


stat_rows = []

for metric_name, speech_col, singing_col in [
    ("WER", "WER_speech", "WER_singing"),
    ("CER", "CER_speech", "CER_singing"),
]:

    result = wilcoxon_report(
        df[speech_col],
        df[singing_col],
    )

    stat_rows.append({
        "metric": metric_name,
        **result,
    })

stats_df = pd.DataFrame(stat_rows)

stats_file = RESULTS_DIR / "statistical_tests.csv"
stats_df.to_csv(stats_file, index=False)

print("\nWilcoxon signed-rank tests:")
print(stats_df.round(4))


# ============================================================
# ============================================================
# 8. Figure 1
# Ranked paired WER (Dumbbell Plot)
# ============================================================

print("\nCreating Figure 1...")

# Sort utterances by singing-related degradation
plot_df = df.sort_values(
    "delta_WER",
    ascending=True
).copy()

# Create readable labels
plot_df["label"] = (
    plot_df["language_clean"].str[:2].str.upper()
    + "_"
    + plot_df["utterance_id"].astype(str)
)

# Reverse order so the largest degradation is at the top
plot_df = plot_df.reset_index(drop=True)

y_positions = np.arange(len(plot_df))

plt.figure(figsize=(10, 9))

# ------------------------------------------------------------
# Draw connecting lines
# ------------------------------------------------------------

for y, (_, row) in zip(
    y_positions,
    plot_df.iterrows()
):

    plt.plot(
        [
            row["WER_speech"],
            row["WER_singing"],
        ],
        [y, y],
        linewidth=1.5,
        alpha=0.45,
    )

# ------------------------------------------------------------
# Plot speech points
# ------------------------------------------------------------

plt.scatter(
    plot_df["WER_speech"],
    y_positions,
    s=65,
    marker="o",
    label="Speech",
    zorder=3,
)

# ------------------------------------------------------------
# Plot singing points
# ------------------------------------------------------------

plt.scatter(
    plot_df["WER_singing"],
    y_positions,
    s=75,
    marker="D",
    label="Singing",
    zorder=3,
)

# ------------------------------------------------------------
# Add zero-degradation reference conceptually
# ------------------------------------------------------------

plt.yticks(
    y_positions,
    plot_df["label"],
    fontsize=9,
)

plt.xlabel(
    "Word Error Rate (WER)",
    fontsize=12,
)

plt.ylabel(
    "Utterance",
    fontsize=12,
)

plt.title(
    "Paired WER Change from Speech to Singing",
    fontsize=15,
)

plt.grid(
    axis="x",
    alpha=0.25,
)

plt.legend(
    frameon=True,
)

plt.tight_layout()

fig1 = FIGURES_DIR / "fig1_paired_wer.png"

plt.savefig(
    fig1,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {fig1}")


# ============================================================
# 9. Figure 2
# WER by Language and Mode
# ============================================================

print("\nCreating Figure 2...")

languages = ["english", "french"]

speech_data = []
singing_data = []

for language in languages:

    subset = df[
        df["language_clean"] == language
    ]

    speech_data.append(
        subset["WER_speech"].dropna().values
    )

    singing_data.append(
        subset["WER_singing"].dropna().values
    )


positions = [
    1,
    2,
    4,
    5,
]

data_for_boxplot = [
    speech_data[0],
    singing_data[0],
    speech_data[1],
    singing_data[1],
]

labels = [
    "English\nSpeech",
    "English\nSinging",
    "French\nSpeech",
    "French\nSinging",
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    data_for_boxplot,
    positions=positions,
    widths=0.55,
    patch_artist=False,
    showfliers=True,
)

plt.xticks(
    positions,
    labels,
    fontsize=11,
)

plt.ylabel(
    "Word Error Rate (WER)",
    fontsize=12,
)

plt.title(
    "WER by Language and Mode",
    fontsize=15,
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.xlim(0.4, 5.6)

plt.tight_layout()

fig2 = FIGURES_DIR / "fig2_wer_language.png"

plt.savefig(
    fig2,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {fig2}")


# ============================================================
# 10. Figure 3
# ΔWER by Sentence Length
# Individual points + median
# ============================================================

print("\nCreating Figure 3...")

length_order = [
    "short",
    "medium",
    "long",
]

length_labels = [
    "Short",
    "Medium",
    "Long",
]

plt.figure(figsize=(8, 6))

for x, length_group in enumerate(
    length_order,
    start=1,
):

    values = df[
        df["length_clean"] == length_group
    ]["delta_WER"].dropna().values

    # Small horizontal jitter
    jitter = np.linspace(
        -0.10,
        0.10,
        len(values),
    )

    plt.scatter(
        x + jitter,
        values,
        s=55,
        alpha=0.75,
    )

    # Median
    median_value = np.median(values)

    plt.plot(
        [x - 0.18, x + 0.18],
        [median_value, median_value],
        linewidth=3,
    )

plt.axhline(
    0,
    linewidth=1,
    alpha=0.7,
)

plt.xticks(
    [1, 2, 3],
    length_labels,
    fontsize=12,
)

plt.xlabel(
    "Sentence Length",
    fontsize=12,
)

plt.ylabel(
    "ΔWER (Singing − Speech)",
    fontsize=12,
)

plt.title(
    "ASR Degradation by Sentence Length",
    fontsize=15,
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

fig3 = FIGURES_DIR / "fig3_delta_wer_length.png"

plt.savefig(
    fig3,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {fig3}")


# ============================================================
# 11. Figure 4
# Normalized Error Types
# ============================================================

print("\nCreating Figure 4...")

if METRICS_FILE.exists():

    metrics = pd.read_csv(METRICS_FILE)

    print("\nMetrics columns:")
    print(metrics.columns.tolist())

    required_error_cols = [
        "substitutions",
        "deletions",
        "insertions",
        "mode",
    ]

    if all(
        col in metrics.columns
        for col in required_error_cols
    ):

        # ----------------------------------------------------
        # Get reference word counts
        # ----------------------------------------------------

        reference_lengths = None

        # Preferred method: use an existing reference length
        possible_length_cols = [
            "reference_length",
            "ref_length",
            "reference_words",
            "ref_words",
        ]

        for col in possible_length_cols:

            if col in metrics.columns:
                reference_lengths = (
                    metrics[
                        ["utterance_id", col]
                    ]
                    .drop_duplicates(
                        "utterance_id"
                    )
                    .rename(
                        columns={
                            col: "reference_length"
                        }
                    )
                )
                break

        # Otherwise calculate from metadata
        if reference_lengths is None:

            if not METADATA_FILE.exists():

                raise FileNotFoundError(
                    "Could not find metadata_ground_truth.csv "
                    "for reference word counts."
                )

            metadata = pd.read_csv(
                METADATA_FILE
            )

            if (
                "utterance_id" not in metadata.columns
                or "reference" not in metadata.columns
            ):
                raise ValueError(
                    "metadata_ground_truth.csv must contain "
                    "'utterance_id' and 'reference'."
                )

            metadata["reference_length"] = (
                metadata["reference"]
                .astype(str)
                .str.split()
                .str.len()
            )

            reference_lengths = (
                metadata[
                    [
                        "utterance_id",
                        "reference_length",
                    ]
                ]
                .drop_duplicates(
                    "utterance_id"
                )
            )

        # ----------------------------------------------------
        # Merge word counts
        # ----------------------------------------------------

        metrics = metrics.merge(
            reference_lengths,
            on="utterance_id",
            how="left",
        )

        if metrics["reference_length"].isna().any():

            raise ValueError(
                "Some recordings are missing reference word counts."
            )

        # ----------------------------------------------------
        # Normalize each error type
        # ----------------------------------------------------

        error_types = [
            "substitutions",
            "deletions",
            "insertions",
        ]

        for error_type in error_types:

            metrics[
                f"{error_type}_rate"
            ] = (
                metrics[error_type]
                / metrics["reference_length"]
            )

        # ----------------------------------------------------
        # Average error rate by mode
        # ----------------------------------------------------

        error_rate_summary = (
            metrics.groupby("mode")[
                [
                    "substitutions_rate",
                    "deletions_rate",
                    "insertions_rate",
                ]
            ]
            .mean()
        )

        error_rate_summary = error_rate_summary.reindex(
            ["speech", "singing"]
        )

        # Save normalized summary
        error_rate_summary.to_csv(
            RESULTS_DIR / "normalized_error_rates.csv"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        ax = error_rate_summary.T.plot(
            kind="bar",
            figsize=(8, 6),
        )

        ax.set_xlabel(
            "Error Type",
            fontsize=12,
        )

        ax.set_ylabel(
            "Mean Error Rate per Reference Word",
            fontsize=12,
        )

        ax.set_title(
            "Normalized ASR Error Types: Speech vs Singing",
            fontsize=15,
        )

        ax.set_xticklabels(
            [
                "Substitutions",
                "Deletions",
                "Insertions",
            ],
            rotation=0,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        plt.legend(
            title="Mode"
        )

        plt.tight_layout()

        fig4 = FIGURES_DIR / "fig4_error_types.png"

        plt.savefig(
            fig4,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {fig4}")

    else:

        print(
            "\nWarning: metrics.csv is missing "
            "required error columns."
        )

else:

    print(
        "\nWarning: metrics.csv not found. "
        "Skipping Figure 4."
    )


# ============================================================
# 12. Figure 5
# Acoustic features vs ΔWER
# ============================================================

print("\nCreating acoustic analysis figures...")

acoustic_candidates = [

    (
        "duration_singing",
        "Singing Duration (sec)",
        "fig5_duration_vs_delta_wer.png",
    ),

    (
        "singing_mean_f0",
        "Mean F0 in Singing (Hz)",
        "fig5_mean_f0_vs_delta_wer.png",
    ),

    (
        "singing_f0_std",
        "F0 Variability in Singing (Hz)",
        "fig5_f0_std_vs_delta_wer.png",
    ),

    (
        "singing_f0_range",
        "F0 Range in Singing (Hz)",
        "fig5_f0_range_vs_delta_wer.png",
    ),

    (
        "singing_rms",
        "Singing RMS",
        "fig5_rms_vs_delta_wer.png",
    ),
]

correlation_rows = []


for column, x_label, filename in acoustic_candidates:

    if column not in df.columns:

        print(
            f"Skipping {column}: "
            "column not found."
        )

        continue

    subset = df[
        [column, "delta_WER"]
    ].dropna()

    if len(subset) < 3:

        print(
            f"Skipping {column}: "
            "not enough observations."
        )

        continue

    x = subset[column].to_numpy()
    y = subset["delta_WER"].to_numpy()

    correlation = np.corrcoef(
        x,
        y,
    )[0, 1]

    correlation_rows.append({
        "feature": column,
        "n": len(subset),
        "pearson_r": correlation,
    })

    plt.figure(figsize=(7, 6))

    plt.scatter(
        x,
        y,
        s=60,
        alpha=0.75,
    )

    # Regression line
    if np.std(x) > 0:

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            100,
        )

        y_line = (
            slope * x_line
            + intercept
        )

        plt.plot(
            x_line,
            y_line,
            linewidth=2,
        )

    plt.axhline(
        0,
        linewidth=1,
        alpha=0.7,
    )

    plt.xlabel(
        x_label,
        fontsize=12,
    )

    plt.ylabel(
        "ΔWER (Singing − Speech)",
        fontsize=12,
    )

    plt.title(
        f"{x_label} vs ASR Degradation",
        fontsize=14,
    )

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR / filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_file = (
    RESULTS_DIR
    / "acoustic_correlations.csv"
)

correlation_df.to_csv(
    correlation_file,
    index=False,
)


# ============================================================
# 13. Create analysis summary
# ============================================================

summary_lines = []

summary_lines.append(
    "Singing ASR Benchmark - Exploratory Analysis"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append(
    f"\nNumber of paired utterances: {len(df)}"
)

summary_lines.append(
    "\nOverall performance:"
)

for _, row in summary_df.iterrows():

    summary_lines.append(
        f"\n{row['metric']}:"
        f"\n  Speech mean    = "
        f"{row['speech_mean']:.4f}"
        f"\n  Singing mean   = "
        f"{row['singing_mean']:.4f}"
        f"\n  Mean Δ         = "
        f"{row['mean_delta']:.4f}"
        f"\n  Speech median  = "
        f"{row['speech_median']:.4f}"
        f"\n  Singing median = "
        f"{row['singing_median']:.4f}"
        f"\n  Median Δ       = "
        f"{row['median_delta']:.4f}"
    )


summary_lines.append(
    "\n\nWilcoxon signed-rank tests:"
)

for _, row in stats_df.iterrows():

    summary_lines.append(
        f"\n{row['metric']}:"
        f"\n  n pairs   = "
        f"{int(row['n_pairs'])}"
        f"\n  statistic = "
        f"{row['statistic']:.4f}"
        f"\n  p-value   = "
        f"{row['p_value']:.6f}"
        f"\n  effect r  = "
        f"{row['effect_r']:.4f}"
    )


summary_lines.append(
    "\n\nAcoustic correlations:"
)

if len(correlation_df) > 0:

    for _, row in correlation_df.iterrows():

        summary_lines.append(
            f"\n  {row['feature']}: "
            f"r = {row['pearson_r']:.4f} "
            f"(n={int(row['n'])})"
        )

else:

    summary_lines.append(
        "\n  No acoustic correlations calculated."
    )


summary_lines.append(
    "\n\nInterpretation note:"
)

summary_lines.append(
    "Acoustic analyses are exploratory because "
    "the current dataset contains only 18 paired utterances."
)

summary_file = (
    RESULTS_DIR
    / "analysis_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(summary_lines)
    )

print(
    f"\nSaved: {summary_file}"
)


# ============================================================
# 14. Finish
# ============================================================

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

print("\nGenerated files:")

print(
    "  results/overall_summary.csv"
)

print(
    "  results/statistical_tests.csv"
)

print(
    "  results/normalized_error_rates.csv"
)

print(
    "  results/acoustic_correlations.csv"
)

print(
    "  results/analysis_summary.txt"
)

print(
    "  results/figures/"
)

print("\nDone.")
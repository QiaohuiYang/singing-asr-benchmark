"""
Final analysis pipeline for the Singing ASR Benchmark.

This script:
1. Loads the paired analysis table.
2. Computes descriptive statistics.
3. Performs paired Wilcoxon signed-rank tests.
4. Generates result figures.
5. Performs exploratory acoustic correlation analysis.
6. Saves analysis summaries for reproducibility.

Input:
    results/final_analysis.csv
    results/metrics.csv
    data/metadata_ground_truth.csv

Outputs:
    results/overall_summary.csv
    results/statistical_tests.csv
    results/normalized_error_rates.csv
    results/acoustic_correlations.csv
    results/analysis_summary.txt
    results/figures/*.png

Usage:
    python analyze_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FINAL_ANALYSIS_FILE = (
    RESULTS_DIR / "final_analysis.csv"
)

METRICS_FILE = (
    RESULTS_DIR / "metrics.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata_ground_truth.csv"
)

OVERALL_SUMMARY_FILE = (
    RESULTS_DIR / "overall_summary.csv"
)

STATISTICS_FILE = (
    RESULTS_DIR / "statistical_tests.csv"
)

NORMALIZED_ERRORS_FILE = (
    RESULTS_DIR / "normalized_error_rates.csv"
)

ACOUSTIC_CORRELATIONS_FILE = (
    RESULTS_DIR / "acoustic_correlations.csv"
)

ANALYSIS_SUMMARY_FILE = (
    RESULTS_DIR / "analysis_summary.txt"
)


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [
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


# ============================================================
# Data loading and validation
# ============================================================

def load_final_analysis() -> pd.DataFrame:
    """Load and validate the final paired analysis table."""

    if not FINAL_ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{FINAL_ANALYSIS_FILE}\n\n"
            "Please run analyze_data.py first."
        )

    df = pd.read_csv(
        FINAL_ANALYSIS_FILE
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns in final_analysis.csv:\n"
            + "\n".join(missing)
        )

    return df


def standardize_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create normalized language and length labels."""

    df = df.copy()

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

    return df


# ============================================================
# Overall descriptive statistics
# ============================================================

def calculate_overall_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate overall speech-vs-singing descriptive statistics."""

    summary_rows = []

    metric_specs = [
        (
            "WER",
            "WER_speech",
            "WER_singing",
            "delta_WER",
        ),
        (
            "CER",
            "CER_speech",
            "CER_singing",
            "delta_CER",
        ),
    ]

    for (
        metric_name,
        speech_col,
        singing_col,
        delta_col,
    ) in metric_specs:

        summary_rows.append({
            "metric": metric_name,
            "n_pairs": len(df),

            "speech_mean":
                df[speech_col].mean(),

            "singing_mean":
                df[singing_col].mean(),

            "mean_delta":
                df[delta_col].mean(),

            "speech_median":
                df[speech_col].median(),

            "singing_median":
                df[singing_col].median(),

            "median_delta":
                df[delta_col].median(),

            "speech_std":
                df[speech_col].std(),

            "singing_std":
                df[singing_col].std(),
        })

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# Wilcoxon signed-rank tests
# ============================================================

def wilcoxon_report(
    speech: pd.Series,
    singing: pd.Series,
) -> dict:
    """
    Perform a paired Wilcoxon signed-rank test.

    The effect_r value is an approximate rank-based effect
    size derived from the Wilcoxon statistic.
    """

    speech = np.asarray(
        speech,
        dtype=float,
    )

    singing = np.asarray(
        singing,
        dtype=float,
    )

    differences = (
        singing - speech
    )

    nonzero_differences = (
        differences[
            differences != 0
        ]
    )

    if len(nonzero_differences) == 0:

        return {
            "n_pairs": len(differences),
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

    n = len(nonzero_differences)

    # Expected Wilcoxon statistic under H0
    mean_w = (
        n * (n + 1) / 4
    )

    sd_w = np.sqrt(
        n
        * (n + 1)
        * (2 * n + 1)
        / 24
    )

    if sd_w > 0:

        z = (
            result.statistic
            - mean_w
        ) / sd_w

    else:

        z = 0.0

    effect_r = (
        abs(z) / np.sqrt(n)
    )

    return {
        "n_pairs": len(differences),
        "n_nonzero": n,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "effect_r": effect_r,
    }


def calculate_statistical_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run paired Wilcoxon tests for WER and CER."""

    rows = []

    metric_specs = [
        (
            "WER",
            "WER_speech",
            "WER_singing",
        ),
        (
            "CER",
            "CER_speech",
            "CER_singing",
        ),
    ]

    for (
        metric_name,
        speech_col,
        singing_col,
    ) in metric_specs:

        result = wilcoxon_report(
            df[speech_col],
            df[singing_col],
        )

        rows.append({
            "metric": metric_name,
            **result,
        })

    return pd.DataFrame(rows)


# ============================================================
# Figure 1: Paired WER
# ============================================================

def plot_paired_wer(
    df: pd.DataFrame,
) -> None:
    """Create a ranked paired WER dumbbell plot."""

    plot_df = (
        df.sort_values(
            "delta_WER",
            ascending=True,
        )
        .copy()
        .reset_index(drop=True)
    )

    y_positions = np.arange(
        len(plot_df)
    )

    plt.figure(
        figsize=(10, 9)
    )

    # --------------------------------------------------------
    # Connecting lines
    # --------------------------------------------------------

    for y, (_, row) in zip(
        y_positions,
        plot_df.iterrows(),
    ):

        plt.plot(
            [
                row["WER_speech"],
                row["WER_singing"],
            ],
            [
                y,
                y,
            ],
            linewidth=1.5,
            alpha=0.45,
        )

    # --------------------------------------------------------
    # Speech
    # --------------------------------------------------------

    plt.scatter(
        plot_df["WER_speech"],
        y_positions,
        s=65,
        marker="o",
        label="Speech",
        zorder=3,
    )

    # --------------------------------------------------------
    # Singing
    # --------------------------------------------------------

    plt.scatter(
        plot_df["WER_singing"],
        y_positions,
        s=75,
        marker="D",
        label="Singing",
        zorder=3,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.yticks(
        y_positions,
        plot_df["utterance_id"],
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

    output_file = (
        FIGURES_DIR
        / "fig1_paired_wer.png"
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


# ============================================================
# Figure 2: WER by language
# ============================================================

def plot_wer_by_language(
    df: pd.DataFrame,
) -> None:
    """Create WER comparison across languages and modes."""

    languages = [
        "english",
        "french",
    ]

    data = []
    positions = []
    labels = []

    position = 1

    for language in languages:

        subset = df[
            df["language_clean"]
            == language
        ]

        speech_values = (
            subset["WER_speech"]
            .dropna()
            .values
        )

        singing_values = (
            subset["WER_singing"]
            .dropna()
            .values
        )

        if len(speech_values) == 0:
            continue

        data.extend([
            speech_values,
            singing_values,
        ])

        positions.extend([
            position,
            position + 1,
        ])

        language_label = (
            language.capitalize()
        )

        labels.extend([
            f"{language_label}\nSpeech",
            f"{language_label}\nSinging",
        ])

        position += 3

    if not data:
        print(
            "Warning: no language data available."
        )
        return

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        data,
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

    plt.xlim(
        0.4,
        max(positions) + 0.6,
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "fig2_wer_language.png"
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


# ============================================================
# Figure 3: ΔWER by sentence length
# ============================================================

def plot_delta_wer_by_length(
    df: pd.DataFrame,
) -> None:
    """Create an individual-point plot of ΔWER by sentence length."""

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

    plt.figure(
        figsize=(8, 6)
    )

    for x, length_group in enumerate(
        length_order,
        start=1,
    ):

        values = (
            df[
                df["length_clean"]
                == length_group
            ]["delta_WER"]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        # Horizontal jitter
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

        # Median marker
        median_value = (
            np.median(values)
        )

        plt.plot(
            [
                x - 0.18,
                x + 0.18,
            ],
            [
                median_value,
                median_value,
            ],
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

    output_file = (
        FIGURES_DIR
        / "fig3_delta_wer_length.png"
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


# ============================================================
# Figure 4: Normalized error types
# ============================================================

def calculate_normalized_error_rates(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean error rates normalized by reference word count.
    """

    required_columns = [
        "utterance_id",
        "mode",
        "substitutions",
        "deletions",
        "insertions",
    ]

    missing = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in metrics.csv:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Obtain reference lengths
    # --------------------------------------------------------

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    required_metadata_columns = [
        "utterance_id",
        "reference",
    ]

    missing_metadata = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Missing columns in metadata_ground_truth.csv:\n"
            + "\n".join(missing_metadata)
        )

    metadata = metadata.copy()

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

    # --------------------------------------------------------
    # Merge reference lengths
    # --------------------------------------------------------

    metrics = metrics.merge(
        reference_lengths,
        on="utterance_id",
        how="left",
        validate="many_to_one",
    )

    if metrics[
        "reference_length"
    ].isna().any():

        raise ValueError(
            "Some recordings are missing "
            "reference word counts."
        )

    # --------------------------------------------------------
    # Normalize errors
    # --------------------------------------------------------

    for error_type in [
        "substitutions",
        "deletions",
        "insertions",
    ]:

        metrics[
            f"{error_type}_rate"
        ] = (
            metrics[error_type]
            / metrics["reference_length"]
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = (
        metrics
        .groupby("mode")[
            [
                "substitutions_rate",
                "deletions_rate",
                "insertions_rate",
            ]
        ]
        .mean()
        .reindex(
            ["speech", "singing"]
        )
    )

    return summary


def plot_normalized_error_types(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create normalized error-type comparison plot."""

    error_rate_summary = (
        calculate_normalized_error_rates(
            metrics
        )
    )

    error_rate_summary.to_csv(
        NORMALIZED_ERRORS_FILE,
        encoding="utf-8-sig",
    )

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

    ax.legend(
        title="Mode"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "fig4_error_types.png"
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

    return error_rate_summary


# ============================================================
# Acoustic analysis
# ============================================================

ACOUSTIC_FEATURES = [
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


def calculate_acoustic_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Pearson correlations between acoustic features and ΔWER."""

    rows = []

    for (
        feature,
        _,
        _,
    ) in ACOUSTIC_FEATURES:

        if feature not in df.columns:

            print(
                f"Skipping {feature}: column not found."
            )

            continue

        subset = (
            df[
                [
                    feature,
                    "delta_WER",
                ]
            ]
            .dropna()
        )

        if len(subset) < 3:

            print(
                f"Skipping {feature}: "
                "not enough observations."
            )

            continue

        x = subset[feature].to_numpy()
        y = subset["delta_WER"].to_numpy()

        if np.std(x) == 0:

            correlation = np.nan

        else:

            correlation = np.corrcoef(
                x,
                y,
            )[0, 1]

        rows.append({
            "feature": feature,
            "n": len(subset),
            "pearson_r": correlation,
        })

    return pd.DataFrame(rows)


def plot_acoustic_feature(
    df: pd.DataFrame,
    feature: str,
    x_label: str,
    filename: str,
) -> None:
    """Create one acoustic feature vs ΔWER scatter plot."""

    subset = (
        df[
            [
                feature,
                "delta_WER",
            ]
        ]
        .dropna()
    )

    if len(subset) < 3:

        print(
            f"Skipping {feature}: "
            "not enough observations."
        )

        return

    x = subset[feature].to_numpy()
    y = subset["delta_WER"].to_numpy()

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        x,
        y,
        s=60,
        alpha=0.75,
    )

    # --------------------------------------------------------
    # Linear trend
    # --------------------------------------------------------

    if np.std(x) > 0:

        slope, intercept = (
            np.polyfit(
                x,
                y,
                1,
            )
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


def run_acoustic_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate acoustic plots and correlation table."""

    print(
        "\nCreating acoustic analysis figures..."
    )

    for (
        feature,
        x_label,
        filename,
    ) in ACOUSTIC_FEATURES:

        if feature not in df.columns:

            print(
                f"Skipping {feature}: "
                "column not found."
            )

            continue

        plot_acoustic_feature(
            df,
            feature,
            x_label,
            filename,
        )

    correlation_df = (
        calculate_acoustic_correlations(
            df
        )
    )

    correlation_df.to_csv(
        ACOUSTIC_CORRELATIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return correlation_df


# ============================================================
# Summary text
# ============================================================

def create_text_summary(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> None:
    """Write a human-readable analysis summary."""

    lines = []

    lines.append(
        "Singing ASR Benchmark - Exploratory Analysis"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"\nNumber of paired utterances: {len(df)}"
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    lines.append(
        "\nOverall performance:"
    )

    for _, row in summary_df.iterrows():

        lines.append(
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

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------

    lines.append(
        "\n\nWilcoxon signed-rank tests:"
    )

    for _, row in stats_df.iterrows():

        lines.append(
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

    # --------------------------------------------------------
    # Direction of effect
    # --------------------------------------------------------

    delta = df["delta_WER"]

    lines.append(
        "\n\nDirection of singing effect:"
    )

    lines.append(
        f"\n  Singing WER > Speech WER: "
        f"{(delta > 0).sum()}"
    )

    lines.append(
        f"\n  Singing WER = Speech WER: "
        f"{(delta == 0).sum()}"
    )

    lines.append(
        f"\n  Singing WER < Speech WER: "
        f"{(delta < 0).sum()}"
    )

    # --------------------------------------------------------
    # Acoustic analysis
    # --------------------------------------------------------

    lines.append(
        "\n\nAcoustic correlations:"
    )

    if len(correlation_df) > 0:

        for _, row in correlation_df.iterrows():

            correlation = row[
                "pearson_r"
            ]

            if pd.isna(correlation):

                correlation_text = "NaN"

            else:

                correlation_text = (
                    f"{correlation:.4f}"
                )

            lines.append(
                f"\n  {row['feature']}: "
                f"r = {correlation_text} "
                f"(n={int(row['n'])})"
            )

    else:

        lines.append(
            "\n  No acoustic correlations calculated."
        )

    # --------------------------------------------------------
    # Interpretation note
    # --------------------------------------------------------

    lines.append(
        "\n\nInterpretation note:"
    )

    lines.append(
        "Acoustic analyses are exploratory because "
        "the current dataset contains only 18 paired utterances "
        "and pitch-based features may contain measurement noise."
    )

    with open(
        ANALYSIS_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    """Run the complete final analysis pipeline."""

    print("=" * 70)
    print("SINGING ASR BENCHMARK")
    print("Final Exploratory Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_final_analysis()

    df = standardize_categories(
        df
    )

    print(
        f"\nFinal analysis rows: "
        f"{len(df)}"
    )

    print(
        f"Languages: "
        f"{df['language'].unique().tolist()}"
    )

    print(
        f"Length groups: "
        f"{df['length_group'].unique().tolist()}"
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_df = (
        calculate_overall_summary(
            df
        )
    )

    summary_df.to_csv(
        OVERALL_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nOverall summary:"
    )

    print(
        summary_df
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------

    stats_df = (
        calculate_statistical_tests(
            df
        )
    )

    stats_df.to_csv(
        STATISTICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nWilcoxon signed-rank tests:"
    )

    print(
        stats_df
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print(
        "\nCreating figures..."
    )

    plot_paired_wer(
        df
    )

    plot_wer_by_language(
        df
    )

    plot_delta_wer_by_length(
        df
    )

    # --------------------------------------------------------
    # Normalized error types
    # --------------------------------------------------------

    if METRICS_FILE.exists():

        metrics = pd.read_csv(
            METRICS_FILE
        )

        print(
            "\nCreating normalized error analysis..."
        )

        normalized_errors = (
            plot_normalized_error_types(
                metrics
            )
        )

        print(
            "\nNormalized error rates:"
        )

        print(
            normalized_errors
            .round(4)
            .to_string()
        )

    else:

        print(
            "\nWarning: metrics.csv not found. "
            "Skipping normalized error analysis."
        )

        normalized_errors = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Acoustic analysis
    # --------------------------------------------------------

    correlation_df = (
        run_acoustic_analysis(
            df
        )
    )

    print(
        "\nAcoustic correlations:"
    )

    print(
        correlation_df
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    create_text_summary(
        df,
        summary_df,
        stats_df,
        correlation_df,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  {OVERALL_SUMMARY_FILE}"
    )

    print(
        f"  {STATISTICS_FILE}"
    )

    print(
        f"  {NORMALIZED_ERRORS_FILE}"
    )

    print(
        f"  {ACOUSTIC_CORRELATIONS_FILE}"
    )

    print(
        f"  {ANALYSIS_SUMMARY_FILE}"
    )

    print(
        f"  {FIGURES_DIR}"
    )

    print("\nDone.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
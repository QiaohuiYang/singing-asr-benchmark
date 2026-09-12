"""
Acoustic feature extraction for the Singing ASR Benchmark.

This script:
1. Loads the experiment metadata.
2. Converts M4A recordings to temporary 16 kHz mono WAV files.
3. Extracts duration, RMS energy, and fundamental-frequency (F0)
   features using librosa.
4. Saves the extracted features to results/acoustic_features.csv.

The extracted acoustic features are used for exploratory analysis
of utterance-level variation in singing-related ASR degradation.

Usage:
    python extract_features.py
"""

from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata_ground_truth.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "acoustic_features.csv"
)

SAMPLE_RATE = 16000

F0_MIN_NOTE = "C2"
F0_MAX_NOTE = "C7"


# ============================================================
# FFmpeg
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# Data loading
# ============================================================

def load_metadata() -> pd.DataFrame:
    """Load and validate the experiment metadata."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    required_columns = [
        "utterance_id",
        "language",
        "language_code",
        "mode",
        "length_group",
        "filename",
        "relative_path",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required metadata columns:\n"
            + "\n".join(missing_columns)
        )

    return metadata


# ============================================================
# Audio conversion
# ============================================================

def convert_m4a_to_wav(
    m4a_path: Path,
    wav_path: Path,
) -> None:
    """
    Convert an M4A file to a temporary mono WAV file.

    Audio is resampled to 16 kHz, which is sufficient for the
    acoustic features used in this project.
    """

    command = [
        FFMPEG,
        "-y",
        "-i",
        str(m4a_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg conversion failed.\n"
            f"File: {m4a_path}\n"
            f"FFmpeg output:\n{result.stderr}"
        )


# ============================================================
# Acoustic feature extraction
# ============================================================

def extract_acoustic_features(
    wav_path: Path,
) -> dict:
    """
    Extract acoustic features from one WAV recording.

    Returns:
        Dictionary containing:
        - sample rate
        - duration
        - RMS energy
        - mean / median F0
        - F0 standard deviation
        - robust F0 range
        - number of voiced frames
    """

    y, sr = librosa.load(
        wav_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = librosa.get_duration(
        y=y,
        sr=sr,
    )

    # --------------------------------------------------------
    # RMS energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
    )

    mean_rms = float(
        np.mean(rms)
    )

    # --------------------------------------------------------
    # Fundamental frequency
    # --------------------------------------------------------

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(F0_MIN_NOTE),
        fmax=librosa.note_to_hz(F0_MAX_NOTE),
        sr=sr,
    )

    valid_f0 = f0[
        np.isfinite(f0)
    ]

    if len(valid_f0) > 0:

        mean_f0 = float(
            np.mean(valid_f0)
        )

        median_f0 = float(
            np.median(valid_f0)
        )

        std_f0 = float(
            np.std(valid_f0)
        )

        # Robust range:
        # 90th percentile - 10th percentile
        f0_p10 = float(
            np.percentile(
                valid_f0,
                10,
            )
        )

        f0_p90 = float(
            np.percentile(
                valid_f0,
                90,
            )
        )

        f0_range = (
            f0_p90 - f0_p10
        )

        voiced_frames = len(
            valid_f0
        )

    else:

        mean_f0 = np.nan
        median_f0 = np.nan
        std_f0 = np.nan
        f0_range = np.nan
        voiced_frames = 0

    return {
        "sample_rate": sr,
        "duration_sec": duration,
        "mean_rms": mean_rms,
        "mean_f0_hz": mean_f0,
        "median_f0_hz": median_f0,
        "std_f0_hz": std_f0,
        "f0_range_hz": f0_range,
        "voiced_frames": voiced_frames,
    }


# ============================================================
# Process one recording
# ============================================================

def process_recording(
    row: pd.Series,
) -> dict:
    """
    Process one metadata row and return its acoustic features.
    """

    audio_path = (
        PROJECT_ROOT
        / row["relative_path"]
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    # Use a temporary directory so converted WAV files
    # are automatically deleted after processing.
    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = (
            Path(temp_dir)
            / "audio.wav"
        )

        convert_m4a_to_wav(
            audio_path,
            wav_path,
        )

        features = extract_acoustic_features(
            wav_path
        )

    return {
        "utterance_id": row["utterance_id"],
        "language": row["language"],
        "language_code": row["language_code"],
        "mode": row["mode"],
        "length_group": row["length_group"],
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        **features,
    }


# ============================================================
# Main extraction pipeline
# ============================================================

def main() -> None:
    """Run the complete acoustic feature extraction pipeline."""

    print("=" * 70)
    print("SINGING ASR BENCHMARK")
    print("Acoustic Feature Extraction")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    print("\nDataset check")
    print("-" * 70)
    print(
        f"Number of recordings: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    results = []
    failed_files = []

    for index, row in metadata.iterrows():

        print("\n" + "=" * 70)

        print(
            f"[{index + 1}/{len(metadata)}] "
            f"{row['utterance_id']} | "
            f"{row['language']} | "
            f"{row['mode']} | "
            f"{row['length_group']}"
        )

        print("=" * 70)

        try:

            result = process_recording(
                row
            )

            results.append(result)

            print(
                f"Duration: "
                f"{result['duration_sec']:.2f}s | "
                f"Mean F0: "
                f"{result['mean_f0_hz']:.1f} Hz | "
                f"F0 range: "
                f"{result['f0_range_hz']:.1f} Hz | "
                f"RMS: "
                f"{result['mean_rms']:.4f}"
            )

        except Exception as exc:

            audio_path = (
                PROJECT_ROOT
                / row["relative_path"]
            )

            print(
                f"ERROR processing:\n"
                f"{audio_path}"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error message: {exc}"
            )

            failed_files.append(
                str(audio_path)
            )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    features_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Successfully processed: "
        f"{len(features_df)} / {len(metadata)}"
    )

    print(
        f"Failed recordings: "
        f"{len(failed_files)}"
    )

    if failed_files:

        print("\nFailed files:")

        for file_path in failed_files:
            print(
                f"  - {file_path}"
            )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    if len(features_df) > 0:

        summary_columns = [
            "duration_sec",
            "mean_f0_hz",
            "median_f0_hz",
            "std_f0_hz",
            "f0_range_hz",
            "mean_rms",
        ]

        print("\nFeature summary:")

        print(
            features_df[
                summary_columns
            ]
            .describe()
            .round(2)
            .to_string()
        )

    else:

        print(
            "\nNo files were successfully processed."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
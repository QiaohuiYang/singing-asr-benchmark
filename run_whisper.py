"""
Whisper inference pipeline for the Singing ASR Benchmark.

This script:
1. Loads the experiment metadata.
2. Runs Whisper on each speech and singing recording.
3. Saves the transcriptions and references to results/transcriptions.csv.

The experiment uses Whisper base as a fixed pretrained ASR baseline.
No model fine-tuning is performed.

Usage:
    python run_whisper.py
"""

from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata_ground_truth.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "transcriptions.csv"
)

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5

EXPECTED_RECORDINGS = 36


# ============================================================
# Utility functions
# ============================================================

def load_metadata() -> pd.DataFrame:
    """Load and validate the experiment metadata."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    required_columns = [
        "utterance_id",
        "language",
        "language_code",
        "mode",
        "length_group",
        "filename",
        "relative_path",
        "reference",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required metadata columns:\n"
            + "\n".join(missing_columns)
        )

    return df


def load_model() -> WhisperModel:
    """Load the pretrained Whisper model."""

    print("\nLoading Whisper model...")
    print(f"Model: {MODEL_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Compute type: {COMPUTE_TYPE}")

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Whisper loaded successfully.")

    return model


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language_code: str,
) -> str:
    """
    Transcribe one audio file using Whisper.

    Returns:
        The concatenated transcription from all non-empty segments.
    """

    segments, _ = model.transcribe(
        str(audio_path),
        language=language_code,
        beam_size=BEAM_SIZE,
    )

    text_parts = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            text_parts.append(text)

    return " ".join(text_parts)


def save_results(results: list[dict]) -> None:
    """Save transcription results to CSV."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    """Run the complete Whisper transcription pipeline."""

    print("=" * 70)
    print("SINGING ASR BENCHMARK")
    print("Whisper transcription")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = load_metadata()

    print("\nDataset check")
    print("-" * 70)
    print(f"Number of recordings: {len(df)}")

    if len(df) != EXPECTED_RECORDINGS:

        print(
            "WARNING: "
            f"Expected {EXPECTED_RECORDINGS} recordings, "
            f"but found {len(df)}."
        )

    # --------------------------------------------------------
    # Load Whisper
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    results = []
    failed_files = []

    for index, row in df.iterrows():

        print("\n" + "=" * 70)

        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['utterance_id']} | "
            f"{row['language']} | "
            f"{row['mode']} | "
            f"{row['length_group']}"
        )

        print("=" * 70)

        audio_path = (
            PROJECT_ROOT
            / row["relative_path"]
        )

        # ----------------------------------------------------
        # Check audio file
        # ----------------------------------------------------

        if not audio_path.exists():

            print(
                f"WARNING: Audio file not found:\n"
                f"{audio_path}"
            )

            failed_files.append(
                str(audio_path)
            )

            continue

        # ----------------------------------------------------
        # Run Whisper
        # ----------------------------------------------------

        try:

            transcription = transcribe_audio(
                model=model,
                audio_path=audio_path,
                language_code=row["language_code"],
            )

        except Exception as exc:

            print(
                f"ERROR during transcription: {exc}"
            )

            failed_files.append(
                str(audio_path)
            )

            continue

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print("\nReference:")
        print(row["reference"])

        print("\nWhisper prediction:")
        print(transcription)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "utterance_id": row["utterance_id"],
            "language": row["language"],
            "language_code": row["language_code"],
            "mode": row["mode"],
            "length_group": row["length_group"],
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "reference": row["reference"],
            "prediction": transcription,
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 70)

    print(
        f"Successful transcriptions: "
        f"{len(results)}/{len(df)}"
    )

    print(
        f"Failed recordings: "
        f"{len(failed_files)}"
    )

    if failed_files:

        print("\nFailed files:")

        for file_path in failed_files:
            print(f"  - {file_path}")

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
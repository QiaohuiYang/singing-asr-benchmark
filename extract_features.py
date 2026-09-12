from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

METADATA_FILE = Path("data/metadata_ground_truth.csv")
OUTPUT_FILE = Path("results/acoustic_features.csv")


# ============================================================
# 2. FFmpeg executable
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# 3. Load metadata
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

metadata = pd.read_csv(METADATA_FILE)

print("=" * 70)
print("ACOUSTIC FEATURE EXTRACTION")
print("=" * 70)

print(f"Number of recordings in metadata: {len(metadata)}")


# ============================================================
# 4. Function: convert M4A -> WAV
# ============================================================

def convert_m4a_to_wav(m4a_path: Path, wav_path: Path) -> None:
    """
    Convert an M4A file to mono WAV using the bundled FFmpeg.
    We use 16 kHz because it is sufficient for the acoustic
    features needed in this project and keeps processing faster.
    """

    command = [
        FFMPEG,
        "-y",
        "-i", str(m4a_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
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
            f"FFmpeg conversion failed for {m4a_path}\n"
            f"{result.stderr}"
        )


# ============================================================
# 5. Feature extraction
# ============================================================

results = []

for i, row in metadata.iterrows():

    audio_path = Path(row["relative_path"])

    print(
        f"\n[{i + 1}/{len(metadata)}] "
        f"{row['utterance_id']} | "
        f"{row['language']} | "
        f"{row['mode']} | "
        f"{row['length_group']}"
    )

    if not audio_path.exists():
        print(f"WARNING: file not found -> {audio_path}")
        continue

    try:

        # ----------------------------------------------------
        # Convert M4A to temporary WAV
        # ----------------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:

            wav_path = Path(temp_dir) / "audio.wav"

            convert_m4a_to_wav(
                audio_path,
                wav_path
            )

            # ------------------------------------------------
            # Load WAV
            # ------------------------------------------------

            y, sr = librosa.load(
                wav_path,
                sr=16000,
                mono=True,
            )

            # ------------------------------------------------
            # Duration
            # ------------------------------------------------

            duration = librosa.get_duration(
                y=y,
                sr=sr
            )

            # ------------------------------------------------
            # RMS energy
            # ------------------------------------------------

            rms = librosa.feature.rms(
                y=y
            )

            mean_rms = float(
                np.mean(rms)
            )

            # ------------------------------------------------
            # Fundamental frequency
            # ------------------------------------------------

            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )

            # Keep only valid F0 estimates
            valid_f0 = f0[np.isfinite(f0)]

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

                # Robust F0 range:
                # 90th percentile - 10th percentile
                f0_p10 = float(
                    np.percentile(valid_f0, 10)
                )

                f0_p90 = float(
                    np.percentile(valid_f0, 90)
                )

                f0_range = f0_p90 - f0_p10

                voiced_frames = len(valid_f0)

            else:

                mean_f0 = np.nan
                median_f0 = np.nan
                std_f0 = np.nan
                f0_range = np.nan
                voiced_frames = 0

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "utterance_id": row["utterance_id"],
            "language": row["language"],
            "language_code": row["language_code"],
            "mode": row["mode"],
            "length_group": row["length_group"],
            "filename": row["filename"],
            "relative_path": row["relative_path"],

            "sample_rate": sr,
            "duration_sec": duration,

            "mean_rms": mean_rms,

            "mean_f0_hz": mean_f0,
            "median_f0_hz": median_f0,
            "std_f0_hz": std_f0,
            "f0_range_hz": f0_range,
            "voiced_frames": voiced_frames,
        })

        print(
            f"Duration: {duration:.2f}s | "
            f"Mean F0: {mean_f0:.1f} Hz | "
            f"F0 range: {f0_range:.1f} Hz"
        )

    except Exception as e:

        print(
            f"ERROR processing {audio_path}"
        )

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {e}"
        )


# ============================================================
# 6. Create DataFrame
# ============================================================

features_df = pd.DataFrame(results)


# ============================================================
# 7. Save results
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. Final check
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Successfully processed: "
    f"{len(features_df)} / {len(metadata)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


# ============================================================
# 9. Summary
# ============================================================

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
        features_df[summary_columns]
        .describe()
        .round(2)
        .to_string()
    )

else:

    print("\nNo files were successfully processed.")
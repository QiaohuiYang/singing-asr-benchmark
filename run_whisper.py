from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel


# ============================================================
# 1. Paths
# ============================================================

METADATA_FILE = Path("data/metadata_ground_truth.csv")
OUTPUT_DIR = Path("results")
OUTPUT_FILE = OUTPUT_DIR / "transcriptions.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load metadata
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("=" * 70)
print("DATASET CHECK")
print("=" * 70)
print(f"Number of recordings: {len(df)}")

if len(df) != 36:
    print("WARNING: Expected 36 recordings.")


# ============================================================
# 3. Load Whisper
# ============================================================

print("\nLoading Whisper base model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)

print("Whisper loaded successfully.")


# ============================================================
# 4. Run transcription
# ============================================================

results = []

for i, row in df.iterrows():

    audio_path = Path(row["relative_path"])

    print("\n" + "=" * 70)
    print(
        f"[{i + 1}/{len(df)}] "
        f"{row['utterance_id']} | "
        f"{row['language']} | "
        f"{row['mode']} | "
        f"{row['length_group']}"
    )
    print("=" * 70)

    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        continue

    # Run Whisper
    segments, info = model.transcribe(
        str(audio_path),
        language=row["language_code"],
        beam_size=5,
    )

    # Collect transcription
    text_parts = []
    segment_list = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            text_parts.append(text)

        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })

    transcription = " ".join(text_parts)

    print("Reference:")
    print(row["reference"])

    print("\nWhisper:")
    print(transcription)

    # Save result
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


# ============================================================
# 5. Save all transcriptions
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Successful transcriptions: {len(results_df)}")
print(f"Saved to: {OUTPUT_FILE}")
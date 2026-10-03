"""Build train/val manifests for the Mongolian-only TTS model.

Source: LJSpeech-style dataset (id|text|normalized) with wavs/ folder.
Filters clips to 0.8-11.5 s (stable training windows), max 240 chars.
Outputs: data/train_manifest.jsonl, data/val_manifest.jsonl + prints stats.
Only needs soundfile -> can run with any python that has it.
"""
import json
import random
from pathlib import Path

import soundfile as sf

DATASET_ROOT = Path(r"C:\Users\hi\Documents\chatterbox-finetuning\MyTTSDataset_combined")
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

MIN_D, MAX_D = 0.8, 11.5
MAX_CHARS = 240
VAL_FRACTION = 0.02

rows, skipped, seconds = [], 0, 0.0
with open(DATASET_ROOT / "metadata.csv", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 2:
            skipped += 1
            continue
        uid = parts[0].strip()
        text = (parts[2] if len(parts) > 2 else parts[1]).strip()
        text = " ".join(text.split())
        if not (2 <= len(text) <= MAX_CHARS):
            skipped += 1
            continue
        wav = DATASET_ROOT / "wavs" / f"{uid}.wav"
        if not wav.exists():
            skipped += 1
            continue
        try:
            info = sf.info(str(wav))
        except Exception:
            skipped += 1
            continue
        if not (MIN_D <= info.duration <= MAX_D):
            skipped += 1
            continue
        rows.append({"audio": str(wav), "text": text, "duration": info.duration})
        seconds += info.duration

rng = random.Random(42)
rng.shuffle(rows)
n_val = max(1, int(len(rows) * VAL_FRACTION))
val, train = rows[:n_val], rows[n_val:]

for name, data in (("train", train), ("val", val)):
    with open(OUT / f"{name}_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"kept={len(rows)}  skipped={skipped}  total_hours={seconds/3600:.2f}")
print(f"train={len(train)}  val={len(val)}")
print(f"wrote {OUT}\\train_manifest.jsonl and val_manifest.jsonl")

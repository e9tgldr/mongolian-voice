"""Repair the faster-whisper cache, then diagnose the empty-STT issue."""
import os
import shutil
import time
from pathlib import Path

os.environ["HF_HOME"] = r"C:\Users\hi\Documents\mongolian-voice\.hf"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

hub = Path(os.environ["HF_HOME"]) / "hub"
tgt = hub / "models--Systran--faster-whisper-small"
if tgt.exists():
    print("removing corrupt cache ...", flush=True)
    shutil.rmtree(tgt, ignore_errors=True)

from huggingface_hub import snapshot_download

p = None
for attempt in range(6):
    try:
        p = snapshot_download("Systran/faster-whisper-small")
        print("downloaded to", p, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(5)
if p is None:
    raise SystemExit("download failed after 6 attempts")

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import torch

os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
from faster_whisper import WhisperModel

wav, sr = sf.read(r"C:\Users\hi\Documents\mongolian-voice\roundtrip_tts.wav",
                  dtype="float32")
print("audio: dur", round(len(wav) / sr, 2), "peak", round(float(np.abs(wav).max()), 3), flush=True)

m = WhisperModel("small", device="cuda", compute_type="float16")
print("model loaded", flush=True)

for vad in (False, True):
    segs, info = m.transcribe(wav, language="mn", beam_size=1, vad_filter=vad)
    print(f"vad={vad} ->", repr(" ".join(s.text for s in segs))[:250], flush=True)

segs, info = m.transcribe(wav, beam_size=1, vad_filter=False)
print("auto-lang", info.language, "->", repr(" ".join(s.text for s in segs))[:250], flush=True)

segs, info = m.transcribe(wav, language="mn", beam_size=5, vad_filter=False)
print("beam5 ->", repr(" ".join(s.text for s in segs))[:250], flush=True)

"""Resume the whisper download to completion, then diagnose STT quality.

Tests both the synthetic roundtrip clip and a REAL dataset clip with a known
transcript, with and without VAD, beam 1 and 5 — to separate synth-audio
quality issues from whisper-Mongolian issues.
"""
import os
import time
from pathlib import Path

os.environ["HF_HOME"] = r"C:\Users\hi\Documents\mongolian-voice\.hf"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

p = None
for attempt in range(10):
    try:
        p = snapshot_download("Systran/faster-whisper-small")
        print("cache complete:", p, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}", flush=True)
        time.sleep(10)
if p is None:
    raise SystemExit("download did not complete")

os.environ["HF_HUB_OFFLINE"] = "1"

import torch

os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
import soundfile as sf
from faster_whisper import WhisperModel

m = WhisperModel("small", device="cuda", compute_type="float16")
print("model loaded", flush=True)

root = Path(r"C:\Users\hi\Documents\mongolian-voice")
for path, label in [(root / "roundtrip_tts.wav", "SYNTH"),
                    (Path(r"C:\Users\hi\Documents\chatterbox-finetuning"
                           r"\MyTTSDataset_combined\wavs\mn_000000.wav"), "REAL")]:
    wav, sr = sf.read(str(path), dtype="float32")
    if sr != 16000:
        from scipy.signal import resample_poly
        import numpy as np
        g = np.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype("float32")
    for beam in (1, 5):
        for vad in (False, True):
            segs, _ = m.transcribe(wav, language="mn", beam_size=beam, vad_filter=vad)
            print(f"{label} beam={beam} vad={vad} ->",
                  " ".join(s.text for s in segs)[:150] or "<EMPTY>", flush=True)

print("GROUND TRUTH: Цэрэгмааг бүжигт урихыг горилхийлсэн залуус үүдээр бултас "
      "бултасхийн харахуй түүнд зовох сэтгэл төрнө.", flush=True)

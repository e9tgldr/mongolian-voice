"""Download faster-whisper-medium to the project cache with patient retries."""
import os
import time

os.environ["HF_HOME"] = r"C:\Users\hi\Documents\mongolian-voice\.hf"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

for attempt in range(12):
    try:
        p = snapshot_download("Systran/faster-whisper-medium")
        print("medium downloaded:", p, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}", flush=True)
        time.sleep(8)
else:
    raise SystemExit("medium download did not complete")

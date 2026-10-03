"""Smoke test: TTS round trip. Speaks Mongolian, then transcribes its own audio."""
import json
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8771"
TEXT = "Сайн байна уу? Би Монголын шинэ хоолой. Надад юу хэлэхийг хэлээрэй."

req = urllib.request.Request(
    f"{BASE}/tts",
    data=json.dumps({"text": TEXT}).encode(),
    headers={"Content-Type": "application/json"},
)
wav = urllib.request.urlopen(req, timeout=120).read()
out = Path(__file__).parent / "roundtrip_tts.wav"
out.write_bytes(wav)
print(f"TTS ok -> {out} ({len(wav)/1024:.0f} KB)")

boundary = "----mnvoice"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"roundtrip_tts.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    f"{BASE}/stt",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
res = json.loads(urllib.request.urlopen(req, timeout=120).read())
print(f"STT ok -> {res['text']!r}  ({res['elapsed']}s on {res['device']})")

"""API test: STT quality on a real clip + full TTS->STT round trip."""
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8771"


def stt(path_or_bytes, is_path=True):
    data = Path(path_or_bytes).read_bytes() if is_path else path_or_bytes
    b = "----" + uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + data \
        + f'\r\n--{b}--\r\n'.encode()
    req = urllib.request.Request(f"{BASE}/stt", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=300).read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} -> {e.read().decode('utf-8', 'replace')}")


def tts(text):
    req = urllib.request.Request(
        f"{BASE}/tts", data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300).read()


res = stt(r"C:\Users\hi\Documents\chatterbox-finetuning"
          r"\MyTTSDataset_combined\wavs\mn_000000.wav")
print("STT : ", res["text"][:160])
print(f"       ({res['elapsed']}s for {res['audio_duration']}s audio, "
      f"device {res['device']})")
print("TRUE:  Цэрэгмааг бүжигт урихыг горилхийлсэн залуус үүдээр бултас "
      "бултасхийн харахуй түүнд зовох сэтгэл төрнө.")

wav = tts("Сайн байна уу? Би өнөөдөр маш их ажилласан.")
Path("roundtrip_tts.wav").write_bytes(wav)
res2 = stt(wav, is_path=False)
print("ROUND TRIP:", res2["text"][:160])

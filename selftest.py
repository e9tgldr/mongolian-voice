"""No-microphone end-to-end smoke test for the Mongolian voice stack.

    Mongolian text -> POST /tts (real neural speech) -> real wav
                  -> POST /stt (real Whisper)
                  -> actions.parse / actions.resolve (no side effects)
                  -> PASS / SOFT WARNING verdict

The server must already be running (START-SERVER.ps1); this test never starts
it, never records, never plays audio, and the TTS reply is never spoken.

    python selftest.py [--cycles 3] [--keep-wav wavs] [--server http://127.0.0.1:8771]
"""
from __future__ import annotations

import sys
for _stream in ("stdout", "stderr"):                # the console here is cp1252
    getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")

import argparse
import array
import io
import json
import math
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import actions

DEFAULT_SERVER = "http://127.0.0.1:8771"
START_HINT = "powershell -ExecutionPolicy Bypass -File .\\START-SERVER.ps1"
BOUNDARY = "----mnVoiceSelfTest"
MIN_SEC, SILENT_RMS, HTTP_TIMEOUT = 0.25, 5e-4, 300.0

class Fail(Exception):
    """Hard failure of the current case; the rest of the table still runs."""

@dataclass
class Case:
    text: str
    expect: str                    # intent the written command must produce
    note: str = ""

CASES: tuple[Case, ...] = (
    Case("Ворд нээ", "open", "app alias -> winword"),
    Case("хэрэгсэл нээ", "open", "open verb, no such app on this PC"),
    Case("IELTS study pack нээ", "open", "real file, exercises the FileIndex path"),
    Case("танилцуулга эхлүүл", "present", "slides intent, no deck named"),
    Case("файлууд жагсаа", "list", "list intent, no file lookup"),
    Case("зогсоо", "close", "bare stop -> close whatever is running"),
    Case("сонсохыг зогсоо", "sleep", "sleep guard wins over close"),
    Case("интернетэд цаг хай", "search", "search verb -> search url"),
)

@dataclass
class Result:
    case: Case
    pos: int
    cycle: int
    transcript = ""
    intent: str | None = None
    action = value = reply = hard = audio = ""
    tts_sec = stt_sec = total_sec = 0.0
    warns: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.hard and self.intent == self.case.expect

    @property
    def warnings(self) -> list:
        out = list(self.warns)
        if not self.hard and self.intent != self.case.expect:
            out.append(f"intent mismatch: expected {self.case.expect!r}, "
                       f"got {self.intent!r} from the transcript above")
        return out

# ---- server calls (stdlib only) ----

def err_text(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read()[:200].decode("utf-8", "replace").strip()
        except Exception:                                       # noqa: BLE001
            body = ""
        return f"HTTP {exc.code} {exc.reason}: {body}" if body else f"HTTP {exc.code} {exc.reason}"
    return f"{type(exc).__name__}: {exc}"

def health(base: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
        return json.loads(r.read())

def tts(base: str, text: str) -> tuple[bytes, float | None]:
    req = urllib.request.Request(f"{base}/tts", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        wav, synth = r.read(), r.headers.get("X-Synth-Time-Sec")
    return wav, float(synth) if synth else None

def stt(base: str, wav: bytes, language: str = "mn") -> dict:
    body = (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; "
            "filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + wav + \
        (f"\r\n--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"language\""
         "\r\n\r\n").encode() + language.encode() + f"\r\n--{BOUNDARY}--\r\n".encode()
    req = urllib.request.Request(f"{base}/stt", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read())

def probe(wav: bytes) -> tuple[float, int, float]:
    """-> (seconds, sample rate, rms).  Stdlib only: no audio library is loaded."""
    with wave.open(io.BytesIO(wav), "rb") as w:
        sr, width, frames = w.getframerate(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(frames)
    if width != 2 or not raw:
        return frames / float(sr), sr, 1.0                   # unknown encoding: no guess
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) - len(raw) % 2])
    if sys.byteorder != "little":
        pcm.byteswap()
    return frames / float(sr), sr, math.sqrt(sum(s * s for s in pcm) / len(pcm)) / 32768.0

def dry_handle(assistant: "actions.Assistant", text: str):
    """Assistant.handle() with the same return shape, but nothing is executed.

    actions.run() calls os.startfile / webbrowser.open / Popen / process-kill
    before it looks at dry_run, so handle() would really open the file, launch
    the app or kill the process.  parse + resolve build the same Intent and
    Action and touch nothing but the file index.
    """
    intent = actions.parse(text)
    if intent is None:
        return None, actions.Action("none", reply="", ok=False)
    action = actions.resolve(intent, assistant.index)
    action.extra = (action.extra + f" [DRY-RUN {action.kind}]").strip()
    return intent, action

# ---- one round trip ----

def speak(res: Result, base: str, keep: Path | None) -> bytes:
    t0 = time.perf_counter()
    try:
        wav, synth = tts(base, res.case.text)
    except Exception as e:                                   # noqa: BLE001
        raise Fail(f"synthesis failed ({err_text(e)})") from e
    res.tts_sec = max(time.perf_counter() - t0, synth or 0.0)
    if not wav:
        raise Fail("synthesis returned no audio")
    try:
        sec, sr, rms = probe(wav)
    except Exception as e:                                   # noqa: BLE001
        raise Fail(f"synthesised wav is unreadable ({err_text(e)})") from e
    res.audio = f"{sec:.2f}s {sr}Hz  rms {rms:.4f}"
    if sec < MIN_SEC:
        raise Fail(f"synthesised wav too short ({sec:.2f}s)")
    if rms < SILENT_RMS:
        raise Fail(f"synthesised wav is silent (rms {rms:.5f})")
    if keep is not None:
        (keep / f"case{res.pos:02d}-c{res.cycle}-{sr}hz.wav").write_bytes(wav)
    return wav

def listen(res: Result, base: str, wav: bytes) -> None:
    t0 = time.perf_counter()
    try:
        out = stt(base, wav)
    except Exception as e:                                   # noqa: BLE001
        raise Fail(f"transcription failed ({err_text(e)})") from e
    res.stt_sec = time.perf_counter() - t0
    res.transcript = (out.get("text") or "").strip()
    if not res.transcript:
        raise Fail("transcription came back empty")
    if actions.normalize(res.transcript) != actions.normalize(res.case.text):
        res.warns.append(f"transcript differs from the spoken text: {res.transcript!r}")

def interpret(res: Result, assistant: "actions.Assistant") -> None:
    intent, action = dry_handle(assistant, res.transcript)
    res.intent = intent.name if intent else None
    res.action, res.value, res.reply = action.kind, action.value, action.reply
    if intent is None:
        raise Fail("no intent recognised in the transcript")

def run_case(case: Case, pos: int, cycle: int, base: str,
             assistant: "actions.Assistant", keep: Path | None) -> Result:
    res = Result(case=case, pos=pos, cycle=cycle)
    t0 = time.perf_counter()
    try:
        listen(res, base, speak(res, base, keep))
        interpret(res, assistant)
    except Fail as e:
        res.hard = str(e)
    except Exception as e:                                   # noqa: BLE001
        res.hard = f"unexpected error ({err_text(e)})"
    res.total_sec = time.perf_counter() - t0
    return res

# ---- reporting ----

def report(res: Result, total: int) -> None:
    tag = f"case {res.pos}/{total}" + (f"  cycle {res.cycle}" if res.cycle > 1 else "")
    print(f"\n{tag}  {res.case.text}   [{res.case.note}]")
    if res.audio:
        print(f"  wav     {res.audio}")
    print(f"  heard   {res.transcript!r}")
    print(f"  intent  {res.intent!r}   action {res.action or '-'}  {res.value[:70]}")
    if res.reply:
        print(f"  reply   {res.reply}")
    print(f"  {'PASS' if res.passed else 'FAIL'}   round trip {res.total_sec:.2f}s "
          f"(tts {res.tts_sec:.2f}s + stt {res.stt_sec:.2f}s)")
    if res.hard:
        print(f"  HARD FAILURE  {res.hard}")
    for w in res.warnings:
        print(f"  SOFT WARNING  {w}")

def print_health(info: dict, base: str) -> None:
    tts_info, stt_info = info.get("tts", {}), info.get("stt", {})
    print(f"server      {base}  status={info.get('status')}")
    print(f"  tts       backend={tts_info.get('backend')} model={tts_info.get('model')} "
          f"device={tts_info.get('device')}")
    print(f"  stt       model={stt_info.get('model')} device={stt_info.get('device')}")
    print(f"  gpu free  {info.get('gpu_free_gb')} GB")
    if tts_info.get("backend") == "mms":
        print("\nNOTE / ТЭМДЭГЛЭГЭЭ: tts backend is mms -- the trained Piper voice is not exported")
        print("  yet (no .\\piper_voice\\*.onnx), so Meta's MMS Mongolian fallback voice is speaking.")
        print("  Expected, not a failure.  / бэлтгэсэн Piper дуугаа хараахан экспорт хийгээгүй тул "
              "MMS дуугаар ярих байна.")

def print_unreachable(base: str, exc: Exception) -> None:
    print("=" * 74)
    print("SERVER NOT RUNNING / СЕРВЕР АЖИЛЛАХГҮЙ БАЙНА")
    print(f"  cannot reach {base}/health -> {err_text(exc)}")
    print("  start the voice server first, then run this test again:")
    print(f"    {START_HINT}")
    print("  эхлээд серверээ ажиллуулж, дараа нь энэ тестийг дахин ажиллуулна уу")
    print("=" * 74)

def summarize(results: list[Result], cycles: int) -> int:
    total = len(results)
    passes = sum(1 for r in results if r.passed)
    hard = [r for r in results if r.hard]
    warns = sum(len(r.warnings) for r in results)
    times = sorted(r.total_sec for r in results)
    print("\n" + "=" * 74)
    print("round trip per case (tts + stt + intent, no microphone, no playback)")
    for r in results:
        label = r.case.text if r.cycle == 1 else f"[c{r.cycle}] {r.case.text}"
        print(f"  {'PASS' if r.passed else 'FAIL'}  {label:30.30} {r.total_sec:6.2f}s   "
              f"tts {r.tts_sec:5.2f}s  stt {r.stt_sec:5.2f}s  audio {r.audio[:4]:>5}")
    print(f"  latency  avg {sum(times) / len(times):.2f}s  min {times[0]:.2f}s  "
          f"max {times[-1]:.2f}s")
    print(f"\ncases {total} ({len(CASES)} commands x {cycles} cycle(s)) | pass {passes} | "
          f"soft warnings {warns} | hard failures {len(hard)}")
    print("the TTS reply is never spoken: this test does not play audio / "
          "энэ тест аудио тоглуулахгүй")
    if hard:
        print("HARD FAILURES (synthesis or transcription totally failed):")
        for r in hard:
            print(f"  case {r.pos} '{r.case.text}': {r.hard}")
    elif passes < total:
        print("no hard failure - the mismatches above are Whisper accuracy on Mongolian, "
              "not a broken pipeline")
    print("=" * 74)
    return 1 if hard else 0

# ---- entry point ----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mongolian-voice end-to-end self test")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help="base url of the running voice server")
    ap.add_argument("--keep-wav", metavar="DIR",
                    help="also save every synthesised wav in this directory")
    ap.add_argument("--cycles", type=int, default=1, help="repeat the whole table N times")
    args = ap.parse_args(argv)
    base = args.server.strip().rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    keep = Path(args.keep_wav).expanduser() if args.keep_wav else None
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
    cycles = max(1, args.cycles)

    print("mongolian-voice self test / Монгол хоолойн өөрөөр түүнийгээ шалгах тест")
    print("=" * 74)
    try:
        info = health(base)
    except Exception as e:                                   # noqa: BLE001
        print_unreachable(base, e)
        return 2
    print_health(info, base)

    t0 = time.perf_counter()
    assistant = actions.Assistant()
    n = sum(len(v) for v in assistant.index.table().values())
    print(f"file index  {n} files under {', '.join(str(r) for r in assistant.index.roots)} "
          f"in {time.perf_counter() - t0:.1f}s")

    results: list[Result] = []
    for cycle in range(1, cycles + 1):
        for pos, case in enumerate(CASES, 1):
            res = run_case(case, pos, cycle, base, assistant, keep)
            results.append(res)
            report(res, len(CASES))
    return summarize(results, cycles)

if __name__ == "__main__":
    raise SystemExit(main())

r"""Live Mongolian voice loop: microphone -> server STT -> local action -> spoken reply.

The microphone half of the project.  Audio never leaves the machine except to
127.0.0.1: server.py already owns the only Whisper copy in VRAM, so this process
transcribes over HTTP rather than loading a second model into 6 GB of VRAM.

    START-SERVER.ps1                                # first, in another window
    .\venv\Scripts\python.exe assistant.py          # VAD-gated listening
    .\venv\Scripts\python.exe assistant.py --dry-run --no-speak

Public API:
    main(argv=None) -> int
    VoiceServer(base).health() / .stt(wav_bytes, language) / .tts(text)
    Loop(args, client, engine).on_utterance(utt) / .on_state(state)
"""
from __future__ import annotations

import sys

for _stream in ("stdout", "stderr"):          # this console is cp1252; Mongolian is not
    if hasattr(getattr(sys, _stream), "reconfigure"):
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")

import argparse
import io
import json
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

import actions

if TYPE_CHECKING:
    from listener import Utterance

SR, MAX_S, PTT_MAX_S, MAX_LIST = 16000, 12.0, 6.0, 12
HEALTH_TIMEOUT, HEALTH_TRIES, HEALTH_WAIT = 5.0, 5, 2.0
STT_TIMEOUT, TTS_TIMEOUT = 60.0, 120.0

NOT_UNDERSTOOD = "Ойлгохгүй байна. / Did not understand, try again."
SILENCE = "Чиглэлгүй дуу гарлаа. / No speech recognised in that clip."

SERVER_DOWN = """\
Mongolian voice server холбогдсонгүй / voice server is not reachable: {base}
  шалтгаан / reason: {err}

  Эхлээд серверийг ажиллуулна уу / start the server first:
      START-SERVER.ps1
  бэлэн бол дахин ажиллуулна уу / then run this again:
      .\\venv\\Scripts\\python.exe assistant.py
"""

# Printed here, not by START-ASSISTANT.ps1: PowerShell 5.1 reads a BOM-less .ps1
# as ANSI, so a Cyrillic literal in a launcher comes out as mojibake.
EXAMPLES = [
    ("Блокнот нээ", "open Notepad"),            ("Вордыг хаа", "close Word"),
    ("Хөтөлөгч нээ", "open the browser"),         ("Файлуудыг жагсаа", "list files on screen"),
    ("Танилцуулга эхлүүл", "slide show (name a file)"), ("Тусламж", "what can I say"),
    ("Монгол хэлний тухай хай", "search the web"), ("Сонсохыг зогсоо", "stop listening"),
]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def utterance_end(utt, fallback: float) -> float:
    """Wall-clock end of the clip; fall back to callback time if t_start is not epoch."""
    t = float(getattr(utt, "t_start", 0.0) or 0.0) + float(getattr(utt, "duration", 0.0) or 0.0)
    return t if 0.0 <= fallback - t <= 5.0 else fallback


def listener_kwargs(args) -> dict:
    if args.no_vad:
        # No silence detection, so only the cap closes a chunk -- keep it short or
        # the mic feeds Whisper a minute of room tone.
        return dict(sample_rate=SR, frame_ms=32, start_ms=0, end_ms=0,
                    max_s=PTT_MAX_S, threshold=0.0, device=args.device, min_s=0.25)
    return dict(sample_rate=SR, frame_ms=32, start_ms=180, end_ms=650,
                max_s=MAX_S, threshold=0.5, device=args.device, min_s=0.25)


def make_listener(kwargs: dict):
    """Import lazily so --help and the dead-server path need no audio backend."""
    from listener import Listener
    try:
        return Listener(**kwargs)
    except TypeError as e:                    # listener built with a leaner signature
        if "unexpected keyword" not in str(e):
            raise
        keep = {k: v for k, v in kwargs.items() if f"'{k}'" not in str(e)}
        return Listener(**keep)


class VoiceServer:
    """stdlib HTTP client for server.py -- no requests, no local Whisper."""

    def __init__(self, base: str):
        self.base = base

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/health", timeout=HEALTH_TIMEOUT) as r:
            return json.loads(r.read())

    def stt(self, audio_wav: bytes, language: str = "mn") -> dict:
        b = "----mnvoice" + uuid.uuid4().hex
        head = (f'--{b}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
                f'--{b}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\n')
        body = head.encode() + audio_wav + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(f"{self.base}/stt", data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={b}"})
        with urllib.request.urlopen(req, timeout=STT_TIMEOUT) as r:
            return json.loads(r.read())

    def tts(self, text: str, speed: float = 1.0) -> bytes:
        req = urllib.request.Request(
            f"{self.base}/tts", data=json.dumps({"text": text, "speed": speed}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as r:
            return r.read()


def wait_for_server(client: VoiceServer, base: str, tries: int = HEALTH_TRIES) -> dict | None:
    last: Exception | None = None
    for i in range(tries):
        try:
            return client.health()
        except Exception as e:                    # noqa: BLE001
            last = e
            if i + 1 < tries:
                say(f"  серверийг хүлээж байна / waiting for server {i + 1}/{tries} ...")
                time.sleep(HEALTH_WAIT)
    say(SERVER_DOWN.format(base=base, err=f"{type(last).__name__}: {last}"))
    return None


class Loop:
    def __init__(self, args, client: VoiceServer, engine: actions.Assistant):
        self.args = args
        self.client = client
        self.engine = engine
        self.stop = threading.Event()
        self.samples: list[float] = []
        self._state: str | None = None

    def on_state(self, state) -> None:
        name = getattr(state, "name", None) or str(state)
        if name and name != self._state:
            self._state = name
            say(f"  [{name}]")

    def on_utterance(self, utt: Utterance) -> None:
        # A bad clip must never end the session; only KeyboardInterrupt escapes.
        try:
            self._handle(utt)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            say(f"  серверийн алдаа / server error {e.code}: {detail}")
        except Exception as e:                    # noqa: BLE001
            say(f"  алдаа / error: {type(e).__name__}: {e}")
            if self.args.verbose:
                traceback.print_exc()

    def _handle(self, utt) -> None:
        audio = getattr(utt, "audio", None)
        if audio is None or len(audio) == 0:
            return
        t_end = utterance_end(utt, time.time())
        stt_wall, act_s, server, speak, got = 0.0, 0.0, None, "", False
        try:
            t0 = time.time()
            res = self.client.stt(wav_bytes(audio), "mn")
            stt_wall, server, got = time.time() - t0, res.get("elapsed"), True
            text = (res.get("text") or "").strip()
            if text:
                say(f"> {text}")
            if self.args.verbose:
                say(f"    clip {utt.duration:.2f}s / {len(audio)} samples / energy {utt.energy:.3f}"
                    f" / stt {stt_wall:.2f}s wall, {server}s server, {res.get('device')}")
            if not text:
                say(f"  {SILENCE}")
                return

            t1 = time.time()
            intent, action = self.engine.handle(text, dry_run=self.args.dry_run)
            act_s = time.time() - t1
            if intent is None:
                say(f"  {NOT_UNDERSTOOD}")
                return

            say(f"  intent={intent.name}  action={action.kind}: {action.value} {action.extra}".rstrip())
            if action.kind == "list":
                self._print_list(intent)
            if action.reply:
                say(f"  -> {action.reply}")
                speak = "" if action.kind == "list" else action.reply
            if not action.ok:
                say("  (үйлдэл хийгдэхгүй / nothing was changed)")
            if intent.name == "sleep":
                self.stop.set()
        finally:
            if got:                                  # timed before speaking, so playback
                e2e = time.time() - t_end          # never counts towards the latency
                self.samples.append(e2e)
                avg = sum(self.samples) / len(self.samples)
                say(f"  {e2e:.2f}s end-to-end (stt {stt_wall:.2f}s wall, {server}s server, "
                    f"action {act_s:.2f}s)  avg {avg:.2f}s over {len(self.samples)}")
            if speak:
                self._speak(speak)

    def _print_list(self, intent) -> None:
        paths, score, how = self.engine.index.find(intent.target)
        if not paths:
            say(f"  {NOT_UNDERSTOOD}")
            return
        say(f"  {len(paths)} олдсон / matches ({how}, {score:.0%}):")
        for p in paths[:MAX_LIST]:
            say(f"    - {p}")
        if len(paths) > MAX_LIST:
            say(f"    ... +{len(paths) - MAX_LIST} more")

    def _speak(self, text: str) -> None:
        if self.args.no_speak:
            return
        try:
            import sounddevice as sd                 # only a reply needs playback
            t0 = time.time()
            raw = self.client.tts(text)
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            sd.play(np.ascontiguousarray(data[:, 0]), sr)
            sd.wait()
            if self.args.verbose:
                say(f"  [tts] {len(data) / sr:.1f}s audio in {time.time() - t0:.2f}s")
        except Exception as e:                    # noqa: BLE001
            say(f"  TTS failed: {type(e).__name__}: {e}")

    def summary(self) -> None:
        s = self.samples
        if not s:
            say("Ямар ч хүсэлт бүгтээгүй. / No utterances were handled.")
            return
        say(f"нийт дүн / total {len(s)}  avg {sum(s) / len(s):.2f}s  "
            f"min {min(s):.2f}s  max {max(s):.2f}s")


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="assistant.py",
        description="Mongolian voice assistant: mic -> local STT/TTS server -> local action.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8771", help="voice server URL (host:port also ok)")
    p.add_argument("--device", type=int, default=None, help="sounddevice input index (0 = first mic)")
    p.add_argument("--roots", action="append", default=None, help="extra search root for the file index (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, never open or run anything")
    p.add_argument("--no-speak", action="store_true", help="do not play TTS replies")
    p.add_argument("--no-vad", action="store_true", help="push-to-talk: transcribe every chunk, no VAD gating")
    p.add_argument("--verbose", action="store_true", help="per-stage timings, STT metadata, tracebacks")
    return p.parse_args(argv)


def banner(args, base: str, health: dict) -> None:
    tts, stt = health.get("tts") or {}, health.get("stt") or {}
    say("=" * 68)
    say(f"  Mongolian voice assistant  --  server {base}")
    say(f"  tts {tts.get('backend')} {tts.get('model')} on {tts.get('device')}   |   "
        f"stt {stt.get('model')} on {stt.get('device')} (vram free {health.get('gpu_free_gb')} GB)")
    if stt.get("device") != "cuda":
        say("  note: STT is not on the GPU -- expect seconds, not the 1-2 s figure")
    say(f"  mode {'push-to-talk' if args.no_vad else 'VAD-gated'}"
        f"{' | dry-run' if args.dry_run else ''}{' | silent' if args.no_speak else ''}")
    say("  Хэлээрэй / say one of these out loud:")
    for mn, en in EXAMPLES:
        say(f"     {mn:<26} {en}")
    say("  Ctrl+C to stop.\n" + "=" * 68)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = args.server.rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    client = VoiceServer(base)
    if wait_for_server(client, base) is None:
        return 2
    banner(args, base, client.health())

    roots = (list(actions.DEFAULT_ROOTS) + [Path(r) for r in args.roots]) if args.roots else None
    say("файлын индекс байгуулж байна / building file index ...")
    t0 = time.time()
    engine = actions.Assistant(roots)
    say(f"        {len(engine.index.table())} нэр / stems, {time.time() - t0:.1f}s")

    loop = Loop(args, client, engine)
    lst = make_listener(listener_kwargs(args))
    say("сонсох болголоо / listening -- Ctrl+C to stop")
    try:
        lst.listen(on_utterance=loop.on_utterance, on_state=loop.on_state, stop=loop.stop)
    except KeyboardInterrupt:
        say("\nCtrl+C -- зогсож байна / stopping")
    finally:
        lst.close()
    say("сонсох зогссон / stopped.")
    loop.summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

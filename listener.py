"""Microphone capture + voice activity detection for the Mongolian assistant.

Splits a 16 kHz mono stream into speech segments with Silero VAD and hands each
finished segment to a callback.  The state machine lives in `_frames`, which is
shared by the microphone path and by `segment`, so a test on an in-memory buffer
exercises exactly the code the microphone runs.

    from listener import Listener
    Listener().listen(print, on_state=print)      # blocks until the stop event is set

    from listener import segment                   # no hardware needed
    for utt in segment(samples_16k_mono):
        print(utt.duration, utt.t_start, utt.energy)

States reported through on_state: "listening" (idle, waiting for speech), "speech"
(a confirmed utterance is open), "silence" (inside that utterance's end-of-speech
hangover, so speech may yet resume).  Each state is reported once per flip.
"""
from __future__ import annotations

import queue
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from math import gcd

for _s in ("stdout", "stderr"):        # the console on this machine is cp1252
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector
from scipy.signal import resample_poly

try:
    import sounddevice as sd
except Exception:                       # no PortAudio DLL: keep segment() usable
    sd = None

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

_VAD_RATE = 16000       # pysilero_vad v5 is hard-wired to 16 kHz ...
_VAD_CHUNK = 512        # ... and to exactly 512 samples per call
_PAD_MS = 160           # audio kept on both sides of a segment, so onsets/offsets survive
_STALE_S = 0.32         # newest audio kept after a slow callback, ~the pre-roll window
_IDLE, _SPEECH, _DRAIN = 0, 1, 2

_NO_MIC = (
    "Микрофон ашиглаж болохгүй байна: микрофон олдсонгүй, өөр програм барьж авсан, "
    "эсвэл энэ sampling rate дээр ажиллахгүй байна. Listener(device=...) гэж өөр "
    "микрофон зааж болно.\n"
    "No usable microphone: it is missing, busy (another app holds it), or does not "
    "support this sample rate. Close the other app, or pass "
    "Listener(device='<name or index>')."
)
_CLOSED = "Listener хаагдсан байна / this Listener is already closed."


@dataclass(eq=False, repr=False)
class Utterance:
    """One finished speech segment."""

    audio: np.ndarray    # float32, mono, sample_rate
    duration: float      # seconds
    t_start: float       # seconds since listen() began
    energy: float        # mean absolute amplitude

    def __repr__(self) -> str:
        return (f"Utterance(n={self.audio.size}, duration={self.duration:.2f}s, "
                f"t_start={self.t_start:.2f}s, energy={self.energy:.4f})")


def _frame_len(sample_rate: int, frame_ms: int) -> int:
    return max(1, round(sample_rate * max(1, int(frame_ms)) / 1000))


# --------------------------------------------------------------------------
# VAD adapter
# --------------------------------------------------------------------------


class _Vad:
    """Speech probability per frame, hiding the model's fixed 512-sample window."""

    def __init__(self, threshold: float, sample_rate: int) -> None:
        self._model = SileroVoiceActivityDetector()
        self._threshold = threshold
        g = gcd(_VAD_RATE, sample_rate)
        self._up, self._down = _VAD_RATE // g, sample_rate // g
        self._buf = np.empty(0, dtype=np.float32)
        self._last = 0.0

    def reset(self) -> None:
        # reset() clears the LSTM state but leaves the 64-sample left context of the
        # previous utterance behind, so zero that too.
        self._model.reset()
        self._model._context = np.zeros((1, 64), dtype=np.float32)
        self._buf = self._buf[:0]
        self._last = 0.0

    def feed(self, frame: np.ndarray) -> float:
        """Speech probability of one frame; repeats the last one if it was too short."""
        if (self._up, self._down) != (1, 1):
            frame = resample_poly(frame, self._up, self._down).astype(np.float32)
        self._buf = np.concatenate((self._buf, frame)) if self._buf.size else frame
        while self._buf.size >= _VAD_CHUNK:
            self._last = float(self._model.process_array(self._buf[:_VAD_CHUNK]))
            self._buf = self._buf[_VAD_CHUNK:]
        return self._last


# --------------------------------------------------------------------------
# the state machine
# --------------------------------------------------------------------------


def _make(cur: list, sample_rate: int, min_samples: int, t_start: float,
          n_pad: int, quiet_run: int) -> Utterance | None:
    drop = max(0, quiet_run - n_pad)   # the hangover past the pad only proved the pause
    audio = np.concatenate(cur[: max(1, len(cur) - drop)]).astype(np.float32, copy=False)
    if audio.size < min_samples:
        return None
    return Utterance(audio=audio, duration=audio.size / sample_rate, t_start=t_start,
                     energy=float(np.mean(np.abs(audio))))


def _frames(frames: Iterable[np.ndarray], sample_rate: int, frame_ms: int, start_ms: int,
            end_ms: int, max_s: float, threshold: float, min_s: float, *,
            t0: float = 0.0, flush: bool = True,
            on_state: Callable[[str], None] | None = None) -> Iterator[Utterance]:
    """Turn a stream of equal-length frames into utterances.

    Speech opens after start_ms of consecutive voiced frames, closes after end_ms
    of silence, and is cut hard at max_s; segments shorter than min_s are dropped.
    """
    frame_ms = max(1, int(frame_ms))
    frame_s = frame_ms / 1000.0
    n_start = max(1, round(start_ms / frame_ms))
    n_end = max(1, round(end_ms / frame_ms))
    n_max = max(1, round(max_s / frame_s))
    n_pad = max(0, round(_PAD_MS / frame_ms))
    min_samples = int(min_s * sample_rate)

    vad = _Vad(threshold, sample_rate)
    pre: deque[np.ndarray] = deque(maxlen=n_pad)
    cur: list[np.ndarray] = []
    state, voiced_run, quiet_run, t_start, index = _IDLE, 0, 0, 0.0, 0
    last = ""

    def say(what: str) -> None:
        nonlocal last
        if what != last:
            last = what
            if on_state is not None:
                on_state(what)

    say("listening")

    for frame in frames:
        t = t0 + index * frame_s
        index += 1
        hot = vad.feed(frame) >= threshold

        if state == _DRAIN:
            # a max_s cut already emitted this speech; hold the VAD warm but keep no
            # audio, so the rest of one long utterance cannot return as a second segment
            quiet_run = 0 if hot else quiet_run + 1
            if quiet_run >= n_end:
                state, quiet_run, voiced_run = _IDLE, 0, 0
                say("listening")
            continue

        if state == _IDLE:
            pre.append(frame)
            voiced_run = voiced_run + 1 if hot else 0
            if voiced_run < n_start:
                continue
            vad.reset()                       # every segment starts from a clean model
            cur, voiced_run, quiet_run = list(pre), 0, 0
            pre.clear()
            t_start = t - (len(cur) - 1) * frame_s
            state = _SPEECH
            say("speech")
            continue

        cur.append(frame)
        if hot:
            if quiet_run:
                quiet_run = 0
                say("speech")
        elif not quiet_run:
            quiet_run = 1
            say("silence")
        else:
            quiet_run += 1

        cut = len(cur) >= n_max
        if cut or quiet_run >= n_end:
            utt = _make(cur, sample_rate, min_samples, t_start, n_pad, quiet_run)
            state, cur, quiet_run, voiced_run = (_DRAIN if cut else _IDLE), [], 0, 0
            say("silence" if cut else "listening")
            if utt is not None:
                yield utt

    if flush and state == _SPEECH:     # speech that ran into the end of the buffer
        utt = _make(cur, sample_rate, min_samples, t_start, n_pad, quiet_run)
        if utt is not None:
            yield utt


def _slice(samples: np.ndarray, n_frame: int) -> Iterator[np.ndarray]:
    """Exact frames only; a ragged tail would desync the frame accounting."""
    for i in range(0, samples.size - n_frame + 1, n_frame):
        yield samples[i: i + n_frame]


def segment(samples: np.ndarray, sample_rate: int = 16000, frame_ms: int = 32,
            start_ms: int = 180, end_ms: int = 650, max_s: float = 12.0,
            threshold: float = 0.5, min_s: float = 0.25) -> list[Utterance]:
    """Run the live state machine over an in-memory buffer. No microphone involved."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return list(_frames(_slice(samples, _frame_len(sample_rate, frame_ms)), sample_rate,
                        frame_ms, start_ms, end_ms, max_s, threshold, min_s))


# --------------------------------------------------------------------------
# microphone
# --------------------------------------------------------------------------


def _pump(blocks: queue.Queue, n_frame: int, stop: threading.Event,
          closed: threading.Event) -> Iterator[np.ndarray]:
    """Re-frame PortAudio blocks into whole VAD frames, stopping when asked."""
    buf = np.empty(0, dtype=np.float32)
    while True:
        try:
            block = blocks.get(timeout=0.05)
        except queue.Empty:
            if stop.is_set() or closed.is_set():
                return
            continue
        if block is None:
            return
        buf = np.concatenate((buf, block)) if buf.size else block
        while buf.size >= n_frame:
            yield buf[:n_frame]
            buf = buf[n_frame:]


def _drain(blocks: queue.Queue, keep: int = 0) -> None:
    """Throw away queued audio, optionally keeping the newest `keep` blocks."""
    recent: deque = deque()
    while True:
        try:
            item = blocks.get_nowait()
        except queue.Empty:
            break
        if item is not None and len(recent) < keep:
            recent.append(item)
    for item in recent:
        blocks.put(item)


class Listener:
    """Blocks a thread on the microphone, emitting one Utterance per speech segment."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 32,
                 start_ms: int = 180, end_ms: int = 650, max_s: float = 12.0,
                 threshold: float = 0.5, device: str | int | None = None,
                 min_s: float = 0.25) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.max_s = max_s
        self.threshold = threshold
        self.min_s = min_s
        self.device = device            # index, or a name / substring PortAudio can match
        self._blocks: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._stream = None

    def listen(self, on_utterance: Callable[[Utterance], None],
               on_state: Callable[[str], None] | None = None,
               stop: threading.Event | None = None) -> None:
        """Open the microphone and call on_utterance until `stop` is set (or close())."""
        if sd is None:
            raise RuntimeError(_NO_MIC)
        if self._closed.is_set():
            raise RuntimeError(_CLOSED)
        stop = stop if stop is not None else threading.Event()

        def _callback(indata, frames, time_info, status):
            # PortAudio reuses indata, so copy it; user code must never run on this
            # thread, or the audio device starves and drops frames
            self._blocks.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

        try:
            stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                    dtype="float32", device=self.device,
                                    callback=_callback)
            stream.start()
        except (sd.PortAudioError, ValueError) as exc:
            raise RuntimeError(_NO_MIC) from exc
        self._stream = stream

        gen = _frames(_pump(self._blocks, _frame_len(self.sample_rate, self.frame_ms),
                            stop, self._closed),
                      self.sample_rate, self.frame_ms, self.start_ms, self.end_ms,
                      self.max_s, self.threshold, self.min_s, flush=False,
                      on_state=on_state)
        stale = max(1, round(_STALE_S / (self.frame_ms / 1000.0)))
        try:
            while not stop.is_set():
                utt = next(gen, None)
                if utt is None:
                    break
                on_utterance(utt)
                _drain(self._blocks, stale)   # audio from while we were busy is stale
        finally:
            gen.close()
            self._stop_stream()

    def close(self) -> None:
        """Stop the stream and unblock listen(). Safe to call from any thread, twice."""
        self._closed.set()
        self._blocks.put(None)
        self._stop_stream()

    def _stop_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for step in (stream.stop, stream.close):
            try:
                step()
            except Exception:
                pass                      # teardown must not crash the caller
        _drain(self._blocks)

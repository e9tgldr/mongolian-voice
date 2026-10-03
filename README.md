# Mongolian Voice — dedicated MN TTS (training) + STT server + voice assistant

A **completely new, Mongolian-only text-to-speech model** trained from scratch
on your own dataset, plus fast **speech-to-text**, behind one local API that
AI agents can call — and a **live voice assistant** built on top of it that
opens files, apps and websites when you speak Mongolian.

- **Dataset**: 17,036 clips / **26.2 h** Mongolian speech @ 16 kHz
  (from `chatterbox-finetuning\MyTTSDataset_combined`, 0 clips skipped).
- **TTS**: Piper VITS — **70.4 M params** (23.7 M generator + 46.7 M
  discriminator) — trained **from scratch** on the RTX 4050. Mongolian-only,
  character-level "phonemes" (espeak-ng has no Mongolian voice, so we ship our
  own 134-symbol Cyrillic map: `piper_data/phonemes.json`).
- **STT**: `faster-whisper` **medium** (whisper-small turned out unusable for
  Mongolian — it collapses into other scripts; medium is far stronger).
  GPU fp16 when VRAM is free, CPU int8 otherwise.
- **Server**: FastAPI on `http://127.0.0.1:8771`. Until the first Piper
  checkpoint is exported, TTS answers with Meta's Mongolian **MMS** base voice
  (`facebook/mms-tts-mon`) so the API is usable from minute one.
- **Assistant**: `assistant.py` + `listener.py` + `actions.py` — say a
  command, pause, it acts. See `ASSISTANT.md`.

## Voice assistant

```powershell
START-ASSISTANT.ps1     # starts the server if needed, then the assistant
```

You talk, you stop talking, then it acts — recognition is not instant (there is
no streaming Mongolian ASR). Full usage, the list of phrases it understands,
and troubleshooting live in **[ASSISTANT.md](ASSISTANT.md)**.

| File | Purpose |
|---|---|
| `assistant.py` | main loop: mic → `/stt` → action → spoken reply, with latency reporting |
| `listener.py` | microphone capture + Silero VAD, cuts speech into utterances |
| `actions.py` | Mongolian sentence → open/close/search/slideshow/file action |
| `selftest.py` | no-microphone end-to-end check (text → speech → text → action) |
| `test_actions.py` | 30 unit tests for the intent engine (stdlib only, no side effects) |
| `START-ASSISTANT.ps1` | launcher: UTF-8 console, waits for the server, prints the banner |

`--dry-run` is honoured by `actions.run()` itself, so no code path can open a
file, start an app, or kill a process while previewing a command.

## Daily use

```powershell
START-SERVER.ps1     # serve on 127.0.0.1:8771
START-TRAIN.ps1      # train/resume TTS training (GPU, hours-days)
EXPORT-VOICE.ps1     # newest checkpoint -> piper_voice\mn_voice.onnx
.\venv\Scripts\python.exe client_example.py   # TTS+STT round-trip test
```

After `EXPORT-VOICE.ps1`, restart the server — it automatically prefers
`piper_voice\*.onnx` (fast ONNX on CPU, no VRAM needed) over the MMS fallback.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /tts` | `{"text": "Сайн байна уу?", "speed": 1.0}` | `audio/wav` |
| `POST /v1/audio/speech` | OpenAI-compatible (same fields, `input=`) | `audio/wav` |
| `POST /stt` | multipart `file=@x.wav` (`language=mn`) | `{"text": ...}` |
| `POST /v1/audio/transcriptions` | OpenAI-compatible multipart | `{"text": ...}` |
| `GET /health` | — | backend/model/device/VRAM |

OpenAI-compatible, so existing AI tools work with zero code:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8771/v1", api_key="local")
c.audio.speech.create(model="mn-voice", voice="alloy", input="Сайн байна уу?")
```

```powershell
curl.exe -s http://127.0.0.1:8771/tts -H "Content-Type: application/json" -d '{"text":"Сайн байна уу?"}' -o out.wav
curl.exe -s http://127.0.0.1:8771/stt -F file=@out.wav
```

Repeat texts are cached (instant); long texts auto-split by sentence.

## How the training works

- Piper (`piper-tts[train]`, LightningCLI) — `python -m piper.train fit`,
  CSV `wav|text`, `--data.phoneme_type text` (raw Cyrillic chars as symbols),
  16 kHz, batch 8, checkpoints every 2 epochs under `lightning_logs/`.
- **Auto-resume**: START-TRAIN.ps1 finds the newest `*.ckpt` and passes
  `--ckpt_path`. Stop/close anytime; relaunch to continue.
- **Why a patch exists**: piper's alignment step needs a compiled Cython
  extension and this machine has no MSVC. `numpy_monotonic_align.py` is a
  pure-numpy reimplementation of the original VITS algorithm, verified
  39/39 path cases against a brute-force reference
  (`test_monotonic_align.py`). START-TRAIN.ps1 re-applies it after any
  `pip install` (the patch lives inside the venv's piper package).
- **Dead end documented**: `transformers` VitsModel cannot train
  (`NotImplementedError: Training of VITS is not supported yet`) — that's why
  the first attempt (`train_mn.py`, kept for reference) is unused.
- Expect usable speech after ~15-30 epochs (overnight+); export + listen and
  re-export later checkpoints as training continues.

## Resource notes (RTX 4050 6 GB / 16 GB RAM)

- While training runs, the server is forced to CPU (`MN_TTS_DEVICE=cpu`,
  `MN_STT_DEVICE=cpu` in START-SERVER.ps1) so the GPU belongs to training.
  After training, delete those two lines for GPU STT (fp16) — much faster.
- piper's preprocessing cache (`piper_data/cache`, ~12 GB) must stay while
  training/resuming. Delete it only after the model is final (rebuildable via
  prepare_data.py + make_piper_csv.py + a fresh run).
- Fallback STT model/VRAM knobs: `MN_STT_MODEL` (small/medium/...),
  `MN_STT_DEVICE` (auto/cuda/cpu), `MN_TTS_DEVICE` (auto/cuda/cpu).

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | manifests from the dataset (0.8–11.5 s, ≤240 chars) |
| `make_piper_csv.py` / `make_phonemes.py` | piper CSV + the Cyrillic symbol map |
| `numpy_monotonic_align.py` (+`test_monotonic_align.py`) | verified numpy alignment patch |
| `train_mn.py` | (unused) first attempt — transformers VITS can't train |
| `server.py` | FastAPI TTS+STT server, port 8771 |
| `client_example.py` | round-trip smoke test |
| `diag_stt.py` / `diag_stt2.py` / `download_medium.py` | network/cache repair utilities |
| `START-TRAIN.ps1` / `START-SERVER.ps1` / `EXPORT-VOICE.ps1` | launchers |

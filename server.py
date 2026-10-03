"""Mongolian voice server: TTS + STT behind one fast local API.

TTS  : the Mongolian-only VITS model trained in this folder (.\model),
       falling back to Meta's Mongolian MMS checkpoint before first training run.
STT  : faster-whisper 'small' (supports Mongolian), GPU if VRAM allows else CPU.

Endpoints:
  POST /tts                      {text, speed?}                     -> audio/wav
  POST /v1/audio/speech          {model?, input, speed?}            -> audio/wav   (OpenAI-compatible)
  POST /stt                      multipart file, language?          -> {text, ...}
  POST /v1/audio/transcriptions  multipart file                     -> {text}      (OpenAI-compatible)
  GET  /health

Run: python server.py   (listens on 127.0.0.1:8771)
"""
import hashlib
import io
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".hf"))

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly
from transformers import VitsModel, VitsTokenizer

ROOT = Path(__file__).parent
TTS_DIR = ROOT / "model"
BASE = "facebook/mms-tts-mon"
PORT = 8771
MAX_CHARS = 1200

# Whisper drifts into Latin/other scripts on Mongolian audio unless the decoder
# is anchored with a Mongolian-script prompt. Without it "Сайн байна уу" comes
# back as " sain baina uu".
MN_PROMPT = os.environ.get(
    "MN_STT_PROMPT",
    "Сайн байна уу? Бид монгол хэлээр яриа. "
    "Өдөр хоногийн цаг, файл, программ нээх, хаах, хайх, танилцуулга.",
)

state = {}

_tts_pref = os.environ.get("MN_TTS_DEVICE", "auto")
if _tts_pref == "cpu" or not torch.cuda.is_available():
    state["tts_device"] = "cpu"
elif _tts_pref == "cuda":
    state["tts_device"] = "cuda"
else:
    state["tts_device"] = "cuda" if torch.cuda.is_available() else "cpu"


def vram_free_gb():
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return 0.0


@asynccontextmanager
async def lifespan(app):
    # ---- TTS (piper ONNX voice when trained, else base Mongolian MMS) ----
    voice_dir = ROOT / "piper_voice"
    onnx_files = sorted(voice_dir.glob("*.onnx")) if voice_dir.exists() else []
    if onnx_files:
        from piper import PiperVoice

        state["tts_backend"] = "piper"
        state["tts_src"] = onnx_files[0].stem
        state["piper_voice"] = PiperVoice.load(
            str(onnx_files[0]), config_path=str(onnx_files[0]) + ".json")
        state["sr"] = state["piper_voice"].config.sample_rate
        state["tts_device"] = "cpu-onnx"
        torch.set_num_threads(8)
        print(f"[tts] piper voice {onnx_files[0].name} ready (onnx)", flush=True)
    else:
        state["tts_backend"] = "mms"
        src = str(TTS_DIR) if (TTS_DIR / "config.json").exists() else BASE
        print(f"[tts] loading {src} ...", flush=True)
        state["tts_src"] = src
        state["tokenizer"] = VitsTokenizer.from_pretrained(src)
        state["tts"] = VitsModel.from_pretrained(src).eval()
        state["sr"] = state["tts"].config.sampling_rate
        if state["tts_device"] == "cuda":
            state["tts"] = state["tts"].to("cuda")
        torch.set_num_threads(8)
        print(f"[tts] ready on {state['tts_device']}", flush=True)

    # ---- STT ----
    # ctranslate2 needs cuBLAS/cuDNN DLLs; torch bundles them in torch/lib
    if torch.cuda.is_available():
        torchlib = Path(torch.__file__).parent / "lib"
        os.add_dll_directory(str(torchlib))
        os.environ["PATH"] = str(torchlib) + os.pathsep + os.environ.get("PATH", "")
    from faster_whisper import WhisperModel

    def local_whisper_dir(name):
        """Resolve an already-downloaded Systran model from the HF cache
        (avoids faster-whisper's download_root re-downloading everything)."""
        for p in (ROOT / ".hf" / "hub").glob(f"models--Systran--faster-whisper-{name}"):
            for snap in sorted(p.glob("snapshots/*/model.bin")):
                if snap.stat().st_size > 1_000_000:
                    return str(snap.parent)
        return None

    stt_model = os.environ.get("MN_STT_MODEL", "medium")
    pref = os.environ.get("MN_STT_DEVICE", "auto")
    if pref == "cuda":
        stt_dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif pref == "cpu":
        stt_dev = "cpu"
    else:
        stt_dev = "cuda" if torch.cuda.is_available() and vram_free_gb() > 2.2 else "cpu"
    compute = "float16" if stt_dev == "cuda" else "int8"
    stt_src = local_whisper_dir(stt_model) or stt_model
    print(f"[stt] loading {stt_src} ({stt_dev} {compute}) ...", flush=True)
    try:
        state["asr"] = WhisperModel(stt_src, device=stt_dev, compute_type=compute,
                                    download_root=str(ROOT / ".hf"))
        state["stt_device"] = stt_dev
    except Exception as e:
        print(f"[stt] {stt_dev} init failed ({e}); falling back to CPU int8", flush=True)
        state["asr"] = WhisperModel(stt_src, device="cpu", compute_type="int8",
                                    download_root=str(ROOT / ".hf"))
        state["stt_device"] = "cpu"
    state["stt_model"] = stt_model
    print(f"[stt] ready on {state['stt_device']}", flush=True)
    state["cache"] = {}
    yield


app = FastAPI(title="Mongolian Voice API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class SpeechIn(BaseModel):
    text: str
    speed: float = 1.0
    # OpenAI-compat aliases
    input: str | None = None
    model: str | None = None


def sentence_chunks(text: str, n: int = 220):
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?։;])\s+", text)
    out, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 <= n:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                out.append(cur)
            while len(p) > n:
                cut = p.rfind(" ", n // 2, n)
                cut = cut if cut > 0 else n
                out.append(p[:cut].strip())
                p = p[cut:].strip()
            cur = p
    if cur:
        out.append(cur)
    return out or [text]


def to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    pad = np.zeros(int(sr * 0.05), dtype="float32")
    wav = np.concatenate([pad, wav, pad])
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synthesize(text: str, speed: float = 1.0) -> bytes:
    text = text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"text too long (>{MAX_CHARS} chars); split it")
    key = hashlib.md5(f"{speed}|{text}".encode()).hexdigest()
    cache = state["cache"]
    if key in cache:
        return cache[key]
    if len(cache) > 512:
        cache.clear()

    if state.get("tts_backend") == "piper":
        from piper import SynthesisConfig

        pv = state["piper_voice"]
        syn = SynthesisConfig(length_scale=max(0.5, 1.0 / max(speed, 0.25)),
                              normalize_audio=True)
        pieces = [c.audio_float_array for c in pv.synthesize(text, syn_config=syn)]
        wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
        audio = to_wav_bytes(wav.astype("float32"), state["sr"])
        cache[key] = audio
        return audio

    tok, tts = state["tokenizer"], state["tts"]
    dev = state["tts_device"]
    sil = np.zeros(int(tts.config.sampling_rate * 0.12), dtype="float32")
    pieces = []
    for chunk in sentence_chunks(text):
        enc = tok(chunk)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
        am = torch.ones_like(ids)
        try:
            if dev == "cuda":
                ids, am = ids.to("cuda"), am.to("cuda")
            tts.config.speaking_rate = speed
            with torch.no_grad():
                out = tts(input_ids=ids, attention_mask=am)
            pieces.append(out.waveform[0].float().cpu().numpy())
            pieces.append(sil)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # retry this chunk on CPU
            ids, am = ids.cpu(), am.cpu()
            tts = tts.to("cpu")
            state["tts_device"] = "cpu"
            dev = "cpu"
            with torch.no_grad():
                out = tts(input_ids=ids, attention_mask=am)
            pieces.append(out.waveform[0].float().cpu().numpy())
            pieces.append(sil)
    if pieces and len(pieces[-1]) < sil.shape[0]:
        pieces.pop()
    wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
    sr = state["tts"].config.sampling_rate
    audio = to_wav_bytes(wav, sr)
    cache[key] = audio
    return audio


@app.post("/tts")
def tts_endpoint(body: SpeechIn):
    text = body.text or body.input
    t0 = time.time()
    audio = synthesize(text, body.speed)
    dur = len(audio) / 2 / state["sr"]
    return Response(audio, media_type="audio/wav",
                    headers={"X-Audio-Duration-Sec": f"{dur:.2f}",
                             "X-Synth-Time-Sec": f"{time.time()-t0:.2f}"})


@app.post("/v1/audio/speech")
def openai_speech(body: SpeechIn):
    return tts_endpoint(body)


@app.post("/stt")
@app.post("/v1/audio/transcriptions")
async def stt_endpoint(file: UploadFile = File(...),
                       language: str = Form("mn")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(400, "unsupported audio format (use wav/mp3/flac/m4a)")
    wav = wav[:, 0] if wav.shape[1] > 1 else wav.reshape(-1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype("float32")
    t0 = time.time()
    prompt = MN_PROMPT if language == "mn" else None
    segments, info = state["asr"].transcribe(wav, language=language, beam_size=1,
                                             vad_filter=True, initial_prompt=prompt)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "audio_duration": round(info.duration, 2),
        "elapsed": round(time.time() - t0, 2),
        "device": state["stt_device"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tts": {"backend": state.get("tts_backend"), "model": state.get("tts_src"),
                "device": state.get("tts_device")},
        "stt": {"model": state.get("stt_model"), "device": state.get("stt_device")},
        "gpu_free_gb": round(vram_free_gb(), 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

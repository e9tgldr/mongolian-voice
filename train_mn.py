"""Train the Mongolian-only TTS model (VITS) on the local dataset.

Starts from Meta's Mongolian MMS checkpoint (facebook/mms-tts-mon) — a
Mongolian-only VITS — and fine-tunes it on ~25 h of local Mongolian speech
(MyTTSDataset_combined). Result is a NEW dedicated Mongolian model saved to
.\model. Runs on the RTX 4050 GPU.

Usage:
  python train_mn.py [--batch 3] [--accum 6] [--epochs 6] [--lr 1e-4]
Resumes automatically from the latest checkpoint in .\ckpt.
"""
import argparse
import inspect
import json
import os
from math import gcd
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".hf"))

import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, VitsModel, VitsTokenizer

ROOT = Path(__file__).parent
BASE = "facebook/mms-tts-mon"
CKPT = ROOT / "ckpt"
FINAL = ROOT / "model"
DATA = ROOT / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--accum", type=int, default=6)
parser.add_argument("--epochs", type=float, default=6)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

print("Loading base Mongolian model + tokenizer ...", flush=True)
tokenizer = VitsTokenizer.from_pretrained(BASE)
model = VitsModel.from_pretrained(BASE)
SR = model.config.sampling_rate
FWD_PARAMS = set(inspect.signature(model.forward).parameters)
print(f"model sampling_rate = {SR}", flush=True)


class MnTTS(Dataset):
    def __init__(self, manifest_path):
        self.rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        wav = wav[:, 0]
        if sr != SR:
            g = gcd(int(sr), int(SR))
            wav = resample_poly(wav, SR // g, sr // g).astype("float32")
        ids = tokenizer(r["text"])["input_ids"]
        if not ids:
            ids = tokenizer(" ")["input_ids"]
        return {"input_ids": ids, "waveform": wav}


def collate(batch):
    max_id = max(len(b["input_ids"]) for b in batch)
    max_w = max(len(b["waveform"]) for b in batch)
    ids = torch.zeros(len(batch), max_id, dtype=torch.long)
    am = torch.zeros(len(batch), max_id, dtype=torch.long)
    wav = torch.zeros(len(batch), max_w, dtype=torch.float32)
    wlen = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        am[i, :n] = 1
        m = len(b["waveform"])
        wav[i, :m] = torch.from_numpy(b["waveform"])
        wlen[i] = m
    out = {"input_ids": ids, "attention_mask": am, "waveform": wav,
           "waveform_lengths": wlen, "labels": wav}
    # only pass kwargs the model's forward actually accepts
    return {k: v for k, v in out.items() if k in FWD_PARAMS}


train_ds = MnTTS(DATA / "train_manifest.jsonl")
val_ds = MnTTS(DATA / "val_manifest.jsonl")
print(f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

targs = TrainingArguments(
    output_dir=str(CKPT),
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    warmup_steps=500,
    lr_scheduler_type="linear",
    logging_steps=25,
    save_steps=1500,
    eval_strategy="steps",
    eval_steps=1500,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to=[],
    seed=42,
    save_safetensors=True,
)

trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)

checkpoints = sorted(CKPT.glob("checkpoint-*"))
resume = str(checkpoints[-1]) if checkpoints else None
if resume:
    print(f"Resuming from {resume}", flush=True)
trainer.train(resume_from_checkpoint=resume)

FINAL.mkdir(exist_ok=True)
model.save_pretrained(FINAL)
tokenizer.save_pretrained(FINAL)
print(f"DONE — final Mongolian model saved to {FINAL}", flush=True)

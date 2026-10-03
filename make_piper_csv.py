"""Convert the local manifests into piper's CSV format (wav|text)."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
rows = []
for name in ("train_manifest.jsonl", "val_manifest.jsonl"):
    p = ROOT / "data" / name
    rows += [json.loads(l) for l in open(p, encoding="utf-8")]

out = ROOT / "piper_data"
out.mkdir(exist_ok=True)
with open(out / "metadata.csv", "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(f"{Path(r['audio']).name}|{r['text']}\n")
print(f"{len(rows)} utterances -> {out / 'metadata.csv'}")

"""Build the phoneme/id map for Mongolian char-level piper training.

Scans the real corpus (piper NFD-normalizes raw text chars), then assigns
ids: specials (_ ^ $ space + punctuation) first, then every observed char.
"""
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
chars = set()
for line in open(ROOT / "piper_data" / "metadata.csv", encoding="utf-8"):
    parts = line.rstrip("\n").split("|", 1)
    if len(parts) == 2:
        chars.update(unicodedata.normalize("NFD", parts[1]))

SPECIALS = ["_", "^", "$", " ", "!", "'", "(", ")", ",", "-", ".", ":", ";",
            "?", "\u0306"]  # u0306 = combining breve (NFD of й)
for s in SPECIALS:
    chars.discard(s)
assert not (chars & {"_", "^", "$"}), "corpus contains piper special chars!"
ordering = SPECIALS + sorted(chars)
assert len(ordering) <= 256, f"too many symbols: {len(ordering)}"

mapping = {c: [i] for i, c in enumerate(ordering)}
(ROOT / "piper_data" / "phonemes.json").write_text(
    json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")

cyr = sum(1 for c in ordering if "\u0400" <= c <= "\u04FF")
print(f"{len(ordering)} symbols ({cyr} Cyrillic) -> piper_data/phonemes.json")
print("inventory:", "".join(repr(c) if c in "\n\t" else c for c in ordering))

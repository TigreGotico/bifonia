"""
Export the bifonia labeled corpus as CSV, JSON, and HuggingFace-compatible splits.

Output files
------------
dataset.csv          — full corpus: word, sense, pos, ipa, diacritized, sentence, diacritized_sentence
dataset.json         — same as JSON array
hf/train.jsonl       — 80% split, stratified per (word, sense), for HF upload
hf/test.jsonl        — 20% split

The bucket key is MEANING (`sense`); `pos` is a descriptive attribute. The
`diacritized*` columns carry the disambiguating diacritic (acute = open vowel,
circumflex = closed) restored on the homograph — the target for a
diacritics-restoration model.

Usage::

    python dataset.py [--out <dir>] [--hf]
"""

import csv
import json
import argparse
import pathlib
import random
import re
from collections import Counter, defaultdict

from bifonia import HOMOGRAPHS
from bifonia.data import SENSE_POS
from bifonia.__init__ import _DIACRITIZED
from bifonia.corpus import iter_records


def _diacritized_sentence(sentence: str, word: str, sense: str) -> str:
    diac = _DIACRITIZED.get((word, sense))
    if not diac:
        return sentence
    return re.sub(rf"\b{re.escape(word)}\b", diac, sentence, flags=re.IGNORECASE)


def build_records() -> list[dict]:
    records = []
    for word, sense, sentence in iter_records():
        records.append({
            "word": word,
            "sense": sense,
            "pos": SENSE_POS.get(word, {}).get(sense, ""),
            "ipa": HOMOGRAPHS.get(word, {}).get(sense, ""),
            "diacritized": _DIACRITIZED.get((word, sense), word),
            "sentence": sentence,
            "diacritized_sentence": _diacritized_sentence(sentence, word, sense),
        })
    return records


def stratified_split(records: list[dict], test_frac: float = 0.2,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified, shuffled 80/20 split per (word, sense) so every bucket is
    represented i.i.d. in both splits."""
    rng = random.Random(seed)
    by_class: dict = defaultdict(list)
    for r in records:
        by_class[(r["word"], r["sense"])].append(r)

    train, test = [], []
    for key, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_frac))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export bifonia corpus.")
    parser.add_argument("--out", default=".", help="Output directory (default: current)")
    parser.add_argument("--hf", action="store_true",
                        help="Also write HF train/test JSONL splits under <out>/hf/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = build_records()
    fields = ["word", "sense", "pos", "ipa", "diacritized", "sentence", "diacritized_sentence"]

    csv_path = out / "dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows → {csv_path}")

    json_path = out / "dataset.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records → {json_path}")

    if args.hf:
        hf_dir = out / "hf"
        hf_dir.mkdir(exist_ok=True)
        train, test = stratified_split(records, seed=args.seed)
        write_jsonl(train, hf_dir / "train.jsonl")
        write_jsonl(test, hf_dir / "test.jsonl")
        print(f"HF splits → {hf_dir}/  (train={len(train)}, test={len(test)})")

    by_word = Counter(r["word"] for r in records)
    by_sense = Counter(f"{r['word']}/{r['sense']}" for r in records)
    print(f"\nTotal: {len(records)} sentences across {len(by_word)} words, {len(by_sense)} senses")


if __name__ == "__main__":
    main()

"""
Export the bifonia labeled corpus as CSV, JSON, and HuggingFace-compatible splits.

Output files
------------
dataset.csv          — full corpus: word, pos, ipa, diacritized, sentence, diacritized_sentence
dataset.json         — same as JSON array
hf/train.jsonl       — 80% stratified split (per word+POS) for HF upload
hf/test.jsonl        — 20% stratified split

Usage::

    python dataset.py [--out <dir>] [--hf]
"""

import csv
import json
import argparse
import pathlib
import random
import re
from collections import defaultdict

from bifonia import HOMOGRAPHS
from bifonia.__init__ import _DIACRITIZED
from bifonia.corpus import CORPUS, iter_records


def _diacritized_sentence(sentence: str, word: str, pos: str) -> str:
    diac = _DIACRITIZED.get((word, pos))
    if not diac:
        return sentence
    return re.sub(
        rf"\b{re.escape(word)}\b",
        diac,
        sentence,
        flags=re.IGNORECASE,
    )


def build_records() -> list[dict]:
    records = []
    for word, pos, sentence in iter_records():
        ipa = HOMOGRAPHS.get(word, {}).get(pos, "")
        diac_sent = _diacritized_sentence(sentence, word, pos)
        records.append(
            {
                "word": word,
                "pos": pos,
                "ipa": ipa,
                "diacritized": _DIACRITIZED.get((word, pos), word),
                "sentence": sentence,
                "diacritized_sentence": diac_sent,
            }
        )
    return records


def stratified_split(records: list[dict], test_frac: float = 0.2,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified 80/20 split by (word, pos) so every class is represented in both."""
    rng = random.Random(seed)
    by_class: dict = defaultdict(list)
    for r in records:
        by_class[(r["word"], r["pos"])].append(r)

    train, test = [], []
    for key, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_frac))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export bifonia corpus.")
    parser.add_argument("--out", default=".", help="Output directory (default: current)")
    parser.add_argument("--hf", action="store_true",
                        help="Also write HF-compatible train/test JSONL splits under <out>/hf/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = build_records()

    # Full CSV
    csv_path = out / "dataset.csv"
    fields = ["word", "pos", "ipa", "diacritized", "sentence", "diacritized_sentence"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows → {csv_path}")

    # Full JSON
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

    from collections import Counter
    by_word = Counter(r["word"] for r in records)
    by_pos = Counter(r["pos"] for r in records)
    print(f"\nTotal: {len(records)} sentences across {len(by_word)} words")
    print("POS distribution:", dict(sorted(by_pos.items())))
    print("\nPer-word counts:")
    for w, n in sorted(by_word.items()):
        print(f"  {w}: {n}")


if __name__ == "__main__":
    main()

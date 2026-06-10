"""
Export the bifonia labeled corpus as CSV and JSON datasets.

Output files
------------
dataset.csv   — one row per sentence: word, pos, ipa, diacritized, sentence
dataset.json  — same data as a JSON array

Usage::

    python dataset.py [--out <dir>]
"""

import csv
import json
import argparse
import pathlib
import re

from bifonia import HOMOGRAPHS
from bifonia.__init__ import _DIACRITIZED
from bifonia.corpus import CORPUS, iter_records


def _diacritized_sentence(sentence: str, word: str, pos: str) -> str:
    """Return sentence with the target word replaced by its diacritized form."""
    diac = _DIACRITIZED.get((word, pos))
    if not diac:
        return sentence
    # case-insensitive word-boundary replacement, preserve surrounding case
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


def main():
    parser = argparse.ArgumentParser(description="Export bifonia corpus to CSV/JSON.")
    parser.add_argument("--out", default=".", help="Output directory (default: current)")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = build_records()

    # CSV
    csv_path = out / "dataset.csv"
    fields = ["word", "pos", "ipa", "diacritized", "sentence", "diacritized_sentence"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows → {csv_path}")

    # JSON
    json_path = out / "dataset.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records → {json_path}")

    # Summary
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

"""scripts/benchmark_tagger.py scores records whose homograph carries trailing punctuation."""
import json
import subprocess
import sys
from pathlib import Path

from bifonia import guess_sense, tokenize

_ROOT = Path(__file__).parent.parent


def _public_api_accuracy(word):
    records = [
        json.loads(line)
        for name in ("train.jsonl", "test.jsonl")
        for line in (_ROOT / "hf" / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r["word"] == word]
    hits = 0
    for r in records:
        toks = tokenize(r["sentence"].lower())
        idx = next(i for i, t in enumerate(toks) if t.strip(".,;:!?") == word)
        hits += guess_sense(toks, idx) == r["sense"]
    return len(records), hits / len(records) * 100


def test_rule_column_scores_punctuated_homograph():
    n, accuracy = _public_api_accuracy("sede")
    run = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "benchmark_tagger.py"),
         "--no-taggers", "--word", "sede"],
        capture_output=True, text=True, cwd=_ROOT,
    )
    assert run.returncode == 0, run.stderr
    full = next(l for l in run.stdout.splitlines() if l.startswith("full"))
    cells = [c.strip() for c in full.split("|")[1].split()]
    rules, ensemble = cells[0], cells[4]
    assert (full.split()[1], rules, ensemble) == (str(n), f"{accuracy:.2f}%", f"{accuracy:.2f}%")

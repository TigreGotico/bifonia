"""The committed hf/ split is the output of scripts/dataset.py --hf on the corpus."""
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_HF = _ROOT / "hf"


def _dataset_script():
    spec = importlib.util.spec_from_file_location("dataset", _ROOT / "scripts" / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(name):
    return (_HF / name).read_text(encoding="utf-8").splitlines()


def test_every_split_sentence_is_a_corpus_sentence():
    corpus = {
        (r["word"], r["sense"], r["sentence"])
        for r in map(json.loads, (_ROOT / "bifonia" / "data" / "corpus.jsonl")
                     .read_text(encoding="utf-8").splitlines())
    }
    split = {
        (r["word"], r["sense"], r["sentence"])
        for name in ("train.jsonl", "test.jsonl")
        for r in map(json.loads, _lines(name))
    }
    assert (len(split - corpus), len(corpus - split)) == (0, 0)


def test_committed_split_is_the_script_output():
    dataset = _dataset_script()
    train, test = dataset.stratified_split(dataset.build_records())
    for name, records in (("train.jsonl", train), ("test.jsonl", test)):
        expected = [json.dumps(r, ensure_ascii=False) for r in records]
        committed = _lines(name)
        differing = sum(a != b for a, b in zip(committed, expected))
        assert (name, len(committed), differing) == (name, len(expected), 0)

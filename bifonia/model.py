"""
Pure-stdlib inference for the learned per-word sense models.

Both the Naive-Bayes ("wordlist overlap") and the averaged-perceptron models share
one JSON shape and one scoring rule — a sparse dot product:

    score[sense] = bias[word][sense] + Σ_f feats[f] · weights[word][sense][f]

and the predicted sense is the argmax. For NB the weights are per-sense log-odds and
the bias is the log-prior; for the perceptron they are learned. Either way inference
is plain dict arithmetic — no numpy, no sklearn (only ``json``
+ ``pathlib`` + :mod:`bifonia.features`), so it is safe under the zero-dependency
install.

Model JSON::

    {
      "version": 1, "model_type": "nb" | "perceptron_averaged",
      "config": {"lang": "pt-pt", "window": 4, "min_count": 2},
      "vocs_used": [...],
      "words": {
        "<word>": {
          "route": "model" | "rules",       # per-word adoption flag
          "margin_tau": 0.0,
          "senses": ["mould", "shape"],
          "bias": {"mould": -0.4, "shape": 0.4},
          "weights": {"mould": {"W=forno": 1.8, ...}, "shape": {...}}
        }
      }
    }
"""
import functools
import json
import pathlib

from bifonia.features import extract_features, load_structural_vocs

_DATA = pathlib.Path(__file__).parent / "data"
NB_PATH = _DATA / "sense_model_nb.json"
PERCEPTRON_PATH = _DATA / "sense_model_perceptron.json"


class SenseModel:
    """A loaded per-word sense classifier (NB or perceptron)."""

    def __init__(self, blob: dict):
        self.version = blob.get("version", 1)
        self.model_type = blob.get("model_type", "unknown")
        self.config = blob.get("config", {})
        self.lang = self.config.get("lang", "pt-pt")
        self.words = blob.get("words", {})

    @classmethod
    @functools.lru_cache(maxsize=None)
    def load(cls, path: str = None) -> "SenseModel":
        path = pathlib.Path(path) if path else PERCEPTRON_PATH
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ── per-word metadata ─────────────────────────────────────────────────────
    def has(self, word: str) -> bool:
        return word in self.words

    def route(self, word: str) -> str:
        entry = self.words.get(word)
        return entry.get("route", "rules") if entry else "rules"

    def margin_tau(self, word: str) -> float:
        entry = self.words.get(word)
        return entry.get("margin_tau", 0.0) if entry else 0.0

    # ── scoring ───────────────────────────────────────────────────────────────
    def scores(self, word: str, words: list, idx: int, vocs: dict = None) -> dict:
        entry = self.words[word]
        if vocs is None:
            vocs = load_structural_vocs(self.lang)
        feats = extract_features(words, idx, vocs)
        out = {}
        for sense in entry["senses"]:
            w = entry["weights"].get(sense, {})
            s = entry["bias"].get(sense, 0.0)
            for f, v in feats.items():
                wf = w.get(f)
                if wf:
                    s += v * wf
            out[sense] = s
        return out

    def predict(self, word: str, words: list, idx: int, vocs: dict = None) -> str:
        scores = self.scores(word, words, idx, vocs)
        # deterministic tie-break: highest score, then sense name
        return max(sorted(scores), key=lambda s: scores[s])

    def margin(self, word: str, words: list, idx: int, vocs: dict = None) -> float:
        scores = sorted(self.scores(word, words, idx, vocs).values(), reverse=True)
        return scores[0] - scores[1] if len(scores) > 1 else float("inf")

"""Zero-dependency disambiguation baselines, for honest benchmarking.

These are *evaluation tooling*, deliberately kept out of the ``bifonia`` package
so the shipped inference path stays lean and dependency-free. Each baseline
implements the small :class:`Baseline` protocol, so the harness can fit and score
them uniformly and new baselines drop in without touching the runner.

Baselines here are all **pure stdlib** (no numpy):

* :class:`MostCommon`   — per-word majority sense from training. The floor.
* :class:`CueOnly`      — the sense-cue registry alone (no POS, no learning):
                          isolates how far the curated cues get you.
* :class:`DecisionList` — Yarowsky's classic WSD baseline: rank collocations by
                          log-likelihood ratio, predict by the single most
                          confident matching collocation. Interpretable and a
                          strong, well-understood reference point.

A numpy-only tier (logistic regression on the same features, TF-IDF
nearest-neighbour) is a natural next step — it would implement the same protocol.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import NamedTuple, Protocol

from bifonia import HOMOGRAPHS, tokenize
from bifonia.cues import SENSE_CUES
from bifonia.features import extract_features, load_structural_vocs
from bifonia.text import cue_score, folded
from bifonia.vocab import voc

ALPHA = 0.5  # additive smoothing for the decision-list log-likelihood ratio


class Example(NamedTuple):
    """One labelled training/eval instance."""
    word: str
    words: list[str]
    idx: int
    sense: str


class Baseline(Protocol):
    """The interface every baseline implements so the harness stays generic."""
    name: str

    def fit(self, by_word: dict[str, list[Example]]) -> None:
        """Learn whatever per-word state the baseline needs (may be a no-op)."""

    def predict(self, ex: Example) -> str:
        """Return the predicted sense slug for *ex*."""


def _majority(by_word: dict[str, list[Example]]) -> dict[str, str]:
    """Per-word most-frequent sense in the training data."""
    out: dict[str, str] = {}
    for word, exs in by_word.items():
        out[word] = Counter(e.sense for e in exs).most_common(1)[0][0]
    return out


class MostCommon:
    """Always predict a word's training-majority sense — the baseline floor."""
    name = "most-common"

    def fit(self, by_word: dict[str, list[Example]]) -> None:
        self._majority = _majority(by_word)

    def predict(self, ex: Example) -> str:
        return self._majority.get(ex.word, ex.sense)


class CueOnly:
    """The sense-cue registry with no POS and no learning.

    For a word in :data:`bifonia.cues.SENSE_CUES`, score its cue lists exactly as
    the rule resolver does and take the winner (else the registry default). For
    any other word, fall back to the training majority. This measures the cues'
    standalone contribution — an ablation of the full rule engine.
    """
    name = "cue-only"

    def fit(self, by_word: dict[str, list[Example]]) -> None:
        self._majority = _majority(by_word)

    def predict(self, ex: Example) -> str:
        rule = SENSE_CUES.get(ex.word)
        if rule is None:
            return self._majority.get(ex.word, ex.sense)
        best, best_score = rule.default, 0
        for cue in rule.cues:
            if cue.requires_prev is not None:
                prev = ex.words[ex.idx - 1] if ex.idx > 0 else ""
                if prev.strip(".,;:!?-") != cue.requires_prev:
                    continue
            score = cue_score(ex.words, ex.idx, folded(voc(cue.voc)))
            if score > best_score:
                best, best_score = cue.sense, score
        return best


class DecisionList:
    """Yarowsky-style decision list (one sense per collocation).

    For each word, every feature emitted by :func:`extract_features` (positional
    collocations, window bag-of-words, and the CUE: features) is scored by the
    log-likelihood ratio of its most-associated sense versus the rest. At predict
    time the rules fire in descending confidence and the first feature present in
    the context decides; if none match, fall back to the training majority.
    """
    name = "decision-list"

    def __init__(self) -> None:
        self._vocs = load_structural_vocs("pt-pt")
        self._rules: dict[str, list[tuple[str, str, float]]] = {}

    def fit(self, by_word: dict[str, list[Example]]) -> None:
        self._majority = _majority(by_word)
        for word, exs in by_word.items():
            senses = sorted({e.sense for e in exs})
            if len(senses) < 2:
                continue
            # feature → sense → count
            counts: dict[str, Counter] = defaultdict(Counter)
            for e in exs:
                for feat in extract_features(e.words, e.idx, self._vocs):
                    counts[feat][e.sense] += 1
            rules = []
            for feat, sc in counts.items():
                top_sense, top_n = sc.most_common(1)[0]
                rest = sum(sc.values()) - top_n
                llr = math.log((top_n + ALPHA) / (rest + ALPHA))
                rules.append((feat, top_sense, llr))
            # most confident collocation first
            rules.sort(key=lambda r: r[2], reverse=True)
            self._rules[word] = rules

    def predict(self, ex: Example) -> str:
        rules = self._rules.get(ex.word)
        if not rules:
            return self._majority.get(ex.word, ex.sense)
        present = set(extract_features(ex.words, ex.idx, self._vocs))
        for feat, sense, _ in rules:
            if feat in present:
                return sense
        return self._majority.get(ex.word, ex.sense)


class LogisticRegression:
    """Per-word multinomial logistic regression over :func:`extract_features`.

    The one **numpy** baseline (imported lazily) — every other baseline here is
    pure stdlib. It answers the project's open question directly: given the same
    features the rules use (including the CUE: features), can a learned linear
    model beat the rules out-of-distribution? Trained with full-batch gradient
    descent and L2; falls back to the training majority for unseen words.
    """
    name = "logreg"

    def __init__(self, epochs: int = 400, lr: float = 0.5, l2: float = 1e-4) -> None:
        self._vocs = load_structural_vocs("pt-pt")
        self.epochs, self.lr, self.l2 = epochs, lr, l2
        self._models: dict[str, tuple] = {}   # word → (W, feat_index, senses)

    def fit(self, by_word: dict[str, list[Example]]) -> None:
        import numpy as np
        self._majority = _majority(by_word)
        for word, exs in by_word.items():
            senses = sorted({e.sense for e in exs})
            if len(senses) < 2:
                continue
            feat_index: dict[str, int] = {}
            rows = []
            for e in exs:
                feats = extract_features(e.words, e.idx, self._vocs)
                for f in feats:
                    feat_index.setdefault(f, len(feat_index))
                rows.append((feats, senses.index(e.sense)))
            d, k = len(feat_index) + 1, len(senses)   # +1 bias column
            X = np.zeros((len(rows), d)); Y = np.zeros((len(rows), k))
            for r, (feats, y) in enumerate(rows):
                X[r, -1] = 1.0                          # bias
                for f, v in feats.items():
                    X[r, feat_index[f]] = v
                Y[r, y] = 1.0
            W = np.zeros((d, k))
            for _ in range(self.epochs):
                Z = X @ W; Z -= Z.max(1, keepdims=True)
                P = np.exp(Z); P /= P.sum(1, keepdims=True)
                grad = X.T @ (P - Y) / len(rows) + self.l2 * W
                W -= self.lr * grad
            self._models[word] = (W, feat_index, senses)

    def predict(self, ex: Example) -> str:
        import numpy as np
        model = self._models.get(ex.word)
        if model is None:
            return self._majority.get(ex.word, ex.sense)
        W, feat_index, senses = model
        x = np.zeros(W.shape[0]); x[-1] = 1.0
        for f, v in extract_features(ex.words, ex.idx, self._vocs).items():
            j = feat_index.get(f)
            if j is not None:
                x[j] = v
        return senses[int((x @ W).argmax())]


#: Pure-stdlib baselines, instantiated in benchmark order (cheap → strong).
BASELINES: list[Baseline] = [MostCommon(), CueOnly(), DecisionList()]
#: numpy-only tier (kept separate so the stdlib suite stays dependency-free).
NUMPY_BASELINES: list[Baseline] = [LogisticRegression()]


def load_examples(path: str) -> dict[str, list[Example]]:
    """Group labelled JSONL ({word, sense, sentence}) by word into Examples."""
    import json
    by_word: dict[str, list[Example]] = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        word = r["word"]
        words = tokenize(r["sentence"].lower())
        idx = next((i for i, t in enumerate(words)
                    if t.strip(".,;:!?-") == word), None)
        if idx is None or r["sense"] not in HOMOGRAPHS.get(word, {}):
            continue
        by_word[word].append(Example(word, words, idx, r["sense"]))
    return by_word


def score(baseline: Baseline, by_word: dict[str, list[Example]]) -> tuple[int, int]:
    """Return (correct, total) for *baseline*, comparing by IPA so two slugs that
    share a pronunciation count as equal."""
    ok = tot = 0
    for word, exs in by_word.items():
        ipa = HOMOGRAPHS[word]
        for e in exs:
            ok += ipa[baseline.predict(e)] == ipa[e.sense]
            tot += 1
    return ok, tot

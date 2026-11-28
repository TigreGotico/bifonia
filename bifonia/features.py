"""
Language-agnostic feature extraction for sense disambiguation.

`extract_features(words, idx, vocs)` turns a tokenised sentence and the index of
an ambiguous word into a sparse ``{feature_name: weight}`` dict. Training and
inference call the same function, so they cannot skew.

No Portuguese and no per-word branch is hard-coded here. Language knowledge
enters through the ``vocs`` sets (``locale/<lang>/*.voc``), the
:data:`bifonia.cues.FEATURE_CUES` registry and the weights a model learns, so a
fork for a related language swaps those files and retrains.

Feature families (sparse, value 1.0 unless noted):
  L1=… L4= / R1= R2= R3=   positional context tokens; <BOS>/<EOS> at the edges
  W=<tok>                   bag of tokens in the ±4 window (count)
  <slot>∈<SET>              structural membership, slot ∈ {L1, L2, R1, R2, S}
  self_sfx=INF / self_sfx3= / self_pfx3=   morphology of the target word
  R1_mente / R1_deverbal / L1_inf / R1_inf neighbour morphology
  pos0 / L1_looks_verb                     position and shape
  CUE:<sense>                              proximity-weighted cue score of one
                                           of the target's senses (value = score)

Pure stdlib (no numpy), so it is safe on the zero-dependency inference path.
"""
import functools
import pathlib

from bifonia.cues import FEATURE_CUES
from bifonia.text import (
    DEVERBAL_SUFFIXES as _DEVERBAL_SFX, cue_score as _cue_score, folded as _folded,
    looks_verb as _looks_verb, strip_punct as _strip,
)
from bifonia.vocab import _lang_dir, voc as _voc

# Grammatical sets (determiners, pronouns, clitics, copula, …), generic across
# Romance languages. The per-sense lexical cue lists enter as the CUE: features.
STRUCTURAL_VOCS = (
    "determiners", "pronouns", "quantifiers", "articles", "sing_articles",
    "contracted_det", "de_contractions", "locative_contractions", "contracted_a",
    "copula", "aux_verbs", "passive_aux", "post_clitics", "clitics_all",
    "comparatives", "intensifiers", "neg_adv", "pure_negation",
    "temporal_conj", "conj_subjunctive", "numeric", "adv_bridge",
    "after_prep_extra", "after_prep_weak", "never_after_prep", "before_prep_extra",
    "exclamative_det", "function_words", "fem_det", "masc_det",
    "prep_governing", "optative_adv", "material_prep",
)

_INF_SFX = ("ar", "er", "ir")


def _read_voc(path: pathlib.Path) -> frozenset:
    """Return the lowercased terms of a .voc file (one term per line, '#' comments)."""
    terms = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.add(line.lower())
    return frozenset(terms)


@functools.lru_cache(maxsize=None)
def load_structural_vocs(lang: str = "pt-pt") -> dict:
    """Return the STRUCTURAL_VOCS sets of *lang* by name; a missing file is skipped."""
    lang_dir = _lang_dir(lang)
    vocs = {}
    for name in STRUCTURAL_VOCS:
        path = lang_dir / f"{name}.voc"
        if path.exists():
            vocs[name] = _read_voc(path)
    return vocs


def _tok(words: list, i: int) -> str:
    """Stripped lowercase token at position i, or '' if out of range."""
    return _strip(words[i]).lower() if 0 <= i < len(words) else ""


def _is_inf(w: str) -> bool:
    return w.endswith(_INF_SFX) and len(w) > 2


def extract_features(words: list, idx: int, vocs: dict) -> dict:
    """Return the sparse feature dict for the ambiguous word at *idx*.

    *vocs* maps a structural set name to its members, as
    :func:`load_structural_vocs` returns them.
    """
    feats: dict = {}
    n = len(words)
    self_tok = _tok(words, idx)

    for d in range(1, 5):
        i = idx - d
        feats[f"L{d}={_tok(words, i) if i >= 0 else '<BOS>'}"] = 1.0
    for d in range(1, 4):
        i = idx + d
        feats[f"R{d}={_tok(words, i) if i < n else '<EOS>'}"] = 1.0

    for i in range(max(0, idx - 4), min(n, idx + 5)):
        if i == idx:
            continue
        t = _tok(words, i)
        if t:
            feats[f"W={t}"] = feats.get(f"W={t}", 0.0) + 1.0

    slots = {"L1": idx - 1, "L2": idx - 2, "R1": idx + 1, "R2": idx + 2, "S": idx}
    for slot, i in slots.items():
        t = _tok(words, i)
        if not t:
            continue
        for name, members in vocs.items():
            if t in members:
                feats[f"{slot}∈{name}"] = 1.0

    if _is_inf(self_tok):
        feats["self_sfx=INF"] = 1.0
    if len(self_tok) >= 3:
        feats[f"self_sfx3={self_tok[-3:]}"] = 1.0
        feats[f"self_pfx3={self_tok[:3]}"] = 1.0
    r1 = _tok(words, idx + 1)
    l1 = _tok(words, idx - 1)
    if r1.endswith("mente"):
        feats["R1_mente"] = 1.0
    if r1.endswith(_DEVERBAL_SFX):
        feats["R1_deverbal"] = 1.0
    if _is_inf(l1):
        feats["L1_inf"] = 1.0
    if _is_inf(r1):
        feats["R1_inf"] = 1.0
    if idx == 0:
        feats["pos0"] = 1.0
    if l1 and _looks_verb(l1):
        feats["L1_looks_verb"] = 1.0

    # The rules' own cue lexicons, so the model need not rediscover them from the
    # bag of words.
    for voc_name, (cue_word, cue_sense) in FEATURE_CUES.items():
        if cue_word == self_tok:
            score = _cue_score(words, idx, _folded(_voc(voc_name)))
            if score:
                feats[f"CUE:{cue_sense}"] = float(score)

    return feats

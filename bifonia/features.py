"""
Language-agnostic feature extraction for sense disambiguation.

`extract_features(words, idx, vocs)` turns a tokenised sentence and the index of
an ambiguous word into a sparse ``{feature_name: weight}`` dict. The SAME function
is used at training and inference time, so the model can never suffer train/serve
skew.

There is NO hardcoded Portuguese here and NO per-word logic: language knowledge
enters only through (a) the ``vocs`` membership sets passed in (loaded from
``locale/<lang>/*.voc``) and (b) the weights a model learns over these features.
A fork for a related language swaps the ``.voc`` files and retrains — this file is
untouched.

Feature families (all sparse, value 1.0 unless noted):
  L1=… L4= / R1= R2= R3=   positional context tokens (skipgrams); <BOS>/<EOS> at edges
  W=<tok>                   bag of tokens in the ±WINDOW window (the "overlap" features)
  <slot>∈<SET>              structural membership (slot ∈ {L1,L2,R1,R2,S}); from vocs
  self_sfx=INF / self_sfx3= / self_pfx3=   morphology of the target word
  R1_mente / L1_inf / R1_deverbal          neighbour morphology
  pos0 / prev_looks_verb                   position / shape heuristics

Pure stdlib — must stay import-light (no numpy) so it is safe on the
zero-dependency inference path.
"""
import functools
import pathlib
import re

_PUNCT = str.maketrans("", "", ".,;:!?\"'()[]{}«»–—")

# Structural .voc sets used as membership-feature templates. These encode grammar
# (determiners, pronouns, clitics, copula, …) — generic across Romance languages —
# NOT per-word lexical semantics. Lexical cue lists (court_terms, mould_cues, …)
# are intentionally excluded: the model learns those from the W=/positional tokens.
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

_LOCALE_ROOT = pathlib.Path(__file__).parent / "locale"
_INF_SFX = ("ar", "er", "ir")
# nominalised-VP endings (para análise/revisão/avaliação …) — generic Romance suffixes
_DEVERBAL_SFX = ("ção", "são", "gem", "ura", "ência", "ância", "mento", "ismo", "ise")
_VERB_ENDINGS = ("ou", "eu", "iu", "ei", "ava", "ara", "era")
_VOWELS = frozenset("aeiouáéíóúâêîôûãõàèìòùäëïöü")


def _read_voc(path: pathlib.Path) -> frozenset:
    """Read a .voc file (one term per line, '#' comments) with stdlib only."""
    terms = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.add(line.lower())
    return frozenset(terms)


@functools.lru_cache(maxsize=None)
def load_structural_vocs(lang: str = "pt-pt") -> dict:
    """Load the STRUCTURAL_VOCS membership sets for *lang* (stdlib, cached)."""
    lang_dir = _LOCALE_ROOT / lang.lower()
    vocs = {}
    for name in STRUCTURAL_VOCS:
        path = lang_dir / f"{name}.voc"
        if path.exists():
            vocs[name] = _read_voc(path)
    return vocs


def _strip(w: str) -> str:
    return w.translate(_PUNCT)


def _tok(words: list, i: int) -> str:
    """Stripped lowercase token at position i, or '' if out of range."""
    return _strip(words[i]).lower() if 0 <= i < len(words) else ""


def _is_inf(w: str) -> bool:
    return w.endswith(_INF_SFX) and len(w) > 2


def _looks_verb(w: str) -> bool:
    if w.endswith(_VERB_ENDINGS):
        return True
    if w.endswith("ia") and len(w) >= 4 and w[-3] not in _VOWELS:
        return True
    return False


def extract_features(words: list, idx: int, vocs: dict) -> dict:
    """Return the sparse feature dict for the ambiguous word at *idx*.

    *vocs* maps a structural set name → frozenset of members (see
    :func:`load_structural_vocs`).
    """
    feats: dict = {}
    n = len(words)
    self_tok = _tok(words, idx)

    # ── positional lexical (skipgrams) ────────────────────────────────────────
    for d in range(1, 5):
        i = idx - d
        feats[f"L{d}={_tok(words, i) if i >= 0 else '<BOS>'}"] = 1.0
    for d in range(1, 4):
        i = idx + d
        feats[f"R{d}={_tok(words, i) if i < n else '<EOS>'}"] = 1.0

    # ── bag-of-window (the overlap features) ──────────────────────────────────
    for i in range(max(0, idx - 4), min(n, idx + 5)):
        if i == idx:
            continue
        t = _tok(words, i)
        if t:
            feats[f"W={t}"] = feats.get(f"W={t}", 0.0) + 1.0

    # ── structural membership (slot ∈ SET) ────────────────────────────────────
    slots = {"L1": idx - 1, "L2": idx - 2, "R1": idx + 1, "R2": idx + 2, "S": idx}
    for slot, i in slots.items():
        t = _tok(words, i)
        if not t:
            continue
        for name, members in vocs.items():
            if t in members:
                feats[f"{slot}∈{name}"] = 1.0

    # ── morphology / position ─────────────────────────────────────────────────
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

    return feats

"""
bifonia — Portuguese heterophonic homograph disambiguation.

Identifies the correct IPA pronunciation of words whose spelling is identical
but whose phonology depends on their part of speech (e.g. NOUN vs VERB).

Quick start::

    from bifonia import tokenize, is_ambiguous, disambiguate

    sentence = "O autocarro para em frente ao hospital."
    words = tokenize(sentence)
    for i, w in enumerate(words):
        if is_ambiguous(w):
            ipa = disambiguate(words, i)
            print(f"{w} → [{ipa}]")
"""

import re
from functools import lru_cache as _lru_cache

from bifonia.data import (
    AMBIGUOUS_WORDS, HOMOGRAPHS, DEFAULT_POS, POS_SENSES, SENSE_POS,
)
from bifonia.scoring import (
    guess_pos as _scoring_guess_pos,
    resolve_sense as _resolve_sense,
)
from bifonia.version import VERSION_STR

__version__ = VERSION_STR
__all__ = [
    "tokenize",
    "is_ambiguous",
    "guess_pos",
    "guess_sense",
    "disambiguate",
    "add_extra_diacritics",
    "AMBIGUOUS_WORDS",
    "HOMOGRAPHS",
]


def tokenize(text: str) -> list:
    """Lowercase and split *text* into word tokens.

    Two pieces of orthography are kept so the context scorer can use them, both
    chosen so the token count and order are identical to a bare ``\\w+`` split
    (keeping the position-indexed replacement walk in
    :func:`add_extra_diacritics` aligned):

    - Trailing sentence punctuation (``. , ; : ! ?``) stays attached to the token
      it follows (``"querer,"``) → clause-boundary guards
      (``_prev_raw[-1] in punct``).
    - A hyphen that attaches an *enclitic* clitic is kept as a leading marker on
      the clitic token (``"torno-me"`` → ``["torno", "-me"]``). The marker
      distinguishes an enclitic from a homographic article/preposition
      (``"torno-o"`` clitic vs ``"torno o carro"``), an unambiguous finite-verb
      signal — see the enclitic rule in :mod:`bifonia.scoring`.

    Lookup sites strip both with :func:`_strip_edge`, and the target slot is
    normalised before scoring.
    """
    return re.findall(r"-?\w+[.,;:!?]*", text.lower(), re.UNICODE)


_EDGE_PUNCT = ".,;:!?"


def _strip_edge(token: str) -> str:
    """Strip trailing punctuation / a leading clitic-hyphen a token may carry."""
    return token.strip(_EDGE_PUNCT + "-")


def is_ambiguous(word: str) -> bool:
    """Return True if *word* is a known heterophonic homograph."""
    return _strip_edge(word).lower() in AMBIGUOUS_WORDS


def disambiguate(words: list, idx: int, pos: str = None, sense: str = None,
                 postag: str = None) -> str:
    """Return the IPA transcription for the ambiguous word at position *idx*.

    The reading is selected by MEANING (`sense`), not POS: most words have one
    sense per POS, so a POS guess resolves directly, but words like ``sede``
    (thirst vs seat — both nouns) are disambiguated by context meaning cues.

    Parameters
    ----------
    words:
        Tokenized (lowercased) sentence as returned by :func:`tokenize`.
    idx:
        Index of the ambiguous word within *words*.
    pos:
        Override POS guessing with an explicit UDEP tag (``"ADP"``, ``"NOUN"``,
        ``"VERB"``, ``"ADJ"``); the sense is then resolved within that POS.
    sense:
        Override entirely with an explicit meaning slug (see the CSV / HOMOGRAPHS).

    Raises
    ------
    ValueError
        If the word is not a known homograph.
    KeyError
        If *sense* is given but has no IPA entry for this word.
    """
    raw = words[idx]
    # Drop any trailing sentence punctuation, then accept pre-AO1990 /
    # diacritized tokens by normalising to the base form.
    token = _strip_edge(raw)
    word = _DIACRITIZED_TO_BASE.get(token, token)
    if word not in AMBIGUOUS_WORDS:
        raise ValueError(f"{raw!r} is not a known heterophonic homograph")

    if sense is None:
        sense = guess_sense(words, idx, pos=pos, postag=postag)
    return HOMOGRAPHS[word][sense]


# ── diacritics lookup ─────────────────────────────────────────────────────────
# Non-canonical, non-AO1990 forms that force the correct vowel quality in a
# downstream rule-based G2P (convention: acute = open, circumflex = closed).
# Words with no meaningful diacritic change (ADP para/pelo) are omitted so
# add_extra_diacritics returns them unchanged.
_DIACRITIZED: dict = {
    ("acordo",     "agreement"):  "acôrdo",
    ("acordo",     "wake"):       "acórdo",
    ("acerto",     "settlement"): "acêrto",
    ("acerto",     "adjust"):     "acérto",
    ("cerro",      "hill"):       "cêrro",
    ("cerro",      "shut"):       "cérro",
    ("choro",      "weeping"):    "chôro",
    ("choro",      "weep"):       "chóro",
    ("colher",     "spoon"):      "colhér",   # noun = open ɛ
    ("colher",     "harvest"):    "colhêr",   # verb = closed e
    ("começo",     "beginning"):  "comêço",
    ("começo",     "begin"):      "coméço",
    ("conserto",   "repair"):     "consêrto",
    ("conserto",   "mend"):       "consérto",
    ("coro",       "choir"):      "côro",
    ("coro",       "blush"):      "córo",
    ("corte",      "court"):      "côrte",
    ("corte",      "cut"):        "córte",
    ("forma",      "mould"):      "fôrma",
    ("forma",      "shape"):      "fórma",
    ("gosto",      "taste"):      "gôsto",
    ("gosto",      "like"):       "gósto",
    ("gozo",       "enjoyment"):  "gôzo",
    ("gozo",       "enjoy"):      "gózo",
    ("jogo",       "game"):       "jôgo",
    ("jogo",       "play"):       "jógo",
    ("molho",      "sauce"):      "môlho",
    ("molho",      "bundle"):     "mólho",
    ("olho",       "eye"):        "ôlho",
    ("olho",       "look"):       "ólho",
    ("para",       "stop"):       "pára",
    ("pelo",       "hair"):       "pêlo",
    ("pelo",       "peel"):       "pélo",
    ("peso",       "weight"):     "pêso",
    ("peso",       "weigh"):      "péso",
    ("porto",      "harbour"):    "pôrto",
    ("porto",      "carry"):      "pórto",
    ("posto",      "station"):    "pôsto",
    ("posto",      "post"):       "pósto",
    ("rego",       "furrow"):     "rêgo",
    ("rego",       "water"):      "régo",
    ("seco",       "dry"):        "sêco",
    ("seco",       "dry_vb"):     "séco",
    ("sede",       "seat"):       "séde",     # seat (HQ) = open ɛ
    ("sede",       "thirst"):     "sêde",     # thirst    = closed e
    ("sobre",      "sail"):       "sôbre",
    ("sobre",      "leftover"):   "sóbre",
    ("tola",       "head"):       "tóla",     # head/wood = open ɔ
    ("tola",       "foolish"):    "tôla",     # adj       = closed o
    ("torre",      "tower"):      "tôrre",
    ("torre",      "roast"):      "tórre",
    ("transtorno", "disorder"):   "transtôrno",
    ("transtorno", "upset"):      "transtórno",
    ("torno",      "lathe"):      "tôrno",
    ("torno",      "turn"):       "tórno",
    ("troco",      "change"):     "trôco",
    ("troco",      "exchange"):   "tróco",
    ("toco",       "stump"):      "tôco",
    ("toco",       "play"):       "tóco",
    ("rogo",       "plea"):       "rôgo",
    ("rogo",       "beg"):        "rógo",
    ("choco",      "addled"):     "chôco",    # adj/noun = closed o
    ("choco",      "cuttlefish"): "chôco",
    ("choco",      "hatch"):      "chóco",    # verb     = open ɔ
    ("contorno",   "contour"):    "contôrno",
    ("contorno",   "circumvent"): "contórno",
    ("entorno",    "surroundings"): "entôrno",
    ("entorno",    "spill"):      "entórno",
    ("almoço",     "lunch"):      "almôço",
    ("almoço",     "dine"):       "almóço",
    ("rolo",       "roll"):       "rôlo",
    ("rolo",       "tumble"):     "rólo",
    ("soco",       "punch"):      "sôco",
    ("soco",       "strike"):     "sóco",
    ("esforço",    "effort"):     "esfôrço",
    ("esforço",    "strive"):     "esfórço",
    ("conforto",   "comfort"):    "confôrto",
    ("conforto",   "soothe"):     "confórto",
    ("aborto",     "abortion"):   "abôrto",
    ("aborto",     "abort"):      "abórto",
    ("adorno",     "adornment"):  "adôrno",
    ("adorno",     "adorn"):      "adórno",
    ("reforço",    "reinforcement"): "refôrço",
    ("reforço",    "reinforce"):  "refórço",
    ("soldo",      "pay"):        "sôldo",
    ("soldo",      "weld"):       "sóldo",
    ("esboço",     "sketch"):     "esbôço",
    ("esboço",     "outline"):    "esbóço",
    ("governo",    "government"): "govêrno",
    ("governo",    "govern"):     "govérno",
    ("emprego",    "job"):        "emprêgo",
    ("emprego",    "employ"):     "emprégo",
    ("selo",       "stamp"):      "sêlo",
    ("selo",       "seal"):       "sélo",
    ("gelo",       "ice"):        "gêlo",
    ("gelo",       "freeze"):     "gélo",
    ("zelo",       "zeal"):       "zêlo",
    ("zelo",       "tend"):       "zélo",
    ("cerco",      "siege"):      "cêrco",
    ("cerco",      "surround"):   "cérco",
    ("erro",       "error"):      "êrro",
    ("erro",       "err"):        "érro",
    ("sopro", "breath"): "sôpro",
    ("sopro", "blow"): "sópro",
    ("forro", "lining"): "fôrro",
    ("forro", "line"): "fórro",
    ("dobro", "double"): "dôbro",
    ("dobro", "fold"): "dóbro",
    ("abono", "allowance"): "abôno",
    ("abono", "vouch"): "abóno",
    ("logro", "deceit"): "lôgro",
    ("logro", "deceive"): "lógro",
    ("topo", "summit"): "tôpo",
    ("topo", "bump_into"): "tópo",
    ("jorro", "jet"): "jôrro",
    ("jorro", "gush"): "jórro",
    ("golfo", "gulf"): "gôlfo",
    ("golfo", "spew"): "gólfo",
    ("colmo", "culm"): "côlmo",
    ("colmo", "thatch"): "cólmo",
    ("fosso", "ditch"): "fôsso",
    ("fosso", "root_up"): "fósso",
    ("toldo", "awning"): "tôldo",
    ("toldo", "cloud_over"): "tóldo",
    ("arrojo", "boldness"): "arrôjo",
    ("arrojo", "hurl"): "arrójo",
    ("despojo", "spoils"): "despôjo",
    ("despojo", "strip"): "despójo",
    ("cobro", "cessation"): "côbro",
    ("cobro", "collect"): "cóbro",
    ("domo", "dome"): "dômo",
    ("domo", "tame"): "dómo",
    ("sobro", "cork_oak"): "sôbro",
    ("sobro", "be_left_over"): "sóbro",
    ("retorno", "return"): "retôrno",
    ("retorno", "go_back"): "retórno",
    ("desgosto", "sorrow"): "desgôsto",
    ("desgosto", "dislike"): "desgósto",
    ("desconforto", "discomfort"): "desconfôrto",
    ("desconforto", "discomfit"): "desconfórto",
    ("desdobro", "unfolding"): "desdôbro",
    ("desdobro", "unfold"): "desdóbro",
    ("redobro", "redoubling"): "redôbro",
    ("redobro", "redouble"): "redóbro",
    ("reboco", "plaster"): "rebôco",
    ("reboco", "tow"): "rebóco",
    ("decoro", "decorum"): "decôro",
    ("decoro", "memorize"): "decóro",
    ("estofo", "stuffing"): "estôfo",
    ("estofo", "upholster"): "estófo",
    ("destroço", "wreckage"): "destrôço",
    ("destroço", "wreck"): "destróço",
    ("desafogo", "relief"): "desafôgo",
    ("desafogo", "relieve"): "desafógo",
    ("arroto", "belch"): "arrôto",
    ("arroto", "burp"): "arróto",
    ("consolo", "consolation"): "consôlo",
    ("consolo", "console"): "consólo",
    ("desconsolo", "disconsolation"): "desconsôlo",
    ("desconsolo", "dishearten"): "desconsólo",
    ("desacordo", "disagreement"): "desacôrdo",
    ("desacordo", "disagree"): "desacórdo",
    ("engodo", "bait"): "engôdo",
    ("engodo", "lure"): "engódo",
    ("escorço", "foreshortening"): "escôrço",
    ("escorço", "foreshorten"): "escórço",
    ("desaforo", "insolence"): "desafôro",
    ("desaforo", "affront"): "desafóro",
    ("abrolho", "caltrop"): "abrôlho",
    ("abrolho", "sprout"): "abrólho",
    ("rola", "turtledove"): "rôla",
    ("rola", "rolls"): "róla",
    ("solto", "loose"): "sôlto",
    ("solto", "release"): "sólto",
    ("encosto", "backrest"): "encôsto",
    ("encosto", "lean"): "encósto",
    ("apelo", "appeal"): "apêlo",
    ("apelo", "call_out"): "apélo",
    ("desprezo", "contempt"): "desprêzo",
    ("desprezo", "despise"): "desprézo",
    ("enredo", "plot"): "enrêdo",
    ("enredo", "entangle"): "enrédo",
    ("desenredo", "denouement"): "desenrêdo",
    ("desenredo", "disentangle"): "desenrédo",
    ("espeto", "skewer"): "espêto",
    ("espeto", "stab"): "espéto",
    ("aperto", "squeeze"): "apêrto",
    ("aperto", "tighten"): "apérto",
    ("desvelo", "devotion"): "desvêlo",
    ("desvelo", "unveil"): "desvélo",
    ("repelo", "hair_pull"): "repêlo",
    ("repelo", "pluck"): "repélo",
    ("arrepelo", "hair_pulling"): "arrepêlo",
    ("arrepelo", "snatch"): "arrepélo",
    ("congelo", "freezing"): "congêlo",
    ("congelo", "freeze"): "congélo",
    ("arremesso", "throw"): "arremêsso",
    ("arremesso", "fling"): "arremésso",
    ("arremedo", "imitation"): "arremêdo",
    ("arremedo", "mimic"): "arremédo",
    ("despego", "detachment"): "despêgo",
    ("despego", "detach"): "despégo",
    ("desapego", "indifference"): "desapêgo",
    ("desapego", "let_go"): "desapégo",
    ("apego", "attachment"): "apêgo",
    ("apego", "cling"): "apégo",
    ("desemprego", "noun"): "desemprêgo",
    ("desemprego", "verb"): "desemprégo",
    ("desmantelo", "dismantling"): "desmantêlo",
    ("desmantelo", "dismantle"): "desmantélo",
    ("atropelo", "trampling"): "atropêlo",
    ("atropelo", "run_over"): "atropélo",
    ("degelo", "thaw"): "degêlo",
    ("degelo", "thaw_out"): "degélo",
    ("desgelo", "defrosting"): "desgêlo",
    ("desgelo", "defrost"): "desgélo",
    ("desespero", "despair"): "desespêro",
    ("desespero", "despair_at"): "desespéro",
    ("escabelo", "stool"): "escabêlo",
    ("escabelo", "dishevel"): "escabélo",
    ("novelo", "yarn_ball"): "novêlo",
    ("novelo", "narrate"): "novélo",
    ("relevo", "relief_terrain"): "relêvo",
    ("relevo", "emphasize"): "relévo",
    ("aceno", "nod"): "acêno",
    ("aceno", "beckon"): "acéno",
    ("sopeso", "heft"): "sopêso",
    ("sopeso", "weigh_up"): "sopéso",
    ("empeno", "warping"): "empêno",
    ("empeno", "warp"): "empéno",
    ("desempeno", "straightening"): "desempêno",
    ("desempeno", "straighten"): "desempéno",
    ("sossego", "calm"): "sossêgo",
    ("sossego", "soothe"): "sosségo",
    ("desassossego", "disquiet"): "desassossêgo",
    ("desassossego", "disturb"): "desassosségo",
    ("esmero", "meticulousness"): "esmêro",
    ("esmero", "perfect"): "esméro",
    ("azedo", "sour"): "azêdo",
    ("azedo", "turn_sour"): "azédo",
    ("desempeço", "riddance"): "desempêço",
    ("desempeço", "free_up"): "desempéço",
    ("desemperro", "unjamming"): "desempêrro",
    ("desemperro", "unjam"): "desempérro",
    ("emperro", "jam"): "empêrro",
    ("emperro", "stick"): "empérro",
    ("tempero",    "seasoning"):  "tempêro",
    ("tempero",    "season"):     "tempéro",
    # diacritic-collapse + deverbal additions (acute = open ɔ/ɛ, circumflex = closed o/e)
    ("bola",       "ball"):       "bóla",     # ball       = open ɔ
    ("bola",       "loaf"):       "bôla",     # bôla bread = closed o
    ("cor",        "by_heart"):   "cór",      # "de cor"   = open ɔ
    ("cor",        "colour"):     "côr",      # colour     = closed o
    ("lobo",       "lobe"):       "lóbo",     # lobe       = open ɔ
    ("lobo",       "wolf"):       "lôbo",     # wolf       = closed o
    ("polo",       "pole"):       "pólo",     # pole/sport = open ɔ
    ("polo",       "fledgling"):  "pôlo",     # young bird = closed o
    ("renovo",     "renew"):      "renóvo",   # I renew    = open ɔ
    ("renovo",     "shoot"):      "renôvo",   # shoot/noun = closed o
    ("soma",       "add"):        "sóma",     # sums/verb  = open ɔ
    ("soma",       "sum"):        "sôma",     # sum/noun   = closed o
    ("força",      "force"):      "fórça",    # force/verb = open ɔ
    ("força",      "strength"):   "fôrça",    # strength   = closed o
}


# Reverse maps from diacritized form:
#   → base word   (for test normalisation and pre-AO1990 input stripping)
#   → sense slug  (for unambiguous pre-AO1990 / already-diacritized input)
_DIACRITIZED_TO_BASE:  dict = {v: k[0] for k, v in _DIACRITIZED.items()}
_DIACRITIZED_TO_SENSE: dict = {v: k[1] for k, v in _DIACRITIZED.items()}


@_lru_cache(maxsize=1)
def _sense_model():
    """Lazy-load the learned sense model; None if absent (→ rule fallback)."""
    try:
        from bifonia.model import SenseModel
        return SenseModel.load()
    except Exception:
        return None


def _rule_sense(base: str, words: list, idx: int, pos: str = None,
                proper: bool = False) -> str:
    resolved_pos = pos or _scoring_guess_pos(words, idx, proper=proper)
    return _resolve_sense(base, words, idx, resolved_pos)


_UPOS_NORM = {"PROPN": "NOUN", "AUX": "VERB"}


def guess_sense(words: list, idx: int, pos: str = None, proper: bool = False,
                postag: str = None) -> str:
    """Return the most likely MEANING slug for the ambiguous word at *idx*.

    If the token is already a diacritized form (pre-AO1990 orthography or output
    of :func:`add_extra_diacritics`), the sense is read straight off the diacritic
    (e.g. *séde* → seat, *sêde* → thirst, *pára* → stop).

    Otherwise, where a trained statistical model covers the word and is flagged to
    win over the rules (``route == "model"`` with sufficient margin), the model
    predicts the sense; every other case falls back to the corpus-free rule engine.
    An explicit *pos* override always uses the rule resolver.

    *postag* is the hybrid-ensemble hint: an external POS tag (e.g. from spaCy).
    It resolves the reading only when that POS maps to exactly ONE sense
    (a POS-separable word); for same-POS readings (sede/molho/corte/forma — where
    a tagger misses by construction) it is ignored and the rules decide. This lets
    a caller fuse a neural tagger with the rules without adding a dependency.
    """
    raw = words[idx]
    token = _strip_edge(raw)
    if token in _DIACRITIZED_TO_SENSE:
        return _DIACRITIZED_TO_SENSE[token]
    base = _DIACRITIZED_TO_BASE.get(token, token)
    normalised = words
    if base != raw:                         # normalise a punctuated/diacritized slot
        normalised = list(words)
        normalised[idx] = base
    if postag is not None:                  # hybrid-ensemble hint (see docstring)
        cands = POS_SENSES.get(base, {}).get(_UPOS_NORM.get(postag, postag))
        if cands and len(cands) == 1:
            return cands[0]
        # ambiguous / unmapped tag → fall through to model + rules below
    if pos is not None:
        return _rule_sense(base, normalised, idx, pos)
    # A mid-sentence proper noun is a name, not a finite verb — resolve by rules
    # (with the proper-noun bias) and skip the statistical model.
    if proper:
        return _rule_sense(base, normalised, idx, proper=True)
    model = _sense_model()
    if (model is not None and model.has(base) and model.route(base) == "model"
            and model.margin(base, normalised, idx) >= model.margin_tau(base)):
        return model.predict(base, normalised, idx)
    return _rule_sense(base, normalised, idx)


def guess_pos(words: list, idx: int, proper: bool = False) -> str:
    """Return the most likely UDEP POS tag for the ambiguous word at *idx*.

    Thin wrapper over :func:`guess_sense` that maps the resolved meaning back to
    its descriptive POS, preserving the pre-existing POS-tagging interface.
    Diacritized input (e.g. *pára*, *acôrdo*) is resolved without context scoring.
    *proper* biases the NOUN reading for a mid-sentence capitalised token.
    """
    raw = words[idx]
    token = _strip_edge(raw)
    base = _DIACRITIZED_TO_BASE.get(token, token)
    if token in _DIACRITIZED_TO_SENSE:
        return SENSE_POS[base][_DIACRITIZED_TO_SENSE[token]]
    if base != raw:
        normalised = list(words)
        normalised[idx] = base
        return _scoring_guess_pos(normalised, idx, proper=proper)
    return _scoring_guess_pos(words, idx, proper=proper)


def proper_flags(text: str) -> list:
    """Per-token flags (aligned with :func:`tokenize`) marking a likely proper
    noun: a mid-sentence capitalised token (not sentence-initial). Used to bias
    the NOUN reading (names like "Cerro Corá" are not finite verbs)."""
    toks = re.findall(r"-?\w+[.,;:!?]*", text, re.UNICODE)
    flags = [False] * len(toks)
    for i in range(1, len(toks)):
        core = toks[i].lstrip("-")
        prev = toks[i - 1]
        if core[:1].isupper() and not (prev and prev[-1] in ".!?:…"):
            flags[i] = True
    # Sentence-initial proper-noun phrase / title: the first word is always
    # capitalised, so only flag it when the *next* token is itself a proper-noun
    # continuation ("Cerro Corá") or a number / nº marking a title ("Choro nº 13").
    if len(toks) > 1:
        nxt = toks[1].lstrip("-").strip(_EDGE_PUNCT)
        if nxt[:1].isupper() or nxt[:1].isdigit() or nxt.lower() in {"nº", "n.º", "no", "nr"}:
            flags[0] = True
    return flags


def add_extra_diacritics(sentence: str) -> str:
    """Return *sentence* with non-canonical diacritics on heterophonic homographs.

    The inserted diacritics are not AO1990-compliant but force a downstream
    rule-based G2P to emit the correct vowel quality:

    - acute  (ó/é) → open  /ɔ/ or /ɛ/  (typically verb reading)
    - circumflex (ô/ê) → closed /o/ or /e/  (typically noun reading)

    Words that are not ambiguous, or whose POS reading needs no extra diacritic
    (e.g. ADP *para*, ADP *pelo*), are returned unchanged. Original capitalisation
    is preserved, so a sentence-initial homograph keeps its capital (*Acórdo …*).
    """
    words = tokenize(sentence)
    caps = proper_flags(sentence)
    replacements = {}
    for i, word in enumerate(words):
        base = _strip_edge(word)
        if is_ambiguous(base):
            diacritized = _DIACRITIZED.get((base, guess_sense(words, i, proper=caps[i])))
            if diacritized:
                replacements[i] = diacritized
    if not replacements:
        return sentence

    # Walk the same \w+ tokens over the original string so each replacement lands
    # on the right occurrence with its case preserved (a plain str.replace would
    # miss a capitalised token and could hit a substring of another word).
    out, pos, idx = [], 0, 0
    for m in re.finditer(r"\w+", sentence, re.UNICODE):
        out.append(sentence[pos:m.start()])
        tok = m.group(0)
        diac = replacements.get(idx)
        if diac is not None and tok[:1].isupper():
            diac = diac[:1].upper() + diac[1:]
        out.append(diac if diac is not None else tok)
        pos, idx = m.end(), idx + 1
    out.append(sentence[pos:])
    return "".join(out)

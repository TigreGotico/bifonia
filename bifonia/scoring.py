"""
Context-based POS scoring for Portuguese heterophonic homographs.

Each scorer returns an integer; higher = more confident.
Negative scores signal anti-evidence for that POS.

Sources for "fora": the Formulário Ortográfico 1943, rule 14.ª, "Emprega-se o
acento circunflexo como diferencial ou distintivo no e e no o fechados da
sílaba tônica das palavras que estão em homografia com outras em que são
abertos ... fôra (v.) e fora (adv. interj. ou s.m.)"; the Acordo Ortográfico
1945, "fora (ô), flexão de ser e ir, e fora (ó), advérbio, interjeição e
substantivo"; Priberam (https://dicionario.priberam.org/fora), "fora |ó| ...
advérbio"; Infopédia (https://www.infopedia.pt/dicionarios/lingua-portuguesa/fora),
"fora fo.ra ˈfɔrɐ advérbio".
"""

from bifonia.cues import SENSE_CUES
from bifonia.text import (
    DEVERBAL_SUFFIXES as _DEVERBAL_SFX,
    cue_score as _cue_score,
    folded as _folded,
    looks_verb as _looks_verb,
    strip_punct as _strip,
)
from bifonia.data import (
    ADP_IPA, ADJ_IPA, NOUNS_IPA, VERBS_IPA,
    DEFAULT_POS, DEFAULT_SENSE, BASE_SCORE,
    POS_SENSES,
    DET, PRON, AUX_VERBS, NUMERIC,
    BEFORE_PREP, AFTER_PREP, NEVER_AFTER_PREP,
    SOBRE_GOV, QUANT, STOPPABLE_THINGS,
)
from bifonia.vocab import voc

# Wordlists live in locale/<lang>/*.voc (see vocab.py).
_ENCLITICS       = voc("enclitics")         # hyphen-attached enclitic clitics (verb host)
_AFTER_PREP_WEAK = voc("after_prep_weak")   # time adv/pron: weak ADP credit
_COPULA          = voc("copula")            # copular/semi-copular verbs
_INTENSIFIERS    = voc("intensifiers")      # degree intensifiers (modify ADJ)
_TEMPORAL_CONJ   = voc("temporal_conj")     # quando/enquanto
_CONJ_SUBJ       = voc("conj_subjunctive")  # subjunctive-introducing conjunctions
_NEG_ADV         = voc("neg_adv")           # negation/frequency adverbs before finite verb
_PASSIVE_AUX     = voc("passive_aux")       # past/subjunctive of ser (passive)

# per-word semantic lists
_PELO_FIXED   = voc("pelo_fixed")
_PELO_ROUTE   = voc("pelo_route")
_SOBRE_INTENS = voc("sobre_intensifiers")
_FEM_DET      = voc("fem_det")
_MASC_DET     = voc("masc_det")
_COURT_TERMS  = voc("court_terms")
_HEAD_VERBS   = voc("head_verbs")
_WOOD_CONTEXT = voc("wood_context")
_TER          = voc("possession_verbs")
_TRANS_VERB   = voc("trans_verbs")
_CUT_NOUNS    = voc("cut_context")
_COLHER_CTRL  = voc("colher_control_verbs")
_PARA_VERB_NEXT = voc("para_verb_next")
_ADV_BRIDGE   = voc("adv_bridge")
_OPTATIVE_ADV = voc("optative_adv")     # talvez/oxalá/tomara → present subjunctive
_BAKEWARE     = voc("bakeware")
_MOULD_CUES   = voc("mould_cues")
_HEAD_STATE   = voc("head_state")
_FORMA_SHAPE_PREV = voc("forma_shape_prev")
_FORMA_SHAPE_NEXT = voc("forma_shape_next")
_ARTICLES     = voc("articles")
_SING_ARTICLES = voc("sing_articles")
_DE_CONTRACTIONS = voc("de_contractions")
_CONTRACTED_A = voc("contracted_a")
_COMPARATIVES = voc("comparatives")
_PURE_NEG     = voc("pure_negation")
_DEFERRAL_ADV = voc("deferral_adv")
_SOBRE_ADP_QUANT = voc("sobre_adp_quant")
_TOLA_POSS_PREV = voc("tola_poss_prev")
_PERSON_POSS  = voc("person_possessive")
_MATERIAL_PREP = voc("material_prep")
_HEAD_LOCATIVE = voc("head_locative")
_TOLA_DET_PREV = voc("tola_det_prev")
_MOLHO_NOUN_PREV = voc("molho_noun_prev")
_ADVERBIAL_DE = voc("adverbial_de")
_OBJECT_PRONOUNS = voc("object_pronouns")
_OBJECT_DET = voc("object_determiners")
_A_LOCUTIONS = voc("a_locutions")
_POST_VERB_ADVERBS = voc("post_verb_adverbs")
_FREQUENCY_NOUNS = voc("frequency_nouns")
_TIME_NOUNS = voc("time_nouns")
_DEGREE_NOUNS = voc("degree_nouns")
_EVALUATIVE_NOUNS = voc("evaluative_nouns")
_EVALUATIVE_ADJ = voc("evaluative_adjectives")
_DURATION_MODIFIERS = voc("duration_modifiers")
_POR_CONTRACTIONS = voc("por_contractions")
_BAKE_PURPOSE = voc("bake_purpose")
_FIRST_PERSON_NOUNS = voc("first_person_nouns")
_PARA_PREV_SUBJ = voc("para_prev_subj")
_PREP_GOVERNING = voc("prep_governing")
_CLITICS_ALL  = voc("clitics_all")
_EXCL_DET     = voc("exclamative_det")
_LOCATIVE_CONTRACTIONS = voc("locative_contractions")
_FUNC_EXTRA = voc("function_words")
_PRENOMINAL_ADJ = voc("prenominal_adj")  # adjectives that precede a noun head
_COMPLEMENT_PREPS = voc("complement_preps")  # prep + homograph = nominal complement
_COORDINATION = voc("coordination")          # e/ou/nem/como before → noun
_AROUND_PREP = voc("around_prep")            # "em torno" locution
# meaning cues of the two noun senses of "sede"
_SEDE_SEAT   = voc("sede_seat_cues")
_SEDE_THIRST = voc("sede_thirst_cues")
# "molho" bundle/soak cues (open ɔ) vs the default sauce reading (closed o)
_BUNDLE_THINGS = voc("bundle_things")       # unambiguous bundles: "molho de chaves"
_BUNDLE_AMBIG  = voc("bundle_ambiguous")    # greens, bundle only with a gathering verb
_BUNDLE_VERBS  = voc("bundle_verbs")        # pick/buy/tie/hold → resolves the greens
_SOAK_VERBS    = voc("soak_verbs")          # "deixar/pôr/estar de molho" → soaking
# "fora": cues of the pluperfect of ser/ir (closed o) against the adverb (open ɔ)
_FORA_VERB_PREV = voc("fora_verb_prev")
_FORA_ADVERB_PREV = voc("fora_adverb_prev")
_FORA_ADVERB_NEXT = voc("fora_adverb_next")
_FORA_BRIDGE = voc("fora_bridge_adverbs")
_IRREGULAR_PARTICIPLES = voc("irregular_participles")
_PARTICIPLE_LOOKALIKES = voc("participle_lookalikes")
# contracted prep+article forms ("no", "à", "do", …)
_CONTRACTED_DET = voc("contracted_det")

_PARTICIPLE_SFX = ("ado", "ido", "ada", "ida", "ados", "idos", "adas", "idas")
_PARTICIPLE_VOWELS = frozenset("aeiouáéíóúâêîôûãõ")
# Quantifiers that follow their noun ("o peso todo").
_POSTPOSED_QUANT = {"todo", "toda", "todos", "todas", "mesmo", "mesma", "inteiro", "inteira"}
# A contracted preposition opens a prepositional phrase, not a subject.
_SUBJ_DET = DET - _LOCATIVE_CONTRACTIONS
_GOZO_EXCL_PREV = DET | QUANT | _PREP_GOVERNING
_FUNC_WORDS = DET | QUANT | PRON | AUX_VERBS | _FUNC_EXTRA
_SEDE_ABSTRACT = ("ção", "são", "mento", "dade", "ência", "ância", "ismo", "tude")


def _prev(words: list, idx: int) -> str:
    return _strip(words[idx - 1]) if idx > 0 else ""


def _next(words: list, idx: int) -> str:
    return _strip(words[idx + 1]) if idx + 1 < len(words) else ""


def _prev2(words: list, idx: int) -> str:
    return _strip(words[idx - 2]) if idx > 1 else ""


def _next2(words: list, idx: int) -> str:
    return _strip(words[idx + 2]) if idx + 2 < len(words) else ""


def _left(words: list, idx: int, n: int) -> list:
    """Return the stripped tokens up to *n* places before *idx*, nearest first."""
    return [_strip(words[idx - k]) for k in range(1, n + 1) if idx - k >= 0]


def _right(words: list, idx: int, n: int) -> list:
    """Return the stripped tokens up to *n* places after *idx*, nearest first."""
    return [_strip(words[idx + k]) for k in range(1, n + 1) if idx + k < len(words)]


def _is_infinitive(word: str) -> bool:
    w = word.rstrip(".,;:!?")
    return w.endswith(("ar", "er", "ir"))


def _is_numeric(word: str) -> bool:
    """A digit string, optionally negative ("5", "-5"), or a cardinal number word.

    Tokens are split on punctuation, so "3,5" and "1.000" reach here as their
    digit groups.
    """
    return word.removeprefix("-").isdecimal() or word in NUMERIC


def _has_participle_ending(word: str) -> bool:
    """Return True for an -ado/-ido form whose suffix follows a consonant, so a
    name such as "samoeida" does not count."""
    return any(word.endswith(sfx) and len(word) > len(sfx) and word[-len(sfx) - 1] not in _PARTICIPLE_VOWELS
               for sfx in _PARTICIPLE_SFX)


def score_adp(words: list, idx: int) -> int:
    word = words[idx]
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)

    if next_word in NEVER_AFTER_PREP:
        return 0

    score = 0
    if prev_word in BEFORE_PREP:
        score += 5
    if next_word in AFTER_PREP:
        score += 2 if next_word in _AFTER_PREP_WEAK else 5
    if _is_numeric(next_word):
        score += 5
    if _is_infinitive(next_word):
        score += 2

    if word == "para":
        # "para não/nunca X" is purpose + a negated infinitive. Frequency adverbs
        # stay out: "pára sempre" is the verb.
        if next_word in _PURE_NEG:
            score += 4
        # Purpose + a nominalised verb phrase: "para análise", "para revisão".
        if next_word.endswith(_DEVERBAL_SFX):
            score += 4
        # "para depois/amanhã": the weak +2 alone loses to the verb's DET-2-back +3.
        # Kept narrow, because "sempre/logo/aqui/lá" also follow the verb "pára".
        if next_word in _DEFERRAL_ADV:
            score += 3
        # "está pronto para": a copula and a predicative adjective before it.
        if _prev2(words, idx) in _COPULA:
            score += 4

    if word == "pelo":
        # Fixed phrases ("pelo menos", "pelo seu bem", "pelo Natal").
        if next_word in _PELO_FIXED:
            score += 6
        # Route and place nouns that AFTER_PREP lacks.
        if next_word in _PELO_ROUTE:
            score += 5
        # Passive agent: a participle in the 3 tokens before ("foi aprovado pelo X").
        if any(_has_participle_ending(w) for w in _left(words, idx, 3)):
            score += 5
        # por+o already holds the article, so an article after it rules out the
        # contraction.
        if next_word in _ARTICLES:
            score -= 8
        elif idx > 0:
            # por+o + a noun is the dominant reading ("pelo caminho"); a fur cue in
            # score_noun can still outscore it. A sentence-initial "Pelo" is the
            # 1sg verb.
            score += 3

    if word == "sobre":
        # A governing noun or verb before it ("falou sobre X").
        if prev_word in SOBRE_GOV:
            score += 6
        # A quantifier before it: "sabe muito sobre", "aprendeu pouco sobre".
        if prev_word in _SOBRE_ADP_QUANT:
            score += 5
        # "sobre o/este/ela": reinforces AFTER_PREP to beat the verb's DET-2-back cue.
        if next_word in DET | PRON:
            score += 3
        # Mid-sentence "sobre" is almost always the preposition, except after a
        # pronoun, a negation or frequency adverb, or a temporal conjunction.
        if idx > 0 and prev_word not in PRON | _NEG_ADV | _TEMPORAL_CONJ:
            score += 2

    return score


def _molho_takes_object(words: list, idx: int) -> bool:
    """Return True when a clause-initial "molho" is followed by its object.

    The verb molhar is rare next to the sauce noun, so a clause-initial "molho"
    is the noun unless a direct object marks the verb: an enclitic ("Molho-as"),
    an object pronoun ("Molho tudo") or an object determiner ("Molho as
    plantas"). Adverbs and fixed locutions may stand between them ("Molho
    sempre as plantas", "Molho de manhã as plantas", "Molho duas vezes os vasos").
    A degree or duration locution ("um pouco", "todo o dia") is skipped too. An
    evaluative noun after "um"/"uma" ("Molho uma delícia!") and a hyphenated
    compound ("Molho todo-o-terreno") are not objects.
    """
    adverbs = (_ADV_BRIDGE | _NEG_ADV | _POST_VERB_ADVERBS | _DEFERRAL_ADV
               | _INTENSIFIERS | {"mais"})
    if _next(words, idx).startswith("-"):
        return True
    j = idx + 1
    while j < len(words):
        tok = _strip(words[j])
        after = _strip(words[j + 1]) if j + 1 < len(words) else ""
        after2 = _strip(words[j + 2]) if j + 2 < len(words) else ""
        if tok in ("um", "uma") and (after in _EVALUATIVE_NOUNS or (
                after in _EVALUATIVE_ADJ and after2 in _EVALUATIVE_NOUNS)):
            return False                      # "uma delícia", "uma autêntica delícia"
        if tok in ("um", "uma") and after in _DEGREE_NOUNS and after2 == "de":
            return True                       # "um pouco de pão" is the object
        if after.startswith("-") and (tok in _OBJECT_PRONOUNS or tok in _OBJECT_DET):
            return False                      # "todo-o-terreno", "tudo-em-um"
        if tok in ("todo", "toda", "todos", "todas", "um", "uma") and (
                after in _TIME_NOUNS or (after in _ARTICLES and after2 in _TIME_NOUNS)):
            j += 1 if after in _TIME_NOUNS else 2    # "todo o dia", "uma noite"
            if j + 1 < len(words) and _strip(words[j + 1]) in _DURATION_MODIFIERS:
                j += 1                        # "uma noite inteira"
        elif ((tok == "de" and after in _ADVERBIAL_DE)
                or (tok in ("a", "à") and after in _A_LOCUTIONS)
                or ((tok in NUMERIC or tok in QUANT) and after in _FREQUENCY_NOUNS)
                or (tok in ("um", "uma") and after in _DEGREE_NOUNS)):
            j += 1                            # "de manhã", "duas vezes", "um pouco"
        elif tok in _OBJECT_PRONOUNS or tok in _OBJECT_DET:
            return True
        elif tok not in adverbs:
            return False
        if words[j][-1:] in ".!?;:,":
            return False
        j += 1
    return False


def _is_participle(word: str) -> bool:
    """Return True when *word* looks like a past participle: an irregular one
    ("escrito", "aceite") or an -ado/-ido form ("decidido", "baseada",
    "construída"), but not a noun such as "lado" or "sábado"."""
    if word in _IRREGULAR_PARTICIPLES:
        return True
    stem = word.removesuffix("s")
    if stem in _PARTICIPLE_LOOKALIKES:
        return False
    return stem.endswith(("ado", "ada", "ido", "ida", "ído", "ída")) and len(stem) >= 4


def _fora_is_verb(words: list, idx: int) -> bool:
    """Return True when "fora" is the pluperfect of ser or ir (closed fôra).

    The adverb, preposition and interjection (open fóra) are the frequent
    reading, so the verb needs one of these cues:

    - a past participle after it, perhaps after an adverb: the pluperfect
      passive ("assim fora decidido", "fora então construída");
    - a subject pronoun, a proclitic, a relative, the conditional "se" or a
      negation before it, perhaps before an adverb ("se ela fora rica", "lhe
      fora dito", "o homem que fora", "ele já fora rei", "nunca fora tão
      feliz");
    - a clause-initial definite article and noun before it ("O dia fora
      longo"). An indefinite one opens a span of time ("Uma semana fora
      cansou-o").

    A preposition or "lá"/"cá" before it ("para fora", "lá fora") and "de" or a
    contraction of "de" after it ("fora de casa", "fora daqui") mark the
    adverb whatever else is near. An enclitic pronoun after it marks the verb,
    because only a verb hosts one ("fora-lhe dito", "fora-se embora").
    Punctuation after the word before it breaks the cue of that word ("que,
    fora a Netflix, …").
    """
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)
    if next_word.startswith("-") and next_word[1:] in _ENCLITICS:
        return True
    if prev_word in _FORA_ADVERB_PREV or next_word in _FORA_ADVERB_NEXT:
        return False
    j = idx + 1
    while j < len(words) - 1 and j - idx <= 2 and words[j][-1:] not in ".,;:!?" and (
            _strip(words[j]) in _FORA_BRIDGE or _strip(words[j]).endswith("mente")):
        j += 1
    if j < len(words) and _is_participle(_strip(words[j])):
        return True
    subject, gap = prev_word, 1
    if prev_word in _FORA_BRIDGE and idx > 1:
        subject, gap = _prev2(words, idx), 2              # "ele já fora rei"
    if subject in _FORA_VERB_PREV and not any(
            w[-1:] in ".,;:!?" for w in words[idx - gap:idx]):
        return True
    clause_start = idx == 2 or (idx > 2 and words[idx - 3][-1:] in ".,;:!?")
    return (clause_start and _prev2(words, idx) in _ARTICLES
            and words[idx - 1][-1:] not in ".,;:!?"
            and prev_word not in DET | QUANT | PRON | _FUNC_EXTRA)


def score_noun(words: list, idx: int) -> int:
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)

    word = words[idx]
    score = 0
    # An intensifier before "sobre" is an adverb ("sabe muito sobre X"), and
    # "a/uma tola" fits the noun and the substantive adjective equally.
    if prev_word in DET | QUANT and not (word == "sobre" and prev_word in _SOBRE_INTENS) \
            and word != "tola":
        score += 5
        # After a determiner "sobre" can only be the noun (the sail).
        if word == "sobre":
            score += 7
    # "posto" as the participle of "pôr" ("foi posto") has the closed o of the noun.
    if word == "posto" and prev_word in _PASSIVE_AUX:
        score += 8
    # "do/da/dos/das" open a genitive. Plain "de" stays out: "gosto de X" is a verb.
    if next_word in _DE_CONTRACTIONS:
        score += 3
    # Closed-o noun locutions «em torno de», «ao/no/do torno» and «em troco de»;
    # the 1sg verbs «torno»/«troco» never follow these prepositions.
    if word == "torno" and prev_word in _AROUND_PREP:
        score += 6
    if word == "troco" and prev_word == "em":
        score += 6
    # A prenominal adjective before it makes it the noun head ("o atual governo").
    if prev_word in _PRENOMINAL_ADJ:
        score += 5
    # A complement preposition before it ("de gelo", "sem retorno"); not for
    # «colher», where "de colher" can be the infinitive.
    if prev_word in _COMPLEMENT_PREPS and word != "colher":
        score += 3
    # Coordination or comparison before it ("X e Y", "como X"): real text is
    # noun-dominant, so this cuts the most common false-verb error.
    if prev_word in _COORDINATION:
        score += 3
    # The object of a transitive verb ("encontrou consolo", "procura emprego").
    if prev_word in _TRANS_VERB:
        score += 3
    # "pelo" as fur: "tem pelo", "o pelo", "pelo do cão". por+o never takes an
    # article of its own.
    if word == "pelo" and prev_word in _TER:
        score += 6
    if word == "pelo" and prev_word in _ARTICLES:
        score += 6
    if word == "pelo" and next_word in ("do", "da"):
        score += 5
    # "cão de pelo comprido". This +6 still loses to the fixed phrases of
    # score_adp ("de pelo menos" scores ADP 11 or more).
    if word == "pelo" and prev_word == "de":
        score += 6
    # The royal court (closed o) is feminine: «a corte», «na corte». The cut (open
    # ɔ, scored as VERB) is masculine.
    if word == "corte" and prev_word in _FEM_DET:
        score += 7
    # Court terms after it. Generic "de/do/da" stay out: "o corte do bolo".
    if word == "corte" and next_word in _COURT_TERMS:
        score += 3

    if word == "tola":
        # The colloquial head (skull) or the tola hardwood.
        left3 = _left(words, idx, 3)
        # A head-action verb before it ("bati com a tola"). A verb after it makes
        # tola the subject, the substantive adjective ("a tola perdeu X").
        if any(v in _HEAD_VERBS for v in left3):
            score += 6
        # A body-state verb after it ("a tola dele latejava").
        if any(v in _HEAD_STATE for v in _right(words, idx, 2)):
            score += 6
        # "a tola dele/dela"
        if prev_word in _TOLA_POSS_PREV and next_word in _PERSON_POSS:
            score += 5
        # "em tola", "de tola": the material
        if prev_word in _MATERIAL_PREP:
            score += 4
        # "na tola", "pela tola": the body part
        if prev_word in _HEAD_LOCATIVE:
            score += 5
        # "com a tola"
        if _prev2(words, idx) == "com" and prev_word in _SING_ARTICLES:
            score += 5
        # "[a] tola à roda", "tola ao ar"
        if next_word in _CONTRACTED_A and prev_word in _TOLA_DET_PREV:
            score += 4
        # "madeira de tola", "ripas de tola"
        if any(w in _WOOD_CONTEXT for w in left3 + _right(words, idx, 3)):
            score += 6

    # The noun IPA of "forma" (closed o) is the baking mould «fôrma» only. Every
    # other sense (shape, manner, 3sg formar) shares the open-o VERB reading, so
    # the noun keeps its score only with a mould cue.
    if word == "forma":
        next2 = _next2(words, idx)
        toks = [_strip(w) for w in words]
        # Shape and manner idioms win over a stray cooking word ("a forma de
        # preparar a massa").
        shape = (prev_word in _FORMA_SHAPE_PREV
                 or next_word == "como"
                 or (next_word == "de" and _is_infinitive(next2))
                 or next2 in _FORMA_SHAPE_NEXT)
        # "uma forma para fazer/o <bakeware>"
        purpose = (next_word == "para"
                   and (next2 in _BAKE_PURPOSE or next2 in _BAKEWARE
                        or (next2 in _SING_ARTICLES
                            and _strip(words[idx + 3] if idx + 3 < len(words) else "") in _BAKEWARE)))
        if shape:
            score = 0
        elif (any(t in _MOULD_CUES for t in toks)
              or (next_word == "de" and next2 in _BAKEWARE) or purpose):
            score += 6
        else:
            score = 0

    # The verb molhar takes a direct object or a clitic ("molho os pés",
    # "molho-as"), often after an adverb. These mark the noun instead:
    #   - an interrogative or a verb of lack before it ("que molho escolhes",
    #     "falta molho"). Interrogative "qual"/"quais" is decisive even before an
    #     article ("Qual molho, o branco?"); after an article or a contraction it
    #     is the relative pronoun, which a verb follows ("com a qual molho as plantas");
    #   - at the start of a clause, the absence of an object (_molho_takes_object);
    #   - elsewhere, "de" + a noun after it ("molho de tomate"), but not an
    #     adverbial locution ("molho de manhã as plantas").
    if word == "molho":
        no_object = next_word not in _ARTICLES | QUANT and not next_word.startswith("-")
        relative = _prev2(words, idx) in DET | _POR_CONTRACTIONS
        if prev_word in ("qual", "quais") and not relative:
            score += 12
        elif prev_word in _MOLHO_NOUN_PREV and no_object:
            score += 8
        if idx == 0 or words[idx - 1][-1:] in ",;:":
            if not _molho_takes_object(words, idx):
                score += 8
        elif next_word == "de" and _next2(words, idx) not in _ADVERBIAL_DE:
            score += 4
    return score


def score_verb(words: list, idx: int) -> int:
    word = words[idx]
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)
    prev2_word = _prev2(words, idx)

    # The adverb is the default reading of "fora"; only a cue of the pluperfect
    # of ser/ir overrides it, so the generic signals below do not apply.
    if _strip(word) == "fora":
        return 8 if _fora_is_verb(words, idx) else 0

    next2_word = _next2(words, idx)
    # The infinitive tests read the raw token, as a direct caller may pass glued
    # quotes: _is_infinitive("comer»") is False, _is_infinitive("comer") is True.
    next2_raw = words[idx + 2] if idx + 2 < len(words) else ""
    score = 0

    # Sentence-initial, so no article before it: likely a 1st-person verb.
    if idx == 0:
        score += 3

    if prev_word in PRON:
        score += 5
        if idx == 1:
            score += 2

    # A 1sg form (choro, começo, acordo) with no determiner, transitive verb or
    # complement preposition before it is a verb: the noun needs a determiner
    # ("o choro"), and "tem choro fácil" or "de peso" are noun objects.
    if (idx > 0
            and prev_word not in DET | QUANT
            and prev2_word not in DET | QUANT
            and prev_word not in _TRANS_VERB
            and prev_word not in _COMPLEMENT_PREPS):
        if word in _FIRST_PERSON_NOUNS:
            score += 2

    # A 1sg form + a complement preposition + a pronoun ("gosto de ti", "olho
    # para ti", "rio de mim").
    if (word in _FIRST_PERSON_NOUNS
            and next_word in _COMPLEMENT_PREPS | {"para", "a", "ao", "à"}
            and next2_word in PRON
            and prev_word not in DET | QUANT | _COMPLEMENT_PREPS | _TRANS_VERB):
        score += 5

    # An enclitic hyphenated onto it ("torno-me", "fá-lo") marks a finite verb.
    # The tokenizer keeps the hyphen, which rules out the article and preposition
    # readings of o/a/nos/vos. "para se/me/te" is the preposition + a proclitic.
    if word != "para" and next_word.startswith("-") and next_word[1:] in _ENCLITICS:
        score += 4

    # A determiner after it opens a direct object. Excluded: "para o/a" (the
    # preposition); the contracted forms, which open a prepositional phrase ("seco
    # no verão", "gozo na prática"); a postposed quantifier ("o peso todo"); and
    # "posto a [infinitive]", the participle of "pôr" with an infinitive, where
    # "a" is no article ("começo a [inf]" is still a verb + object).
    posto_inf = word == "posto" and next_word == "a" and _is_infinitive(next2_raw)
    if (word != "para" and next_word in DET | QUANT and next_word not in _CONTRACTED_DET
            and next_word not in _POSTPOSED_QUANT and not posto_inf):
        score += 3
    elif idx == 0 and next_word in _DE_CONTRACTIONS:
        score += 1

    # The verb bucket of "corte" is the cut (a masculine noun) and the
    # subjunctive of «cortar», both open ɔ. The royal court is only feminine, so
    # a masculine determiner («o/do/no corte») decides.
    if word == "corte" and prev_word in _MASC_DET:
        score += 10
    # Cut terms within 2 tokens: "um corte profundo", "corte de cabelo".
    if word == "corte" and any(w in _CUT_NOUNS for w in _left(words, idx, 2) + _right(words, idx, 2)):
        score += 4

    if prev_word in _TEMPORAL_CONJ:
        score += 2
    if prev_word in _CONJ_SUBJ:
        score += 4

    # Negation or frequency adverb before it: "não gosto", "nunca sobre".
    if prev_word in _NEG_ADV:
        score += 4

    # Past or subjunctive of "ser" before it: a passive participle. Copular and
    # active forms stay out ("tem gosto", "é sobre X").
    if prev_word in _PASSIVE_AUX:
        score += 4

    if word == "para" and _is_infinitive(next_word):
        score -= 5

    # "pára": a stoppable thing (vehicle, machine, bodily process) as its subject
    # or object.
    if word == "para":
        if any(w in STOPPABLE_THINGS for w in _left(words, idx, 4)):
            score += 4
        if next_word in STOPPABLE_THINGS:
            score += 3
        # "pára o/a [STOPPABLE]"
        if next_word in _ARTICLES and next2_word in STOPPABLE_THINGS:
            score += 3
        # "para a meio": stops halfway
        if next_word == "a" and next2_word == "meio":
            score += 4

    # An infinitive before it, in the same clause ("Depois de nadar, seco…" is
    # two clauses).
    prev_raw = words[idx - 1] if idx > 0 else ""
    if _is_infinitive(prev_word) and not (prev_raw and prev_raw[-1] in ".,;:!?"):
        score -= 5

    # The word itself is an infinitive, unless a determiner makes it a noun
    # ("a colher", the spoon).
    if _is_infinitive(word) and prev_word not in DET | QUANT:
        score += 5
    # "a colher o/os X" (to harvest the X), against "a colher de X" (the spoon).
    if word == "colher" and prev_word == "a" and next_word in DET | QUANT \
            and next_word not in _CONTRACTED_DET:
        score += 6
    # A control verb 2 to 4 tokens before "a colher" ("começou a colher"), unless
    # an infinitive sits between them ("aprendeu a usar a colher").
    left4 = _left(words, idx, 4)
    if word == "colher" and prev_word == "a" and not _is_infinitive(prev2_word) and any(
            w in _COLHER_CTRL for w in left4[1:]):
        score += 6

    # "DET NOUN VERB": a subject determiner two places back.
    if prev2_word in _SUBJ_DET:
        score += 3

    if word == "para" and prev_word in _PARA_PREV_SUBJ:
        score += 3

    if word == "para":
        # "pára no/durante/após X": NEVER_AFTER_PREP already zeroes ADP; this tips the tie.
        if next_word in _PARA_VERB_NEXT:
            score += 5
        # "para de [infinitive]": stops doing X
        if next_word == "de" and _is_infinitive(next2_raw):
            score += 5
        # "para quando/enquanto": the machine stops when…
        if next_word in _TEMPORAL_CONJ:
            score += 3

    # A -mente adverb after it: "seco rapidamente as mãos".
    if next_word.endswith("mente"):
        score += 3

    # A short adverb between the verb and its object: "seco primeiro as mãos".
    # Prepositions such as "debaixo/antes/depois" are not in the list.
    if next_word in _ADV_BRIDGE and next2_word in DET | QUANT:
        score += 1

    # "gozo de X" without a determiner or preposition before it is 3sg «gozar de»;
    # "o gozo de", "em gozo de" are the noun.
    if word == "gozo" and next_word == "de" and prev_word not in _GOZO_EXCL_PREV:
        score += 5

    # "sempre sobre um/uma": the verb sobrar (to be left over).
    if word == "sobre" and prev_word == "sempre" and next_word in DET | QUANT:
        score += 8

    if word == "sobre":
        # A -mente adverb before it ("raramente sobre") modifies a verb.
        if prev_word.endswith("mente"):
            score += 4
        # "Depois de X, sobre Y": what is left after a subtraction.
        left3 = _left(words, idx, 3)
        if "depois" in left3 or "restam" in left3:
            score += 3
        # "talvez sobre", "oxalá sobre": present subjunctive of sobrar.
        if prev_word in _OPTATIVE_ADV:
            score += 6

    return score


def score_adj(words: list, idx: int) -> int:
    prev_word = _prev(words, idx)
    prev2_word = _prev2(words, idx)
    next_word = _next(words, idx)
    prev_raw = words[idx - 1] if idx > 0 else ""

    score = 0

    # Predicative: a copula before it.
    if prev_word in _COPULA:
        score += 5

    # Comparative or superlative: "mais/menos seco".
    if prev_word in _COMPARATIVES:
        score += 4

    # Degree: "muito seco", "particularmente seco".
    if prev_word in _INTENSIFIERS:
        score += 2
    if prev_word.endswith("mente"):
        score += 4
    # "foi particularmente seco"
    if prev2_word in _COPULA and prev_word.endswith("mente"):
        score += 3

    # Attributive "DET NOUN ADJ", "Que ideia tola", "nenhum produto seco", in one
    # clause ("Depois do banho, seco" is two).
    across_boundary = prev_word == "" or (prev_raw and prev_raw[-1] in ".,;:!?")
    if prev2_word in DET | QUANT | _EXCL_DET \
            and prev_word not in DET | QUANT | _EXCL_DET \
            and not across_boundary:
        score += 4

    # A bare determiner before it (a noun phrase head; less common for ADJ).
    if prev_word in DET | QUANT | PRON:
        score += 3

    # Postpositive "NOUN ADJ": a content word right before it, in the same clause.
    # It has more than 2 characters, which skips every one- and two-character word, such
    # as the prepositions "a", "de" and "em". It is not a function word and does
    # not look like a finite verb, and no determiner follows ("seco os pratos" is
    # a verb + object). An enclitic ("disse-me") marks a verb host, not a noun.
    prev_looks_verb = (_looks_verb(prev_word)
                       or ("-" in prev_word and prev_word.rsplit("-", 1)[-1] in _CLITICS_ALL))
    if (prev_word and len(prev_word) > 2
            and prev_raw[-1] not in ".,;:!?"
            and prev_word not in _FUNC_WORDS
            and next_word not in DET | QUANT
            and not prev_looks_verb):
        score += 3

    return score


def guess_pos(words: list, idx: int, proper: bool = False) -> str:
    """Return the most likely UDEP POS tag for the ambiguous word at *idx*.

    *proper* marks a mid-sentence capitalised word (a likely proper noun, as in
    "Cerro Corá"): it adds a strong NOUN bias unless an enclitic after it marks
    a verb host. When no POS scores above 0, or several tie, the word's
    DEFAULT_POS wins.
    """
    word = words[idx]
    # Corpus-frequency priors, small enough that any context signal overrides them.
    bias = BASE_SCORE.get(word, {})

    scores = {}
    if word in ADP_IPA:
        scores["ADP"] = score_adp(words, idx) + bias.get("ADP", 0)
    if word in NOUNS_IPA:
        scores["NOUN"] = score_noun(words, idx) + bias.get("NOUN", 0)
    if word in VERBS_IPA:
        scores["VERB"] = score_verb(words, idx) + bias.get("VERB", 0)
    if word in ADJ_IPA:
        scores["ADJ"] = score_adj(words, idx) + bias.get("ADJ", 0)

    if proper and "NOUN" in scores:
        nxt = _next(words, idx)
        if not (nxt.startswith("-") and nxt[1:] in _ENCLITICS):
            scores["NOUN"] += 12

    best_score = max(scores.values())
    if best_score <= 0:
        return DEFAULT_POS.get(word, "NOUN")

    winners = [pos for pos, s in scores.items() if s == best_score]
    if len(winners) == 1:
        return winners[0]
    return DEFAULT_POS.get(word, winners[0])


def _resolve_sede(words: list, idx: int) -> str:
    """Return SEAT (HQ, open ɛ) or THIRST (closed e) for "sede".

    Both senses are nouns. The preposition frame is the strongest cue ("sede de
    X" → thirst, "sede da/do X" → seat), with the sede_*_cues.voc wordlists in
    the ±3 window.
    """
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)
    next2 = _next2(words, idx)

    seat = thirst = 0
    if next_word in {"da", "do", "das", "dos"}:
        seat += 3
    if next_word == "de":
        if next2 in DET:                       # "sede de uma organização/do clube" = HQ
            seat += 3
        elif next2 in _SEDE_SEAT:              # "sede de futebol clube" = HQ
            seat += 4
        elif next2 in _SEDE_THIRST or next2.endswith(_SEDE_ABSTRACT):  # "sede de poder/reconhecimento"
            thirst += 4
        else:                                  # "sede de Atenas/Lisboa" (place) → HQ
            seat += 2
    if prev_word == "de":                       # "morto de sede", "queixou-se de sede"
        thirst += 3
    if prev_word in {"na", "à", "numa", "pela", "duma"}:   # locative: the HQ building
        seat += 2
    if next_word in {"no", "na", "nos", "nas", "em"}:      # "sede no Porto" = HQ at a place
        seat += 3
    window = [_strip(words[i]) for i in range(max(0, idx - 3), min(len(words), idx + 4))
              if i != idx]
    seat += sum(w in _SEDE_SEAT for w in window)
    thirst += sum(w in _SEDE_THIRST for w in window)

    if seat > thirst:
        return "seat"
    if thirst > seat:
        return "thirst"
    return DEFAULT_SENSE.get("sede", "thirst")


def _resolve_molho(words: list, idx: int) -> str:
    """Return BUNDLE (open ɔ, "um molho de chaves") or SAUCE (closed o) for the
    noun "molho". The verb molhar reaches here as pos=VERB, which maps to
    bundle upstream. Bundle needs an explicit bundled thing; sauce is the default.
    """
    next_word = _next(words, idx)
    next2 = _next2(words, idx)
    if next_word == "de" and next2 in _BUNDLE_THINGS:              # "molho de chaves/lenha"
        return "bundle"
    if (next_word == "de" and next2 in _BUNDLE_AMBIG                # "colhi um molho de salsa"
            and any(v in _BUNDLE_VERBS for v in _left(words, idx, 4))):
        return "bundle"
    return "sauce"


def _resolve_tola(words: list, idx: int) -> str:
    """Return "head" for "tola" once its POS guess is NOUN.

    The colloquial head and the tola hardwood (both open ˈtɔlɐ) share the slug.
    The POS scorer sends the foolish adjective (closed ˈtolɐ) to ADJ, even when
    it stands as a noun; ``foolish`` carries ``ADJ|NOUN`` only so an external
    NOUN tag defers to the rules.
    """
    return "head"


def _resolve_by_cues(word: str, words: list, idx: int) -> str:
    """Return the sense of *word* whose :data:`bifonia.cues.SENSE_CUES` cue list
    scores highest with :func:`cue_score`, or the rule's default when none scores.

    A cue with ``requires_prev`` counts only when the token before the homograph
    is that word (the "de" in "de cor").
    """
    rule = SENSE_CUES[word]
    best, best_score = rule.default, 0
    for cue in rule.cues:
        if cue.requires_prev is not None and _prev(words, idx) != cue.requires_prev:
            continue
        score = _cue_score(words, idx, _folded(voc(cue.voc)))
        if score > best_score:
            best, best_score = cue.sense, score
    return best


# Same-POS words whose resolution needs more than a cue competition.
_SENSE_RESOLVERS = {"sede": _resolve_sede, "molho": _resolve_molho, "tola": _resolve_tola}


def resolve_sense(word: str, words: list, idx: int, pos: str) -> str:
    """Return the meaning slug for *word* at *idx*, given its guessed *pos*.

    When *pos* maps to one sense this is a lookup. A *pos* the word lacks falls
    back to its DEFAULT_POS, then to the senses of its first POS. Several senses
    go to the SENSE_CUES competition, a hand-written resolver, or the first sense.
    """
    senses = POS_SENSES.get(word, {}).get(pos)
    if not senses:
        senses = POS_SENSES.get(word, {}).get(DEFAULT_POS.get(word, ""))
    if not senses:
        senses = next(iter(POS_SENSES.get(word, {}).values()), [None])
    if len(senses) == 1:
        return senses[0]
    if word in SENSE_CUES:
        return _resolve_by_cues(word, words, idx)
    resolver = _SENSE_RESOLVERS.get(word)
    return resolver(words, idx) if resolver else senses[0]

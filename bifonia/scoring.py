"""
Context-based POS scoring for Portuguese heterophonic bifoniaaphs.

Each scorer returns an integer; higher = more confident.
Negative scores signal anti-evidence for that POS.
"""

from bifonia.data import (
    ADP_IPA, ADJ_IPA, NOUNS_IPA, VERBS_IPA,
    DEFAULT_POS, DEFAULT_SENSE, BASE_SCORE,
    POS_SENSES,
    DET, PRON, AUX_VERBS, NUMERIC,
    BEFORE_PREP, AFTER_PREP, NEVER_AFTER_PREP,
    SOBRE_GOV, QUANT, STOPPABLE_THINGS,
)
from bifonia.vocab import voc

# Wordlists are externalised to locale/<lang>/*.voc (see vocab.py) so they can
# be extended without code changes. Derived/structural sets stay in code below.
_POST_CLITICS    = voc("post_clitics")      # enclitic reflexive clitics (verb host)
_AFTER_PREP_WEAK = voc("after_prep_weak")   # time adv/pron: weak ADP credit
_COPULA          = voc("copula")            # copular/semi-copular verbs
_INTENSIFIERS    = voc("intensifiers")      # degree intensifiers (modify ADJ)
_TEMPORAL_CONJ   = voc("temporal_conj")     # quando/enquanto
_CONJ_SUBJ       = voc("conj_subjunctive")  # subjunctive-introducing conjunctions
_NEG_ADV         = voc("neg_adv")           # negation/frequency adverbs before finite verb
_PASSIVE_AUX     = voc("passive_aux")       # past/subjunctive of ser (passive)

# per-word semantic lists (formerly inline in the scorers)
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
_BAKE_PURPOSE = voc("bake_purpose")
_FIRST_PERSON_NOUNS = voc("first_person_nouns")
_PARA_PREV_SUBJ = voc("para_prev_subj")
_PREP_GOVERNING = voc("prep_governing")
_CLITICS_ALL  = voc("clitics_all")
_EXCL_DET     = voc("exclamative_det")
_LOCATIVE_CONTRACTIONS = voc("locative_contractions")
_FUNC_EXTRA = voc("function_words")
# meaning-level cues for words whose senses share a POS (only "sede" today)
_SEDE_SEAT   = voc("sede_seat_cues")
_SEDE_THIRST = voc("sede_thirst_cues")
# contracted prep+article forms (shared by several scorers)
_CONTRACTED_DET = voc("contracted_det")
_VERB_DET_EXCL = _COLHER_DET_EXCL = _CONTRACTED_DET


_PUNCT = str.maketrans("", "", ".,;:!?\"'()[]{}«»–—")


def _strip(w: str) -> str:
    return w.translate(_PUNCT)


def _prev(words: list, idx: int) -> str:
    return _strip(words[idx - 1]) if idx > 0 else ""


def _next(words: list, idx: int) -> str:
    return _strip(words[idx + 1]) if idx + 1 < len(words) else ""


def _prev2(words: list, idx: int) -> str:
    return _strip(words[idx - 2]) if idx > 1 else ""


def _is_infinitive(word: str) -> bool:
    w = word.rstrip(".,;:!?")
    return w.endswith(("ar", "er", "ir"))


def _is_numeric(word: str) -> bool:
    try:
        float(word)
        return True
    except ValueError:
        return word in NUMERIC


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

    # "para não/nunca/jamais X" is always purpose ADP + negated infinitive.
    # Limited to pure negation adverbs — frequency adverbs ("sempre", "raramente")
    # can follow a finite VERB "pára" ("pára sempre" = stops always).
    if word == "para" and next_word in _PURE_NEG:
        score += 4

    if word == "para":
        # "para [deverbal noun]" — purpose ADP + nominalised VP (para análise, para
        # revisão, para avaliação, para consideração, etc.).  Deverbal nouns typically
        # end in -ção/-são/-ão, -gem, -ura, -ência/-ância, -mento, -ismo, or are
        # bare infinitive-like forms ending in -ar/-er/-ir (already handled by
        # _is_infinitive).  Detect by common suffix.
        _DEVERBAL_SFXS = ("ção", "são", "gem", "ura", "ência", "ância",
                          "mento", "ismo", "ise", "ise", "ção")
        if next_word.endswith(_DEVERBAL_SFXS):
            score += 4
        # "para depois/amanhã" — deferred-purpose ADP.
        # "depois" is in AFTER_PREP_WEAK (+2) but needs more to beat DET-2back VERB+3.
        # Keep narrow: "sempre/logo/aqui/lá" can also follow finite VERB "pára".
        if next_word in _DEFERRAL_ADV:
            score += 3
        # "está/é [ADJ] para quando/…" — copula+predicative ADJ before "para" signals
        # purpose ADP, not finite VERB "pára".
        _para_prev2 = _strip(words[idx - 2]) if idx >= 2 else ""
        if _para_prev2 in _COPULA:
            score += 4

    if word == "pelo":
        # Fixed idiomatic ADP phrases: "pelo menos", "pelo contrário", "pelo amor",
        # "pelo visto", "pelo que", "pelo andar", "pelo sim pelo não", etc.
        # Also possessives: "pelo seu bem", "pelo teu cálculo", "pelo meu entender".
        # And temporal/occasion nouns: "pelo Natal", "pelo Páscoa", "pelo verão".
        if next_word in _PELO_FIXED:
            score += 6
        # "pelo [route/place noun]" — por+o contracted ADP, common with geographic/
        # directional nouns that aren't in AFTER_PREP.
        if next_word in _PELO_ROUTE:
            score += 5
        # Passive-voice agent: "foi transmitido pelo X", "foi aprovado pelo Y"
        # A past participle in the ±3 left context strongly signals ADP agent.
        # Guard: the char before the suffix must be a consonant so that animal/proper
        # names ending in -ida/-ada (e.g. "samoeida") don't fire as false positives.
        _PPT_SFXS = ("ado", "ido", "ada", "ida", "ados", "idos", "adas", "idas")
        _VOWELS = frozenset("aeiouáéíóúâêîôûãõ")
        _left3 = [_strip(words[max(0, idx - k)]) for k in range(1, 4) if idx - k >= 0]
        def _is_ppt(w):
            for sfx in _PPT_SFXS:
                if w.endswith(sfx) and len(w) > len(sfx) and w[-len(sfx)-1] not in _VOWELS:
                    return True
            return False
        if any(_is_ppt(w) for w in _left3):
            score += 5
        # "pelo" = por+o (masc.sg.); if followed by any free-standing definite article
        # it cannot be the ADP contraction (por+o already contains the article "o").
        # Return strongly negative so score_verb wins.
        if next_word in _ARTICLES:
            score -= 8

    if word == "sobre":
        # Explicit governing noun/verb before "sobre" (e.g. "caso sobre X",
        # "falou sobre X") — strongest explicit ADP signal.
        if prev_word in SOBRE_GOV:
            score += 6
        # Intensifier scale quantifiers before "sobre": "sabe muito sobre",
        # "aprendeu pouco sobre" — always ADP (about/concerning).
        if prev_word in _SOBRE_ADP_QUANT:
            score += 5
        # "sobre + DET/PRON" — "sobre o/a/este/ela..." is a clear ADP pattern
        # meaning "about the..." (already handled by AFTER_PREP above, but
        # reinforce it so it beats VERB signals from prev2-DET).
        if next_word in DET | PRON:
            score += 3
        # Mild base prior: mid-sentence "sobre" after any non-pronoun is
        # almost always ADP, not VERB — but when a negation/frequency adverb or
        # temporal conjunction precedes, the word is likely a finite VERB, so
        # withhold the prior there.
        if idx > 0 and prev_word not in PRON | _NEG_ADV | _TEMPORAL_CONJ:
            score += 2

    return score


def score_noun(words: list, idx: int) -> int:
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)

    word = words[idx]
    score = 0
    # Intensifiers (muito/pouco/bastante/…) can precede "sobre" as adverbs
    # ("sabe muito sobre X") — don't treat them as NOUN determiners there.
    if prev_word in DET | QUANT and not (word == "sobre" and prev_word in _SOBRE_INTENS) \
            and word != "tola":  # "a/uma tola" is equally DET+NOUN and DET+subst-ADJ
        score += 5
        # "sobre" as a noun (nautical sail) is unambiguous when a DET immediately
        # precedes — it cannot function as ADP after a determiner.  Boost
        # strongly so the ADP signal from the following DET doesn't win.
        if word == "sobre":
            score += 7
    # "posto" as PPT of "pôr" (to place/put) is pronounced like the NOUN (closed-o),
    # not like the VERB "postar" (to post, open-o).  Boost NOUN when a passive
    # auxiliary precedes, e.g. "foi posto", "estava posto", "ficou posto".
    if word == "posto" and prev_word in _PASSIVE_AUX:
        score += 8
    # "do/da/dos/das" strongly indicate a following possessive/partitive phrase.
    # Plain "de" excluded: ambiguous with "gosto de X" (VERB) constructions.
    if next_word in _DE_CONTRACTIONS:
        score += 3
    # "pelo" as fur (NOUN): "tem pelo", "tinha pelo" — transitive possession verb directly
    # before "pelo" signals body-hair/fur reading, not the ADP contraction (por+o).
    if word == "pelo" and prev_word in _TER:
        score += 6
    # "cão de pelo comprido" — genitive "de" directly before "pelo" followed by a
    # qualitative adjective is always the fur-type construction, not ADP (por+o).
    # "pelo menos" / "pelo visto" etc. are guarded by _PELO_FIXED in score_adp (+6),
    # so a NOUN+6 here still loses to those fixed-phrase ADP signals (total ADP≥11).
    if word == "pelo" and prev_word == "de":
        score += 6
    # "tola" as colloquial NOUN (head/skull): detected by verbs of hitting/filling/
    # possessive clitic contexts — "bater com a tola", "meter na tola", "partir a tola",
    # "dói-me a tola", "a tola à roda".
    # "corte" NOUN (royal court, *closed* o) is grammatically feminine in Portuguese:
    # «a corte», «da corte», «na corte», «à corte», «uma corte»
    # Feminine determiner before "corte" is a strong signal for this reading.
    # "corte" NOUN/cut (open o, scored as VERB) is masculine — no boost here.
    if word == "corte" and prev_word in _FEM_DET:
        score += 7
    # Court-specific adjectives/nouns immediately after also confirm royal court.
    # NOTE: do NOT include generic "de/do/da" here — the cut sense also takes
    # them ("o corte do bolo", "corte de cabelo"), so they are not court signals.
    if word == "corte" and next_word in _COURT_TERMS:
        score += 3

    if word == "tola":
        _left3 = [_strip(words[max(0, idx - k)]) for k in range(1, 4) if idx - k >= 0]
        # head-action verb to the LEFT = tola is its object ("bati com a tola",
        # "perdeu a tola").  Right-side action verbs are excluded: "a tola perdeu
        # X" has tola as the SUBJECT (substantivised ADJ), not the head-object.
        if any(v in _HEAD_VERBS for v in _left3):
            score += 6
        # body-state verb to the RIGHT = head as subject ("a tola dele latejava").
        _right2h = [_strip(words[idx + k]) for k in range(1, 3) if idx + k < len(words)]
        if any(v in _HEAD_STATE for v in _right2h):
            score += 6
        # "a tola dele/dela" — possessive of person = colloquial head (NOUN)
        if prev_word in _TOLA_POSS_PREV \
                and next_word in _PERSON_POSS:
            score += 5
        # "em tola" / "de tola" — material construction = hardwood (NOUN)
        if prev_word in _MATERIAL_PREP:
            score += 4
        # "na tola" / "pela tola" — locative preposition signals body-part head
        if prev_word in _HEAD_LOCATIVE:
            score += 5
        # "com a tola" — prev2="com" + prev="a" (article)
        _prev2_tola = _strip(words[idx - 2]) if idx >= 2 else ""
        if _prev2_tola == "com" and prev_word in _SING_ARTICLES:
            score += 5
        # "[a] tola à roda" / "tola ao ar" — "à/ao" immediately after signals idiom
        if next_word in _CONTRACTED_A and prev_word in _TOLA_DET_PREV:
            score += 4
        # tola = African hardwood: "madeira de tola", "ripas de tola", "em tola"
        _right3 = [_strip(words[idx + k]) for k in range(1, 4) if idx + k < len(words)]
        if any(w in _WOOD_CONTEXT for w in _left3 + _right3):
            score += 6

    # "forma" NOUN IPA (closed-o) = the baking mould / fôrma ONLY.  All other
    # senses (figura, modo, maneira, formatura, 3sg formar) share the open-o VERB
    # reading.  The generic DET-before / de-after bonuses above fire for both, so
    # for "forma" keep the NOUN score only when an explicit mould cue is present;
    # otherwise zero it so the open-o reading (VERB default) wins.
    if word == "forma":
        _next2_f = _strip(words[idx + 2]) if idx + 2 < len(words) else ""
        _toks_f = [_strip(w) for w in words]
        # Shape/manner idioms force the open-o VERB reading even if a cooking word
        # happens to appear ("a forma de preparar a massa" = manner, not a tin).
        # Shape/manner idioms force the open-o VERB reading even if a cooking word
        # happens to appear ("a forma de preparar a massa" = manner, not a tin).
        _shape = (prev_word in _FORMA_SHAPE_PREV
                  or next_word == "como"
                  or (next_word == "de" and _is_infinitive(_next2_f))
                  or _next2_f in _FORMA_SHAPE_NEXT)
        # "uma forma para fazer/o <bakeware>" — purpose phrasing of a baking tin.
        _purpose = (next_word == "para"
                    and (_next2_f in _BAKE_PURPOSE or _next2_f in _BAKEWARE
                         or (_next2_f in _SING_ARTICLES
                             and _strip(words[idx + 3] if idx + 3 < len(words) else "") in _BAKEWARE)))
        if _shape:
            score = 0
        elif (any(t in _MOULD_CUES for t in _toks_f)
              or (next_word == "de" and _next2_f in _BAKEWARE) or _purpose):
            score += 6
        else:
            score = 0
    return score


def score_verb(words: list, idx: int) -> int:
    word = words[idx]
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)
    prev2_word = _prev2(words, idx)

    score = 0

    # Sentence-initial position without a preceding article → likely 1st-person verb.
    if idx == 0:
        score += 3

    if prev_word in PRON:
        score += 5
        if idx == 1:
            score += 2

    # 1st-person verb forms (choro, começo, acordo, …) without a preceding
    # determiner are almost certainly finite verbs, not nouns.  The noun reading
    # needs an article ("o choro", "um começo") which provides DET context.
    # Guards:
    #   - DET/QUANT in prev or prev2 → NOUN phrase context, skip
    #   - Transitive verb in prev ("tem", "tinha", "tenho", …) → NOUN object, skip
    #     ("tem choro fácil", "tem gosto refinado" are NOUN objects of "ter")
    if (idx > 0
            and prev_word not in DET | QUANT
            and prev2_word not in DET | QUANT
            and prev_word not in _TRANS_VERB):
        if word in _FIRST_PERSON_NOUNS:
            score += 2

    # Enclitic clitic pronoun right after the word → strong verb host signal.
    # "para" excluded: "para se", "para me", "para te" are always ADP + clitic
    # infinitive, not "para" the finite verb with an enclitic.
    if word != "para" and next_word in _POST_CLITICS:
        score += 4

    # DET directly after signals a direct-object NP — strong VERB evidence.
    # Exclusions:
    #   "do/da/dos/das" — partitive/possessive PPs, not DOs
    #   "ao/à/aos/às/no/na/nos/nas" — contracted preposition+article; introduce
    #     locative or dative PPs, not direct objects ("seco no verão" → ADJ,
    #     "gozo na prática" → NOUN; keeping these fired false VERB+3 there)
    #   "para" entirely excluded: "para o/a" is ADP, not VERB
    # DET/QUANT directly after signals a direct-object NP; excludes contracted
    # preposition+article forms that introduce locative/dative PPs.
    # "posto a [infinitive]" = PPT of "pôr" + infinitival complement; "a" is the
    # infinitive marker, not the article.  Suppress the DET-after signal only for
    # "posto" (other words like "começo a [inf]" are genuine VERB+DO phrases).
    _next_next = words[idx + 2] if idx + 2 < len(words) else ""
    _posto_inf = word == "posto" and next_word == "a" and _is_infinitive(_next_next)
    if word != "para" and next_word in DET | QUANT and next_word not in _VERB_DET_EXCL and not _posto_inf:
        score += 3
    elif idx == 0 and next_word in _DE_CONTRACTIONS:
        score += 1

    # "corte" VERB bucket covers: (1) cut/incision — masculine noun, open ɔ;
    # (2) subjunctive/imperative of «cortar» — open ɔ.
    # Masculine determiner before "corte" is a DECISIVE signal: royal court
    # (closed o, NOUN) is exclusively feminine «a corte», so any masculine
    # «o/do/no corte» is the open-ɔ cut / tennis-court reading (VERB bucket).
    if word == "corte" and prev_word in _MASC_DET:
        score += 10
    # Cut-specific nouns/adjectives in immediate context: "um corte profundo",
    # "corte orçamental", "corte de cabelo", "corte de energia".
    _left2_c = [_strip(words[idx - k]) for k in range(1, 3) if idx - k >= 0]
    _right2_c = [_strip(words[idx + k]) for k in range(1, 3) if idx + k < len(words)]
    if word == "corte" and any(w in _CUT_NOUNS for w in _left2_c + _right2_c):
        score += 4

    # Temporal or subjunctive-introducing conjunction before → word is a verb.
    if prev_word in _TEMPORAL_CONJ:
        score += 2
    if prev_word in _CONJ_SUBJ:
        score += 4

    # Negation / frequency adverb directly before → finite verb form.
    # "não gosto", "nunca sobre", "sempre sobre" etc.
    if prev_word in _NEG_ADV:
        score += 4

    # Passive auxiliary directly before → word is a past participle (VERB).
    # Restricted to clear past/subjunctive forms of "ser" to avoid firing on
    # copular/active uses ("tem gosto", "é sobre X" were false positives).
    if prev_word in _PASSIVE_AUX:
        score += 4

    # Infinitive after the word penalises "para" ADP being scored as VERB.
    if word == "para" and _is_infinitive(next_word):
        score -= 5

    # "para" VERB ("parar") — subject or object is a stoppable thing.
    # Check prev (subject) and next (object) for semantic stoppable entities
    # (vehicles, machines, bodily processes, etc.).  European Portuguese usage.
    if word == "para":
        _left4 = [_strip(words[max(0, idx - k)]) for k in range(1, 5) if idx - k >= 0]
        if any(w in STOPPABLE_THINGS for w in _left4):
            score += 4
        if next_word in STOPPABLE_THINGS:
            score += 3
        # "pára o/a [STOPPABLE]" — when the object is introduced by a bare article,
        # look one position further to find the stoppable noun head.
        _next2 = _strip(words[idx + 2]) if idx + 2 < len(words) else ""
        if next_word in _ARTICLES and _next2 in STOPPABLE_THINGS:
            score += 3
        # "para a meio" — stops halfway through; "meio" in this sense is never ADP.
        if next_word == "a" and _next2 == "meio":
            score += 4

    # Infinitive immediately before → word is probably in a nominal/infinitival context.
    # Guard: if the raw prev token ends in punctuation (clause boundary), the infinitive
    # is in a separate clause ("Depois de nadar, seco…") and must not penalise here.
    _prev_raw_v = words[idx - 1] if idx > 0 else ""
    if _is_infinitive(prev_word) and not (_prev_raw_v and _prev_raw_v[-1] in ".,;:!?"):
        score -= 5

    # Word itself is an infinitive → strong VERB evidence.
    # Guard: if a DET immediately precedes, the word is likely a substantivised
    # infinitive or a NOUN that happens to end in ar/er/ir (e.g. "colher" = spoon),
    # so the infinitive VERB signal is unreliable there.
    if _is_infinitive(word) and prev_word not in DET | QUANT:
        score += 5
    # "a colher <DET>" = infinitive + object NP (to harvest/collect X); contrast with
    # "a colher de X" = the spoon of X (NOUN).  When "a" precedes and a DET/QUANT
    # follows (not "de"), the word is an infinitive verb.
    # "a colher de X" = NOUN (the spoon of X); signal handled by score_noun.
    # "colher" as infinitive VERB: preceded by "a" and a control verb in recent context.
    # Control verbs typically appear at prev2/prev3 distance: "começou a colher",
    # "aprendeu a colher", "voltou a colher", etc.
    # "a colher [article NP]" = to collect [the X] — article directly after is VERB.
    # Exclude contracted preposition+article forms (no/na/ao/à/…) — those introduce
    # locative PPs on the NOUN, not verbal objects.
    if word == "colher" and prev_word == "a" and next_word in DET | QUANT \
            and next_word not in _COLHER_DET_EXCL:
        score += 6
    # Control verb at prev2/prev3/prev4 distance → "colher" is an infinitive VERB.
    prev3 = _strip(words[idx - 3]) if idx >= 3 else ""
    prev4 = _strip(words[idx - 4]) if idx >= 4 else ""
    # Guard: if prev2 is itself an infinitive (e.g. "aprendeu a usar a colher"),
    # "colher" is the NOUN object of the intermediate verb, not of the control verb.
    _colher_ctrl_ok = not _is_infinitive(prev2_word)
    if word == "colher" and prev_word == "a" and _colher_ctrl_ok and (
        prev2_word in _COLHER_CTRL or prev3 in _COLHER_CTRL or prev4 in _COLHER_CTRL
    ):
        score += 6

    # DET two positions back indicates "DET NOUN VERB" subject-verb pattern.
    # Contracted prepositions are excluded: they introduce prepositional phrases,
    # not nominal subjects.
    _SUBJ_DET = DET - _LOCATIVE_CONTRACTIONS
    if prev2_word in _SUBJ_DET:
        score += 3

    if word == "para" and prev_word in _PARA_PREV_SUBJ:
        score += 3

    if word == "para":
        # "pára [contracted-prep/during/after]" — finite verb followed by a
        # locative/temporal PP.  "para no/ao/durante/após/entre/em" is impossible
        # as ADP (NEVER_AFTER_PREP already zeros ADP; add VERB signal to tip the tie).
        if next_word in _PARA_VERB_NEXT:
            score += 5
        # "para de [infinitive]" = stops doing X (finite VERB + complement).
        _next_next_v = words[idx + 2] if idx + 2 < len(words) else ""
        if next_word == "de" and _is_infinitive(_next_next_v):
            score += 5
        # "para quando/enquanto" finite VERB: the machine stops when/while…
        if next_word in _TEMPORAL_CONJ:
            score += 3

    # Adverb ending in -mente directly after a word → the word is a finite verb.
    # "seco rapidamente as mãos", "começo imediatamente".
    if next_word.endswith("mente"):
        score += 3

    # DET/QUANT at distance +2 with a non-DET at +1 → finite verb with an intervening
    # manner/sequence adverb before its object NP: "seco primeiro as mãos".
    # Restricted to known short adverbs that commonly appear between a verb and its
    # object NP; "debaixo/antes/depois/…" are prepositions and must not trigger this.
    _next2 = _strip(words[idx + 2]) if idx + 2 < len(words) else ""
    if next_word in _ADV_BRIDGE and _next2 in DET | QUANT:
        score += 1

    # "gozo de X" with no DET/QUANT before = 3rd person of "gozar de" (to enjoy/benefit from).
    # "o gozo de X" with DET before = the enjoyment of (NOUN); that case gets NOUN+5 elsewhere.
    # "gozo de X" with a nominal/adj subject before = 3rd person of "gozar de".
    # Exclude: DET/QUANT before (→ NOUN via score_noun), or a preposition before
    # ("em gozo de", "com gozo de" are NOUN complement phrases).
    _GOZO_EXCL_PREV = DET | QUANT | _PREP_GOVERNING
    if word == "gozo" and next_word == "de" and prev_word not in _GOZO_EXCL_PREV:
        score += 5

    # "sempre sobre [uma/um/…]" — frequency adverb + "sobrar" (left over); not ADP.
    # "sempre sobre" where a DET/QUANT follows and there is no governing verb is a
    # finite VERB (sobrar) not a preposition.
    if word == "sobre" and prev_word == "sempre" and next_word in DET | QUANT:
        score += 8

    # "sobre" as VERB (sobrar, to be left over): expand signals beyond "sempre".
    if word == "sobre":
        # Degree/frequency adverbs ending in -mente directly before: "raramente sobre",
        # "dificilmente sobre", "normalmente sobre", etc.  Adverbs of this type modify
        # finite verbs, not prepositions.
        if prev_word.endswith("mente"):
            score += 4
        # "Depois de X, sobre Y" — leftover-after-subtraction pattern.
        # "depois" in the ±3 left window strongly implies a subtraction result.
        _left3s = [_strip(words[max(0, idx - k)]) for k in range(1, 4) if idx - k >= 0]
        if "depois" in _left3s or "restam" in _left3s:
            score += 3
        # Optative adverb directly before → present subjunctive of "sobrar"
        # ("talvez sobre", "oxalá sobre", "tomara que sobre"): always finite VERB.
        if prev_word in _OPTATIVE_ADV:
            score += 6

    return score


def score_adj(words: list, idx: int) -> int:
    prev_word = _prev(words, idx)
    prev2_word = _prev2(words, idx)
    next_word = _next(words, idx)

    score = 0

    # Predicative position: copula directly before the adjective.
    if prev_word in _COPULA:
        score += 5

    # Comparative/superlative: "mais/menos" before — strong ADJ signal.
    if prev_word in _COMPARATIVES:
        score += 4

    # Degree intensifier before: "muito seco", "completamente seco".
    if prev_word in _INTENSIFIERS:
        score += 2
    # Degree adverb ending in -mente: "particularmente seco", "especialmente seco".
    if prev_word.endswith("mente"):
        score += 4
    # Copula + degree adverb: "foi particularmente seco" — strong predicative ADJ.
    if prev2_word in _COPULA and prev_word.endswith("mente"):
        score += 3

    # Attributive position: "DET NOUN ADJ" or "Que NOUN ADJ" pattern.
    # _EXCL_DET ("que") is included here only: "Que ideia tola" → ADJ.
    # QUANT included: "nenhum produto seco", "algum produto seco" → ADJ.
    # Guard: if prev is empty or the raw prev token ends in punctuation (comma =
    # clause boundary), the DET-NOUN is in a different clause from the adjective
    # ("Depois do banho, seco").
    _prev_raw_adj = words[idx - 1] if idx > 0 else ""
    _prev_across_boundary = (prev_word == ""
                              or (_prev_raw_adj and _prev_raw_adj[-1] in ".,;:!?"))
    if prev2_word in DET | QUANT | _EXCL_DET \
            and prev_word not in DET | QUANT | _EXCL_DET \
            and not _prev_across_boundary:
        score += 4

    # Bare DET/QUANT immediately before (determiner phrase head, less common for ADJ).
    if prev_word in DET | QUANT | PRON:
        score += 3

    # Post-positive attributive position: "NOUN ADJ" (bare noun immediately before,
    # no clause boundary between them).
    # Portuguese freely places adjectives after nouns.  If prev is a non-empty content
    # word that is NOT in any function-word set (DET/QUANT/PRON/AUX/connective) and
    # is NOT separated from the adjective by punctuation, it is likely a noun head.
    # Guards: prev not empty, len > 2 (exclude single-char preps), no trailing
    # punctuation on the raw token (comma/period signals clause boundary), and prev
    # does not look like a finite verb form (common past-tense endings).
    _prev_raw = words[idx - 1] if idx > 0 else ""
    _FUNC_WORDS = DET | QUANT | PRON | AUX_VERBS | _FUNC_EXTRA
    # Also suppress when next is a DET/article — "seco os pratos" is VERB+DO,
    # not an ADJ with a following object.
    # The "-ia" suffix is a verb imperfect ending only when preceded by a consonant
    # (e.g. "comia", "dormia"); words like "areia", "galeria" end in vowel+"ia" and
    # are nouns — do not suppress the postpositive signal for them.
    # Also: verb+clitic forms like "chamaram-lhe", "disse-me" contain a hyphen followed
    # by a clitic pronoun — clearly a verb, not a noun head.
    _vowels = set("aeiouáéíóúâêîôûãõàèìòùäëïöü")
    _CLITICS = _CLITICS_ALL
    _prev_looks_verb = (
        prev_word.endswith(("ou", "eu", "iu", "ei", "ava", "ara", "era"))
        or (prev_word.endswith("ia") and len(prev_word) >= 4
            and prev_word[-3] not in _vowels)
        or ("-" in prev_word and prev_word.rsplit("-", 1)[-1] in _CLITICS)
    )
    if (prev_word and len(prev_word) > 2
            and not _prev_raw[-1] in ".,;:!?"
            and prev_word not in _FUNC_WORDS
            and next_word not in DET | QUANT
            and not _prev_looks_verb):
        score += 3

    return score


def guess_pos(words: list, idx: int) -> str:
    """Return the most likely UDEP POS tag for the ambiguous word at *idx*."""
    word = words[idx]

    # Seed each candidate POS with its corpus-frequency prior (BASE_SCORE).
    # Values are small (1–3) so any explicit context signal overrides them.
    _bias = BASE_SCORE.get(word, {})

    scores = {}
    if word in ADP_IPA:
        scores["ADP"] = score_adp(words, idx) + _bias.get("ADP", 0)
    if word in NOUNS_IPA:
        scores["NOUN"] = score_noun(words, idx) + _bias.get("NOUN", 0)
    if word in VERBS_IPA:
        scores["VERB"] = score_verb(words, idx) + _bias.get("VERB", 0)
    if word in ADJ_IPA:
        scores["ADJ"] = score_adj(words, idx) + _bias.get("ADJ", 0)

    best_score = max(scores.values())
    if best_score <= 0:
        return DEFAULT_POS.get(word, "NOUN")

    # When multiple POS share the highest score, fall back to the frequency prior.
    winners = [pos for pos, s in scores.items() if s == best_score]
    if len(winners) == 1:
        return winners[0]
    return DEFAULT_POS.get(word, winners[0])


def _resolve_sede(words: list, idx: int) -> str:
    """Disambiguate "sede" between SEAT (HQ, open ɛ) and THIRST (closed e).

    Both senses are nouns, so POS scoring cannot separate them — this reads
    meaning cues from the local context instead. The preposition frame is the
    strongest signal ("sede de X" → thirst, "sede da/do X" → seat), reinforced
    by the sede_{seat,thirst}_cues.voc wordlists in the ±3 window.
    """
    prev_word = _prev(words, idx)
    next_word = _next(words, idx)
    next2 = _strip(words[idx + 2]) if idx + 2 < len(words) else ""

    seat = thirst = 0
    # preposition frame
    if next_word in {"da", "do", "das", "dos"}:
        seat += 3
    if next_word == "de":                      # "sede de <abstract>" = figurative thirst
        thirst += 2
        if next2 in _SEDE_THIRST:
            thirst += 3
        if next2 in _SEDE_SEAT:                # "sede de futebol clube" etc.
            seat += 3
    if prev_word == "de":                       # "morto de sede", "queixou-se de sede"
        thirst += 3
    if prev_word in {"na", "à", "numa", "pela", "duma"}:   # locative: the HQ building
        seat += 2
    # content cues in the window
    window = [_strip(words[i]) for i in range(max(0, idx - 3), min(len(words), idx + 4))
              if i != idx]
    seat += sum(w in _SEDE_SEAT for w in window)
    thirst += sum(w in _SEDE_THIRST for w in window)

    if seat > thirst:
        return "seat"
    if thirst > seat:
        return "thirst"
    return DEFAULT_SENSE.get("sede", "thirst")


# words whose senses share a POS need a meaning-level resolver after guess_pos
_SENSE_RESOLVERS = {"sede": _resolve_sede}


def resolve_sense(word: str, words: list, idx: int, pos: str) -> str:
    """Return the meaning slug for *word* at *idx*, given its guessed *pos*.

    For the common case (each POS maps to exactly one sense) this is a direct
    lookup; only words with two senses under one POS invoke a context resolver.
    """
    senses = POS_SENSES.get(word, {}).get(pos)
    if not senses:
        # guessed POS has no sense for this word → fall back to its default POS
        senses = POS_SENSES.get(word, {}).get(DEFAULT_POS.get(word, ""))
    if not senses:
        # last resort: any sense
        senses = next(iter(POS_SENSES.get(word, {}).values()), [None])
    if len(senses) == 1:
        return senses[0]
    resolver = _SENSE_RESOLVERS.get(word)
    return resolver(words, idx) if resolver else senses[0]

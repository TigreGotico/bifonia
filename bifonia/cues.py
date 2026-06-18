"""Declarative sense-cue registry — the single source of truth for the lexical
cues that disambiguate same-spelling readings of a heterophone.

Two consumers read this module, which is precisely *why* the cues live here as
one table instead of being inlined in either:

* the **rule engine** — :func:`bifonia.scoring._resolve_by_cues` picks the sense
  whose cue list scores highest in context;
* the **model feature extractor** — :func:`bifonia.features.extract_features`
  emits a cue-score feature per sense, so a trained model sees the *same*
  evidence the rules use instead of having to rediscover these lexicons from raw
  bag-of-words counts.

Each cue references a ``<name>.voc`` wordlist under ``locale/<lang>/`` (one term
per line, ``#`` comments).  Adding a new same-POS / diacritic-collapse word is
therefore **data-only**: drop the ``.voc`` file(s) and add one entry here — no
new resolver code.
"""
from __future__ import annotations

from typing import NamedTuple, Optional


class Cue(NamedTuple):
    """One scored cue list voting for a sense.

    Attributes:
        sense: the meaning slug this cue list votes for.
        voc:   the ``.voc`` wordlist name (under ``locale/<lang>/``) of cues.
        requires_prev: if set, the cue only counts when the token immediately
            before the homograph equals this string — e.g. "de cor" → by-heart
            needs a preceding "de"; bare "cor" stays colour.
    """
    sense: str
    voc: str
    requires_prev: Optional[str] = None


class CueRule(NamedTuple):
    """A word's complete cue-resolution rule.

    Attributes:
        default: the sense returned when no cue scores (the dominant reading).
        cues:    the cue lists to weigh; selection is by score, so order is
                 irrelevant.
    """
    default: str
    cues: tuple[Cue, ...]


# ── words resolved purely by cue competition (read by _resolve_by_cues) ────────
# For each, the resolver scores every applicable Cue with the proximity-weighted
# cue_score() and returns the winner, or `default` when nothing scores.
SENSE_CUES: dict[str, CueRule] = {
    # ball (open ɔ, ˈbɔlɐ) vs the «bôla» bread/cake (closed o, ˈbolɐ):
    # baking/charcuterie cues vs sport/play cues — a football sentence with one
    # stray food word still scores ball higher and stays ball.
    "bola": CueRule("ball", (Cue("loaf", "bola_loaf_cues"),
                             Cue("ball", "bola_ball_cues"))),
    # wolf (closed o, ˈlobu) vs anatomical lobe (open ɔ, ˈlɔbu): anatomy cues, else wolf.
    "lobo": CueRule("wolf", (Cue("lobe", "lobo_lobe_cues"),)),
    # pole / polo-sport (open ɔ, ˈpɔlu) vs the rare fledgling bird (closed o, ˈpolu).
    "polo": CueRule("pole", (Cue("fledgling", "polo_fledgling_cues"),)),
    # colour (closed o, ˈkoɾ) vs the fixed adverbial "de cor" = by heart (open ɔ, ˈkɔɾ).
    "cor":  CueRule("colour", (Cue("by_heart", "cor_memory_cues", requires_prev="de"),)),
}


# ── every sense-cue voc, for model features ───────────────────────────────────
# A superset of the cues above: it also lists the cue vocs of words whose *rule*
# resolution is structurally richer than a cue competition (sede, molho) and so
# stays hand-written — but whose cues are still valuable as model features.
# Maps voc name → the (word, sense) it signals.
FEATURE_CUES: dict[str, tuple[str, str]] = {
    "bola_loaf_cues":      ("bola", "loaf"),
    "bola_ball_cues":      ("bola", "ball"),
    "lobo_lobe_cues":      ("lobo", "lobe"),
    "polo_fledgling_cues": ("polo", "fledgling"),
    "cor_memory_cues":     ("cor", "by_heart"),
    "sede_seat_cues":      ("sede", "seat"),
    "sede_thirst_cues":    ("sede", "thirst"),
    "bundle_things":       ("molho", "bundle"),
    "soak_verbs":          ("molho", "soak"),
    "cut_context":         ("corte", "cut"),
}

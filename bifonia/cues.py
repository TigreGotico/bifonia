"""The lexical cues that tell apart the same-spelling readings of a heterophone.

The rule engine (:func:`bifonia.scoring._resolve_by_cues`) and the model
features (:func:`bifonia.features.extract_features`) both read this table, so
the model sees the same evidence as the rules. Each cue names a ``<name>.voc``
wordlist under ``locale/<lang>/``. To add a same-POS word, add its ``.voc``
files and one SENSE_CUES entry; no resolver code is needed.
"""
from __future__ import annotations

from typing import NamedTuple, Optional


class Cue(NamedTuple):
    """A cue wordlist (*voc*) that votes for *sense*. With *requires_prev* set it
    counts only when the token before the homograph is that word: "de cor" is
    by heart, a bare "cor" stays colour."""
    sense: str
    voc: str
    requires_prev: Optional[str] = None


class CueRule(NamedTuple):
    """A word's cue rule: the highest-scoring of *cues* wins (their order does
    not matter), and *default*, the dominant reading, wins when none scores."""
    default: str
    cues: tuple[Cue, ...]


# Words resolved by cue competition alone.
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
    # shape (open ɔ, ˈfɔɾmɐ) vs the «fôrma» baking-tin/mould (closed o, ˈformɐ):
    # baking/casting cues, else the abstract shape reading.
    "forma": CueRule("shape", (Cue("mould", "mould_cues"),)),
}


# Every sense-cue voc → the (word, sense) it signals, for model features. It adds
# cue lists the rules read outside SENSE_CUES (sede, molho, corte).
FEATURE_CUES: dict[str, tuple[str, str]] = {
    "bola_loaf_cues":      ("bola", "loaf"),
    "bola_ball_cues":      ("bola", "ball"),
    "lobo_lobe_cues":      ("lobo", "lobe"),
    "polo_fledgling_cues": ("polo", "fledgling"),
    "cor_memory_cues":     ("cor", "by_heart"),
    "mould_cues":          ("forma", "mould"),
    "sede_seat_cues":      ("sede", "seat"),
    "sede_thirst_cues":    ("sede", "thirst"),
    "bundle_things":       ("molho", "bundle"),
    "soak_verbs":          ("molho", "soak"),
    "cut_context":         ("corte", "cut"),
}

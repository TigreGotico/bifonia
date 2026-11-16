"""Diacritized input (pre-AO1990 or add_extra_diacritics output) resolves to
a sense its diacritic allows."""
import json
import re
from pathlib import Path

import pytest

from bifonia import tokenize, guess_sense, guess_pos, disambiguate, HOMOGRAPHS

_CASES = [
    c for c in (
        json.loads(line) for line in
        (Path(__file__).parent / "test_sentences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if c["word"] == "choco" and not c["xfail"]
]
# Closed o (ô) is shared by the adjective and the noun; open o (ó) is the verb.
_FORM = {"addled": "ô", "cuttlefish": "ô", "hatch": "ó"}
_ALLOWED = {"ô": {"addled", "cuttlefish"}, "ó": {"hatch"}}


def _target(words):
    return next(i for i, t in enumerate(words) if t.strip(".,;:!?").startswith("ch"))


def test_choco_cases_cover_every_sense():
    assert {c["sense"] for c in _CASES} == {"addled", "cuttlefish", "hatch"}


@pytest.mark.parametrize("case", _CASES, ids=[c["sentence"][:40] for c in _CASES])
def test_diacritized_choco_agrees_with_plain_form(case):
    mark = _FORM[case["sense"]]
    plain = tokenize(case["sentence"])
    marked = tokenize(re.sub(r"\b([Cc])hoco\b", lambda m: f"{m.group(1)}h{mark}co",
                             case["sentence"]))
    idx = _target(plain)
    assert marked[idx].startswith(f"ch{mark}co")
    # The closed mark does not tell addled from cuttlefish, so it must leave
    # the context reading of the plain word in place.
    assert guess_sense(plain, idx) in _ALLOWED[mark]
    assert guess_sense(marked, idx) == guess_sense(plain, idx)
    assert disambiguate(marked, idx) == HOMOGRAPHS["choco"][case["sense"]]


def test_spoiled_milk_reads_addled():
    words = tokenize("O leite está chôco, não o bebas.")
    assert guess_sense(words, 3) == "addled"
    assert guess_pos(words, 3) == "ADJ"


def test_closed_diacritic_overrides_verb_context():
    # "Chôco ovos" reads as a verb, but ô marks a closed (non-verb) reading.
    words = tokenize("Chôco ovos de galinha na incubadora.")
    assert guess_sense(words, 0) in _ALLOWED["ô"]
    assert guess_pos(words, 0) in {"ADJ", "NOUN"}

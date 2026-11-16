"""Loader for the bundled infopedia-sourced IPA dataset (`data/infopedia_pt.csv`).

The dataset is built by ``scripts/rebuild_infopedia.py`` and ships full
European-Portuguese dictionary data — IPA, POS, syllabification, etymology,
definitions — one row per (word, sense). The ``pos`` column holds the same
UD tags as the rest of bifonia (``NOUN``, ``VERB``, ``ADJ``, ``ADP``), joined
with ``|`` when one reading has two (``NOUN|VERB``). Heterophones carry one row
per reading (e.g. ``colher`` → ``kuˈʎɛɾ`` NOUN / ``kuˈʎeɾ`` VERB).

Pure standard library; safe on the zero-dependency inference path.
"""
import csv
import functools
import pathlib

_CSV = pathlib.Path(__file__).parent / "data" / "infopedia_pt.csv"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """Return ``word -> [row]``, each row a dict of the CSV columns (sense, pos, ipa, …)."""
    out: dict = {}
    if not _CSV.exists():
        return out
    with _CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["word"], []).append(dict(row))
    return out


def readings(word: str) -> list:
    """All infopedia readings (rows) for *word*."""
    return load().get(word.lower(), [])


def ipa(word: str, pos: str = None) -> str | None:
    """IPA for *word*; if *pos* is given (a UD tag such as ``"NOUN"`` or
    ``"VERB"``), prefer a reading tagged with it. A reading with that tag alone
    wins over one with a combined tag (``tola``: ``NOUN`` head over ``ADJ|NOUN``
    foolish). When no reading has the tag, the first reading is returned.
    Returns None if the word is absent."""
    rows = readings(word)
    if pos:
        rows = ([r for r in rows if r["pos"] == pos]
                or [r for r in rows if pos in r["pos"].split("|")]
                or rows)
    return rows[0]["ipa"] if rows else None

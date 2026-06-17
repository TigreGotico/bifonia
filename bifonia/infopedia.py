"""Loader for the bundled infopedia-sourced IPA dataset (`data/infopedia_pt.csv`).

The dataset is built by ``pyinfopedia.dataset.export_csv`` and ships full
European-Portuguese dictionary data — IPA, POS, syllabification, etymology,
definitions, TTS audio URL — one row per (word, reading, POS). Heterophones
carry one row per reading (e.g. ``colher`` → ``kuˈʎɛr`` nome / ``kuˈʎer`` verbo).

Pure standard library; safe on the zero-dependency inference path.
"""
import csv
import functools
import pathlib

_CSV = pathlib.Path(__file__).parent / "data" / "infopedia_pt.csv"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """Return ``word -> [ {ipa, pos, syllabification, etymology, definition, audio_url} ]``."""
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
    """IPA for *word*; if *pos* is given (e.g. ``"verbo"``, ``"nome"``), prefer
    the reading whose POS starts with it. Returns None if the word is absent."""
    rows = readings(word)
    if pos:
        rows = [r for r in rows if r["pos"].startswith(pos)] or rows
    return rows[0]["ipa"] if rows else None

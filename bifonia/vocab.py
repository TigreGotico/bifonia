"""
Locale-aware vocabulary loader for bifonia.

Wordlists live as ``.voc`` files under ``bifonia/locale/<lang>/``: one term per
line, ``#`` comments and blank lines ignored.
"""
import pathlib
from functools import lru_cache

from linguatag import closest_lang

_LOCALE_ROOT = pathlib.Path(__file__).parent / "locale"
DEFAULT_LANG = "pt-PT"
_LOCALES = sorted(d.name for d in _LOCALE_ROOT.iterdir() if d.is_dir())


def _lang_dir(lang: str) -> pathlib.Path:
    """Resolve a language code to its locale directory.

    ``pt-PT`` / ``pt_PT`` / ``pt`` all map to ``locale/pt-pt``: the directory closest
    to *lang* wins.
    """
    return _LOCALE_ROOT / (closest_lang(lang, _LOCALES) or lang)


@lru_cache(maxsize=None)
def voc(name: str, lang: str = DEFAULT_LANG) -> frozenset:
    """Return the terms in ``<locale>/<lang>/<name>.voc`` as a lowercased frozenset."""
    path = _lang_dir(lang) / f"{name}.voc"
    if not path.exists():
        raise FileNotFoundError(f"vocabulary file not found: {path}")
    return frozenset(
        t.strip().lower()
        for t in path.read_text(encoding="utf-8").splitlines()
        if t.strip() and not t.strip().startswith("#")
    )

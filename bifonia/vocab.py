"""
Locale-aware vocabulary loader for bifonia.

Wordlists live as ``.voc`` files under ``bifonia/locale/<lang>/`` — one term per
line, ``#`` comments and blank lines ignored — so they can be edited and extended
without touching code (data separated from logic). Pure standard library, no
dependencies.
"""
import pathlib
from functools import lru_cache

_LOCALE_ROOT = pathlib.Path(__file__).parent / "locale"
DEFAULT_LANG = "pt-PT"


def _lang_dir(lang: str) -> pathlib.Path:
    """Resolve a language code to its locale directory.

    ``pt-PT`` / ``pt_PT`` / ``pt`` all map to ``locale/pt-pt``; an exact match wins,
    otherwise the first directory sharing the primary subtag is used.
    """
    norm = lang.lower().replace("_", "-")
    if (_LOCALE_ROOT / norm).is_dir():
        return _LOCALE_ROOT / norm
    for d in sorted(_LOCALE_ROOT.glob("*")):
        if d.is_dir() and d.name.split("-")[0] == norm.split("-")[0]:
            return d
    return _LOCALE_ROOT / norm


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

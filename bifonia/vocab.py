"""
Locale-aware vocabulary loader for bifonia.

Wordlists live as OVOS ``.voc`` files under ``bifonia/locale/<lang>/`` — one
term per line, ``#`` comments and blank lines ignored — so they can be edited
and extended without touching code (separation of data from logic).

Locale resolution and the resource reader come from ``ovos_spec_tools`` (the
org-standard for lang/locale primitives) when it is installed, so ``pt-PT``/``pt``
resolve to the ``pt-pt`` directory. When it is absent, a stdlib fallback keeps the
library fully dependency-free (``pip install --no-deps``).
"""

import pathlib
from functools import lru_cache

try:
    from ovos_spec_tools import find_lang_dir, read_resource_file, standardize_lang
    _HAVE_OVOS = True
except ImportError:  # zero-dependency fallback (stdlib only)
    _HAVE_OVOS = False

_LOCALE_ROOT = pathlib.Path(__file__).parent / "locale"
DEFAULT_LANG = "pt-PT"


def _lang_dir(lang: str) -> pathlib.Path:
    if _HAVE_OVOS:
        d = find_lang_dir(_LOCALE_ROOT, lang)
        return d if d is not None else _LOCALE_ROOT / standardize_lang(lang).lower()
    # stdlib: normalise "pt-PT"/"pt_PT"/"pt" → "pt-pt" with a prefix fallback
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
    if _HAVE_OVOS:
        lines = read_resource_file(path)
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(t.strip().lower() for t in lines
                     if t.strip() and not t.strip().startswith("#"))

"""
Locale-aware vocabulary loader for bifonia.

Wordlists live as OVOS ``.voc`` files under ``bifonia/locale/<lang>/`` — one
term per line, ``#`` comments and blank lines ignored — so they can be edited
and extended without touching code (separation of data from logic).

Locale resolution and the resource reader come from ``ovos_spec_tools`` (the
org-standard for lang/locale primitives), so ``pt-PT``/``pt`` resolve to the
``pt-pt`` directory and fall back gracefully.
"""

import pathlib
from functools import lru_cache

from ovos_spec_tools import find_lang_dir, read_resource_file, standardize_lang

_LOCALE_ROOT = pathlib.Path(__file__).parent / "locale"
DEFAULT_LANG = "pt-PT"


@lru_cache(maxsize=None)
def voc(name: str, lang: str = DEFAULT_LANG) -> frozenset:
    """Return the terms in ``<locale>/<lang>/<name>.voc`` as a lowercased frozenset."""
    lang_dir = find_lang_dir(_LOCALE_ROOT, lang)
    if lang_dir is None:
        lang_dir = _LOCALE_ROOT / standardize_lang(lang).lower()
    path = lang_dir / f"{name}.voc"
    if not path.exists():
        raise FileNotFoundError(f"vocabulary file not found: {path}")
    return frozenset(t.strip().lower() for t in read_resource_file(path) if t.strip())

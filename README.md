# homogr

Rule-based Portuguese heterophonic homograph disambiguation.

Identifies the correct IPA pronunciation of words whose orthography is identical but
whose phonology depends on their part of speech (NOUN vs VERB vs ADP/ADJ).

```python
from homogr import tokenize, is_ambiguous, disambiguate

words = tokenize("O autocarro para em frente ao hospital.")
for i, w in enumerate(words):
    if is_ambiguous(w):
        print(w, "→", disambiguate(words, i))
# para → ˈpaɾɐ  (VERB reading — stops)
```

## How it works

Context scoring assigns non-negative points to each applicable POS based on the
surrounding tokens (determiners, pronouns, auxiliaries, numerals, infinitives).
The highest-scoring POS wins; ties fall back to a linguistic default.

IPA data is sourced from [tugalex](../tugalex)'s curated `heterophonic_homographs.csv`,
covering 27 words across the NOUN/VERB/ADP/ADJ distinction.

## Install

```bash
pip install -e . --no-deps
```

## See also

- `docs/usage.md` — full API reference and scoring details
- `examples/basic_usage.py` — runnable demo
- `tests/` — pytest suite, one sentence per POS reading per word

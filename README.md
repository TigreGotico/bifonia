# bifonia

<!-- agent-hint
covers: what bifonia does, how to install it and where the documentation is
source: `bifonia/__init__.py`, `bifonia/data/`
generated: no
verify: diff <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' README.md | grep -vP '^\x60{3}' | python -I -) <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' README.md | grep -oP '(print\(.*\)\s+|^)# \K.*')
see: `docs/index.md`, `docs/usage.md`
-->

bifonia picks the reading of a European-Portuguese heterophonic homograph from its sentence. These are words spelled the same way whose pronunciation depends on meaning, such as `sede` (*thirst* `ˈsedɨ` or *headquarters* `ˈsɛdɨ`). It returns the meaning, the IPA, or the sentence with a disambiguating diacritic, for text-to-speech front ends. Its one runtime dependency is linguatag, for language tags.

## Install

```bash
uv pip install --prerelease=allow bifonia
```

## Quickstart

```python
from bifonia import tokenize, guess_sense, disambiguate

words = tokenize("Tinha tanta sede que bebi a garrafa toda.")
i = words.index("sede")
print(guess_sense(words, i))     # thirst
print(disambiguate(words, i))    # ˈsedɨ

words = tokenize("A sede da empresa fica em Lisboa.")
i = words.index("sede")
print(disambiguate(words, i))    # ˈsɛdɨ
```

## Documentation

| Page | Contents |
|---|---|
| [docs/usage.md](docs/usage.md) | the full API, the scoring signals and word coverage |
| [docs/methodology.md](docs/methodology.md) | the two engines, the architecture and the corpus build |
| [docs/benchmarks.md](docs/benchmarks.md) | accuracy of every approach, and which numbers a command reproduces |
| [docs/data_quality.md](docs/data_quality.md) | how the corpus labels are checked |
| [docs/dataset.md](docs/dataset.md) | the corpus schema, counts, splits and Hub datasets |
| [docs/diacritics_restoration.md](docs/diacritics_restoration.md) | the task framed as diacritics restoration |
| [docs/context_free_marks.md](docs/context_free_marks.md) | marks that need no context, such as *pêra* |
| [docs/lects.md](docs/lects.md) | Brazilian, Uruguayan, African and Asian Portuguese |
| [docs/word_references.md](docs/word_references.md) | the IPA, senses and diacritized forms of each word |

[`docs/index.md`](docs/index.md) gives the reading order and the related projects. [`examples/`](examples) holds runnable scripts. [`AGENTS.md`](AGENTS.md) gives the repository map and the test command.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

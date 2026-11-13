# AGENTS.md

bifonia picks the reading of a European-Portuguese heterophonic homograph from its sentence context. It returns the meaning slug, the IPA, or the sentence with a disambiguating diacritic. It has no runtime dependencies.

## Repository map

| path | contents |
|---|---|
| `bifonia/__init__.py` | public API: `tokenize`, `is_ambiguous`, `guess_sense`, `guess_pos`, `disambiguate`, `add_extra_diacritics`, `proper_flags` |
| `bifonia/scoring.py` | the rule engine: per-POS context scores and the sense resolvers |
| `bifonia/cues.py` | the `SENSE_CUES` registry that the rules and the model features share |
| `bifonia/features.py`, `bifonia/model.py` | features and inference for the learned NB / perceptron models |
| `bifonia/data/` | `heterophonic_homographs.csv` (roster and IPA), `corpus.jsonl`, `lect_overrides.csv`, `context_free_marks.csv`, model JSON |
| `bifonia/locale/pt-pt/*.voc` | context and meaning wordlists |
| `scripts/` | `train.py`, `dataset.py`, `benchmark_tagger.py`, `benchmark_ood.py`, `baselines.py`, `build_context_free_marks.py` |
| `hf/` | the train and test split published as `TigreGotico/bifonia-pt-homographs` |
| `tests/` | pytest suite. `test_sentences.jsonl` is also the behavioural set for the route gate |
| `docs/` | reference pages. [README.md](README.md#documentation) lists them in reading order |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow -e ".[test]"
```

## Test

```bash
.venv/bin/python -m pytest tests
```

CI runs the same suite on Python 3.11 to 3.14 (`.github/workflows/build-tests.yml`).

## Rules the repository docs state

- Keep the runtime dependency list empty. The package uses only the Python standard library (`pyproject.toml`, README). Evaluation code that needs numpy, spaCy or Hugging Face stays in `scripts/`.
- Take pronunciations from Infopédia (README, `docs/word_references.md`). `bifonia/data/infopedia_pt.csv` holds the extracted entries. `scripts/rebuild_infopedia.py` builds that file. It needs the `pyinfopedia` client and a FlareSolverr service (`--flaresolverr URL`), because Infopédia refuses plain scripted requests.
- `bifonia/data/corpus.jsonl` is the single source of truth for the corpus (`docs/dataset.md`). Regenerate `hf/` with `python scripts/dataset.py --hf` after a corpus change.
- `scripts/train.py` trains only on `hf/train.jsonl`, never on the test split (`docs/methodology.md`).
- Give every row of `bifonia/data/lect_overrides.csv` a source in its `source` column (README, Brazilian Portuguese).
- Build `bifonia/data/context_free_marks.csv` with `scripts/build_context_free_marks.py`, never by hand. A row needs a norm source, a transcription of each lect it names and an orthography2ipa gain for that lect (README, Marks that need no context).
- To add a same-POS word to the cue resolver, add its `.voc` wordlists and one `SENSE_CUES` entry in `bifonia/cues.py` (`bifonia/cues.py` docstring).
- The release workflow sets `bifonia/version.py` from the conventional-commit titles of merged pull requests (`.github/workflows/release_workflow.yml`).

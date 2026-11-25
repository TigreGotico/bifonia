# bifonia documentation

<!-- agent-hint
covers: the reading order of every documentation page
source: `docs/`
generated: no
verify: diff <(ls docs/*.md | sed 's|^docs/||' | grep -v '^index.md$' | sort) <(grep -oP '\]\(\K[^)#]+\.md(?=\))' docs/index.md | grep -v -e '^http' -e '^\.\./' | sort)
see: `README.md`, `AGENTS.md`
-->

bifonia picks the reading of a European-Portuguese heterophonic homograph from its sentence. The [README](../README.md) shows the install and a first call. These pages hold the detail.

Read the pages in this order:

1. [Usage](usage.md): the full API and the scoring signals.
2. [Methodology](methodology.md): how the corpus is built and how the two engines decide.
3. [Benchmarks](benchmarks.md): every accuracy number, and which ones a command reproduces.
4. [Data quality](data_quality.md): how the corpus labels are checked.
5. [Dataset](dataset.md): the corpus schema, counts and splits.
6. [Diacritics restoration](diacritics_restoration.md): the task framed as diacritics restoration.
7. [Marks that need no context](context_free_marks.md): the marks that a lookup restores, and the checks a row passes.
8. [Lects](lects.md): Brazilian, Uruguayan, African and Asian Portuguese.
9. [Word references](word_references.md): the IPA, senses and diacritized forms of each word.

The runnable scripts are in [examples/](../examples/).

## Related projects

- [`TigreGotico/tugaphone`](https://github.com/TigreGotico/tugaphone): a Portuguese G2P for several Lusophone lects. It depends on bifonia and runs `add_extra_diacritics` before it phonemizes.
- [`TigreGotico/orthography2ipa`](https://github.com/TigreGotico/orthography2ipa): the grapheme-to-phoneme engine that tugaphone runs. `scripts/build_context_free_marks.py` checks each context-free mark against it.
- [`TigreGotico/o2i-lexicons`](https://github.com/TigreGotico/o2i-lexicons): the `pt-PT` and `pt-BR` residual lexicons of orthography2ipa. Its NILC entries are a pt-BR source for the context-free marks.
- [`TigreGotico/tugalex`](https://github.com/TigreGotico/tugalex): a Portuguese pronunciation lexicon. Its AO1990 maps are the norm source of the context-free marks.
- [`TigreGotico/silabificador`](https://github.com/TigreGotico/silabificador): a rule-based Portuguese syllabifier.

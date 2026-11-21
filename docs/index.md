# bifonia documentation

<!-- agent-hint
covers: the reading order of every documentation page
source: `docs/`
generated: no
verify: diff <(ls docs/*.md | sed 's|^docs/||' | grep -v '^index.md$' | sort) <(grep -oP '\]\(\K[^)#]+\.md(?=\))' docs/index.md | grep -v -e '^http' -e '^\.\./' | sort)
see: `README.md`, `AGENTS.md`
-->

bifonia picks the reading of a European-Portuguese heterophonic homograph from its sentence. The [README](../README.md) shows the install, the first calls and the accuracy. These pages hold the detail.

Read the pages in this order:

1. [Usage](usage.md): the full API and the scoring signals.
2. [Methodology](methodology.md): how the corpus is built and how the two engines decide.
3. [Benchmarks](benchmarks.md): every accuracy number, and which ones a command reproduces.
4. [Data quality](data_quality.md): how the corpus labels are checked.
5. [Dataset](dataset.md): the corpus schema, counts and splits.
6. [Diacritics restoration](diacritics_restoration.md): the task framed as diacritics restoration.
7. [Word references](word_references.md): the IPA, senses and diacritized forms of each word.

The runnable scripts are in [examples/](../examples/).

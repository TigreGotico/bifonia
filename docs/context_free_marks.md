# Marks that need no context

<!-- agent-hint
covers: the table of context-free diacritic marks, the three checks a row passes and the marks that stay out
source: `bifonia/data/context_free_marks.csv`, `scripts/build_context_free_marks.py`, `bifonia/__init__.py:add_extra_diacritics`
generated: no
verify: diff <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' docs/context_free_marks.md | grep -vP '^\x60{3}' | python -I -) <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' docs/context_free_marks.md | grep -oP '(print\(.*\)\s+|^)# \K.*') && python -m pytest -q tests/test_context_free_marks.py
see: `docs/diacritics_restoration.md`, `docs/lects.md`
-->

Some words carry a mark that an older spelling norm wrote and that AO1990 dropped, such as *pêra*, *assembléia*, *heróico* and *seqüela*. After the homographs, `add_extra_diacritics` writes these marks from `bifonia/data/context_free_marks.csv`. A word enters this table only when its sources show one reading of the bare form in the lect, so the mark does not need context.

```python
from bifonia import add_extra_diacritics

print(add_extra_diacritics("Comi uma pera."))                      # Comi uma pêra.
print(add_extra_diacritics("A assembleia votou.", lang="pt-BR"))   # A assembléia votou.
print(add_extra_diacritics("A assembleia votou."))                 # A assembleia votou.
print(add_extra_diacritics("O dia subsequente.", lang="pt-BR"))        # O dia subseqüente.
print(add_extra_diacritics("O dia subsequente.", lang="pt-BR-x-sp"))   # O dia subsequente.
```

A word enters the table for a lect code only when it passes three checks. The CSV records the evidence for each check.

1. **A cited norm wrote the mark.** The word comes from a row of the tugalex AO1990 maps whose old form differs from the new form only by a diacritic. The `norm_source` column names the map row. Where a clause applies, it names the clause too:
   AO1990 Base IX 3.º for ⟨éi ói⟩, Base IX 8.º for ⟨ôo⟩, Base X 4.º for ⟨í ú⟩ after a diphthong and Base XIV 1.º for the trema.
   For *pêra* it is the Acordo de 1945 Base XXII.
2. **The lect says the vowel of the mark, in every reading of the bare form.** For pt-PT the source is Infopédia. For pt-BR it is Michaelis, Wiktionary and the NILC entries of o2i-lexicons. A source entry for a capitalised headword covers the proper noun only, so it does not count. A word with a second reading of another vowel stays out, unless a cited source marks that reading archaic, as for *pera*. So pt-BR leaves *ateia* unmarked: pt.wiktionary gives the noun /ɐˈtɛj.ɐ/ and the verb forms of *atear* /ɐˈtej.ɐ/. A pt-BR ⟨-eia⟩ word that is also a form of an ⟨-ear⟩ verb stays out too, because the verb form is a second reading (`EAR_VERB_RULE` in `scripts/build_context_free_marks.py` gives the sources). This keeps out *ideia* (*idear*), *estreia* (*estrear*) and *traqueia* (*traquear*) by analogy with *ateia*, the one word whose sources show both vowels; it leaves their output as it was. The `pronunciation_source` and `transcription` columns give the readings.
3. **orthography2ipa reads the mark and gets closer to the source.** The check runs for each pt-PT and pt-BR code of orthography2ipa. For the code, `G2P(code).transcribe_word` must give the source vowel for the marked form and another vowel for the unmarked form. The marked output must also be no further from the source than the unmarked output. For pt-PT and pt-BR, the same must hold with the packaged o2i-lexicons lexicon, which tugaphone loads. The sources describe pt-PT and pt-BR only. So a sub-lect code gets a row only when its parent gets the row from the plain engine.

The table has 134 words. pt-PT has 88, its sub-lects 78 to 90, pt-BR 55, and the pt-BR sub-lects 40 or 55. A lect gets a mark even when its own norm never wrote it. pt-PT gets *vôo* and *enjôo* (Infopédia [ˈvou], [ẽˈʒou]) and *combóio* (Infopédia [kõˈbɔju]). A lect gets no mark when it says a different vowel. pt-PT *assembleia* is [ɐsẽˈblɐjɐ], so *assembléia* is pt-BR only. Michaelis gives Brazilian *areia* as [aˈɾejɐ]. So pt-BR keeps *areia*, although the Brazilian norm wrote *aréia*.

The `lects` column lists the orthography2ipa codes where the row passed. A tag takes the rows of the longest code in the table that is the tag or its parent. So `pt-BR-x-sp` takes its own rows, and `pt-BR-x-foo` takes the rows of `pt-BR`. The column also lists the orthography2ipa aliases of each code under its parent (`pt-BR-x-sao-paulo` for `pt-BR-x-sp`). An alias that leaves its parent, such as `pt-PT-x-barrancos` for `ext-PT-x-barranco`, would take the parent's rows, so pass `orthography2ipa.resolve(tag)` when the tag can be such an alias. `None` and a bare `pt` mean `pt-PT`. Other tags, `pt-AO` and `pt-UY` included, get none of these marks, because no transcription of those lects was checked. The homograph table comes first. Its words, such as *para*, *pelo* and *polo*, never enter this table, and the loader rejects such a row. These marks are not in the table:

- The trema of a word whose bare form orthography2ipa already reads with the glide, such as *linguiça* [lĩˈɡwisɐ] and *tranquilo*. Check 3 drops such a word, because the mark does not change the output.
- The trema for 11 of the 13 pt-BR sub-lects. In the pt-BR sub-lects other than `pt-BR-x-cuiaba` and `pt-BR-x-floripa`, orthography2ipa reads ⟨ü⟩ as a full vowel (`pt-BR-x-sp` *tranqüilo* gives [tɾɐ̃kuˈilu]).
- *taoísmo* for pt-PT. The packaged lexicon reads *taoismo* as [tawˈiʒmu], the Infopédia reading, and the mark makes the lookup miss.
- *péla*. Its unmarked homograph *pela*, the contraction of *per* and *la*, is a living word (AO1990 Base IX 9.º).
- Compounds with a hyphen, such as *pára-quedas*, because `add_extra_diacritics` marks single words.
- Plurals and other inflections that the maps do not list, such as *jóias*.
- The grave accent of the 1945 norm (*sòmente*) and the silent consonants (*acção*). The maps hold no grave rows, and a consonant is a letter, not a mark.

`scripts/build_context_free_marks.py` builds the CSV. It reads the tugalex maps, the [`TigreGotico/portuguese-unified-pronunciation-lexicon`](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon) dataset, o2i-lexicons and `bifonia/data/infopedia_pt.csv`, and it runs orthography2ipa. The script docstring gives the command. Do not edit the CSV by hand.

---
[← Diacritics restoration](diacritics_restoration.md) · [Index](index.md) · [Lects →](lects.md)

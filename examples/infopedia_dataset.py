"""Use the bundled Infopédia-sourced dataset for IPA / definitions per reading.

`bifonia/data/infopedia_pt.csv` ships full European-Portuguese dictionary data
(one row per reading: IPA, POS, definitions, synonyms/rhymes/…). It is also
published at https://huggingface.co/datasets/TigreGotico/infopedia-pt-heterophones
"""
from bifonia import infopedia

# all readings of a heterophone, each tied to its IPA + definitions
for r in infopedia.readings("colher"):
    print(f"{r['sense']:>8} ({r['pos']:<18}) {r['ipa']:<10} {r['definitions'][:60]}")

# direct IPA lookup, optionally filtered by POS
print("\nsede (noun):", infopedia.ipa("sede", "NOUN"))
print("governo (verb):", infopedia.ipa("governo", "VERB"))

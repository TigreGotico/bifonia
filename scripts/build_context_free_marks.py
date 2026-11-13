"""Build bifonia/data/context_free_marks.csv, the table of phonetic marks that
need no context.

A mark is an accent or a trema that an older Portuguese spelling norm wrote and
that AO1990 dropped, such as pêra, heróico, idéia or lingüiça. A downstream G2P
reads the vowel from the mark, so add_extra_diacritics writes it back. A row
enters the table for a lect only when it passes three checks:

1. A cited norm wrote the mark. The candidates are the rows of the tugalex
   AO1990 maps whose old form differs from the new form only by a diacritic.
   Words of the context table (bifonia/data/heterophonic_homographs.csv) are
   skipped, and so is a differential mark whose unmarked homograph is in living
   use (pela, the contraction of per and la, AO1990 Base IX 9.º).
2. The lect says the vowel the mark encodes. pt-PT reads the Infopédia
   transcriptions: the Infopédia rows of the unified lexicon, the pt-PT files of
   o2i-lexicons and bifonia/data/infopedia_pt.csv. pt-BR reads the pt-BR rows
   of the unified lexicon (Michaelis and Wiktionary) and the pt-BR files of
   o2i-lexicons. Every reading must say the vowel. The one exception is a word
   whose other readings belong to a homograph that a cited source marks archaic
   (ARCHAIC_COMPETITORS).
3. orthography2ipa reads the mark and gets closer to the source:
   ``G2P(lect).transcribe_word`` gives the source vowel for the marked form and
   another vowel for the unmarked form.

The vowel of an acute or a circumflex, and of the acute on a hiatus i or u, is
the stressed vowel: the first vowel after the primary stress mark. The vowel of
the trema is the [w] after [k] or [ɡ]; the check counts those sequences.

Needs orthography2ipa with its portuguese and lexicons extras, as tugaphone
installs it. The unified lexicon is the JSONL file of the Hugging Face dataset
TigreGotico/portuguese-unified-pronunciation-lexicon::

    python scripts/build_context_free_marks.py \\
        --tugalex-data ../tugalex/tugalex/data \\
        --unified portuguese_pronunciation_lexicon.jsonl \\
        --report marks_report.tsv
"""
import argparse
import csv
import importlib.metadata
import json
import pathlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bifonia.data import AMBIGUOUS_WORDS  # noqa: E402

LECTS = ("pt-PT", "pt-BR")
AO_MAPS = {"pt-PT": "acordo_ortografico_pt_PT.csv", "pt-BR": "acordo_ortografico_pt_BR.csv"}

# The vowel each mark encodes. "w" stands for the [w] of ⟨gü qü⟩.
MARK_VOWEL = {"á": "a", "é": "ɛ", "ê": "e", "í": "i", "ó": "ɔ", "ô": "o", "ú": "u", "ü": "w"}

# Differential marks whose unmarked homograph is a living word. The mark is
# context-dependent, so the context table would have to own it.
LIVING_COMPETITORS = {
    "pela": "AO1990 Base IX 9.º: pela(s), combinação de per e la(s), a living contraction",
}

# Differential marks whose other readings belong to archaic homographs.
ARCHAIC_COMPETITORS = {
    "pera": ("Acordo de 1945 Base XXII: «pêra, substantivo, e pera, preposição arcaica»; "
             "Priberam pera: the preposition is «[Antigo] Para.» and péra is «[Antigo] Pedra.»"),
}

VOWELS = set("aɐeɛəɨiɪoɔuʊyæɑɒʌɜɘɵɤøœ")
VELAR_W = re.compile(r"[kɡg](?:w|ʷ|u̯)")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def stressed_vowel(ipa: str):
    """Return the first vowel after the last primary stress mark, or None."""
    if "ˈ" not in ipa:
        return None
    for char in unicodedata.normalize("NFD", ipa.rsplit("ˈ", 1)[1]):
        if char in VOWELS:
            return char
    return None


def readings(ipa: str) -> list:
    """Split a source cell into its alternative readings."""
    return [r.strip(" /[]") for r in re.split(r"[,~]", nfc(ipa)) if r.strip(" /[]")]


def feature(ipa: str, vowel: str):
    """The value of *ipa* that the mark decides: the stressed vowel, or the count
    of [w] after a velar for the trema."""
    return len(VELAR_W.findall(ipa)) if vowel == "w" else stressed_vowel(ipa)


def load_candidates(tugalex_data: pathlib.Path):
    """Return ({(word, form): [map lect, ...]}, input row count)."""
    candidates = defaultdict(list)
    rows_in = 0
    for lect, name in AO_MAPS.items():
        with open(tugalex_data / name, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows_in += 1
                old, new = nfc(row["old_form"]).lower(), nfc(row["new_form"]).lower()
                if old != new and len(old) == len(new) and fold(old) == fold(new):
                    candidates[(new, old)].append(lect)
    return candidates, rows_in


def load_sources(unified: pathlib.Path, words: set) -> dict:
    """Return {lect: {word: [(source label, ipa), ...]}} for *words*."""
    sources = {lect: defaultdict(list) for lect in LECTS}
    with open(unified, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            word = nfc(row["word"]).lower()
            if word not in words or not row["ipa_narrow"]:
                continue
            if row["region"] == "pt-PT" and row["infopedia_url"]:
                sources["pt-PT"][word].append((f"Infopédia {row['infopedia_url']}", row["ipa_narrow"]))
            elif row["region"] == "pt-BR":
                url = row["michaelis_url"] or row["wiktionary_url"]
                label = "Michaelis" if row["michaelis_url"] else "Wiktionary"
                sources["pt-BR"][word].append((f"{label} {url}", row["ipa_narrow"]))

    import o2i_lexicons
    for lect in LECTS:
        for kind, path in (("exception", o2i_lexicons.tsv_path(lect)),
                           ("patch", o2i_lexicons.patch_tsv_path(lect))):
            label = f"o2i-lexicons {o2i_lexicons.__version__} {lect} {kind} {path.name}"
            for line in path.read_text(encoding="utf-8").splitlines():
                word, _, ipa = line.partition("\t")
                if word in words and ipa:
                    sources[lect][word].append((label, ipa))

    with open(ROOT / "bifonia" / "data" / "infopedia_pt.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["word"] in words:
                sources["pt-PT"][row["word"]].append((f"Infopédia {row['url']}", row["ipa"]))
    return sources


def norm_clause(word: str, form: str) -> str:
    marks = {m for w, m in zip(word, form) if w != m}
    if word in ARCHAIC_COMPETITORS:
        return "Acordo de 1945 Base XXII"
    if marks == {"ü"}:
        return "AO1990 Base XIV 1.º"
    if marks <= {"é", "ó"} and re.search(r"[éó]i", form):
        return "AO1990 Base IX 3.º"
    if marks == {"ô"} and "ôo" in form:
        return "AO1990 Base IX 8.º"
    if marks <= {"í", "ú"} and re.search(r"[aeiou][iuo][íú]", form):
        return "AO1990 Base X 4.º"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tugalex-data", type=pathlib.Path, required=True,
                    help="the tugalex/data directory with the AO1990 maps")
    ap.add_argument("--unified", type=pathlib.Path, required=True,
                    help="portuguese_pronunciation_lexicon.jsonl of the unified lexicon")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "bifonia" / "data" / "context_free_marks.csv")
    ap.add_argument("--report", type=pathlib.Path,
                    help="write one TSV line per candidate and lect with the verdict")
    args = ap.parse_args()

    from orthography2ipa import G2P

    candidates, rows_in = load_candidates(args.tugalex_data)
    print(f"AO map rows in: {rows_in}; rows that differ only by a diacritic: "
          f"{sum(len(v) for v in candidates.values())}; distinct (word, form) pairs: {len(candidates)}")

    report, kept, dropped = [], defaultdict(dict), Counter()
    evaluated = {}
    for (word, form), maps in sorted(candidates.items()):
        if not re.fullmatch(r"\w+", word):
            reason = "not one token: add_extra_diacritics marks single words"
        elif word in AMBIGUOUS_WORDS:
            reason = "check 1: the context table owns the word"
        elif word in LIVING_COMPETITORS:
            reason = f"check 1: {LIVING_COMPETITORS[word]}"
        else:
            evaluated[(word, form)] = maps
            continue
        dropped[reason.split(":")[0]] += 1
        report.append((word, form, "*", "drop", reason, "", "", ""))
    print(f"pairs past the structural and check-1 filters: {len(evaluated)}")

    sources = load_sources(args.unified, {w for w, _ in evaluated})
    g2p = {lect: G2P(lect) for lect in LECTS}
    for (word, form), maps in evaluated.items():
        marks = [m for w, m in zip(word, form) if w != m]
        vowels = sorted({MARK_VOWEL.get(m, "?") for m in marks})
        for lect in LECTS:
            entries = [(label, r) for label, ipa in sources[lect].get(word, []) for r in readings(ipa)]
            unmarked = g2p[lect].transcribe_word(word)
            marked = g2p[lect].transcribe_word(form)
            src_values = sorted({str([feature(r, v) for v in vowels]) for _, r in entries})
            # A trema needs as many [w] after a velar as the source has, and at
            # least one per ü; ⟨qua gua⟩ say [w] with no mark.
            w_counts = {feature(r, "w") for _, r in entries}
            w_target = max(w_counts, default=0)
            if len(w_counts) > 1 or w_target < form.count("ü"):
                w_target = None
            target = [w_target if v == "w" else v for v in vowels]

            def says(ipa):
                return all(feature(ipa, v) == t for v, t in zip(vowels, target))

            agreeing = [(label, r) for label, r in entries if says(r)]
            if "?" in vowels:
                verdict, reason = "drop", "check 2: the mark encodes no vowel quality"
            elif not entries:
                verdict, reason = "drop", "check 2: no transcription for the lect"
            elif "w" in vowels and len(w_counts) > 1:
                verdict, reason = "drop", f"check 2: the readings disagree {src_values}"
            elif not agreeing:
                verdict, reason = "drop", f"check 2: the lect says another vowel {src_values}"
            elif len(agreeing) < len(entries) and word not in ARCHAIC_COMPETITORS:
                verdict, reason = "drop", f"check 2: the readings disagree {src_values}"
            elif not says(marked):
                verdict, reason = "drop", "check 3: o2i misreads the marked form"
            elif says(unmarked):
                verdict, reason = "drop", "check 3: o2i already says the vowel unmarked"
            else:
                verdict, reason = "keep", "all three checks pass"
                labels = list(dict.fromkeys(label for label, _ in agreeing))
                ipas = list(dict.fromkeys(r for _, r in agreeing))
                kept[(word, form)][lect] = (labels, ipas)
            if verdict == "drop":
                dropped[reason.split(":")[0]] += 1
            report.append((word, form, lect, verdict, reason,
                           " | ".join(f"{label}: {r}" for label, r in entries), unmarked, marked))

    rows = []
    for (word, form), per_lect in sorted(kept.items()):
        maps = evaluated[(word, form)]
        clause = norm_clause(word, form)
        norm = "; ".join([f"tugalex {AO_MAPS[m]}: {form} → {word}" for m in maps]
                         + ([clause] if clause else []))
        if word in ARCHAIC_COMPETITORS:
            norm += f"; {ARCHAIC_COMPETITORS[word]}"
        lects = [lect for lect in LECTS if lect in per_lect]
        rows.append({
            "word": word,
            "form": form,
            "lects": "|".join(lects),
            "norm_source": norm,
            "pronunciation_source": "; ".join(f"{lect}: {' | '.join(per_lect[lect][0])}" for lect in lects),
            "transcription": "; ".join(f"{lect}: {' | '.join(per_lect[lect][1])}" for lect in lects),
        })
    seen = Counter((r["word"], lect) for r in rows for lect in r["lects"].split("|"))
    assert all(n == 1 for n in seen.values()), [k for k, n in seen.items() if n > 1]

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(["word", "form", "lect", "verdict", "reason", "source readings",
                             "o2i unmarked", "o2i marked"])
            writer.writerows(report)

    per_lect = Counter(lect for r in rows for lect in r["lects"].split("|"))
    print(f"orthography2ipa {importlib.metadata.version('orthography2ipa')}")
    print(f"rows out: {len(rows)}; kept per lect: {dict(per_lect)}")
    print(f"drops: {dict(dropped)}")
    lect_verdicts = sum(1 for r in report if r[2] != "*")
    print(f"lect verdicts: {lect_verdicts} = {len(evaluated)} pairs x {len(LECTS)} lects: "
          f"{lect_verdicts == len(evaluated) * len(LECTS)}")


if __name__ == "__main__":
    main()

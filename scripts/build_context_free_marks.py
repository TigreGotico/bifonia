"""Build bifonia/data/context_free_marks.csv, the table of phonetic marks that
need no context.

A mark is an accent or a trema that an older Portuguese spelling norm wrote and
that AO1990 dropped, such as pêra, heróico, idéia or lingüiça. A downstream G2P
reads the vowel from the mark, so add_extra_diacritics writes it back. A row
enters the table for a lect code only when it passes three checks:

1. A cited norm wrote the mark. The candidates are the rows of the tugalex
   AO1990 maps whose old form differs from the new form only by a diacritic.
   Words of the context table (bifonia/data/heterophonic_homographs.csv) are
   skipped, and so is a differential mark whose unmarked homograph is in living
   use (pela, the contraction of per and la, AO1990 Base IX 9.º).
2. The lect says the vowel the mark encodes, and its sources show one reading
   of the bare form. pt-PT reads the Infopédia transcriptions: the Infopédia
   rows of the unified lexicon, the pt-PT files of o2i-lexicons and
   bifonia/data/infopedia_pt.csv. pt-BR reads the pt-BR rows of the unified
   lexicon (Michaelis and Wiktionary) and the pt-BR files of o2i-lexicons. A
   source row for a capitalised headword covers the proper noun only, and the
   script does not read it. Every reading must say the vowel; the one
   exception is a word whose other readings belong to a homograph that a
   cited source marks archaic (ARCHAIC_COMPETITORS). The script also drops a
   word whose cited source lists a second reading (SECOND_READINGS), and a
   pt-BR ⟨-eia⟩ word that is also a form of an ⟨-ear⟩ verb of the unified
   lexicon, because the verb form is a second reading (EAR_VERB_RULE).
3. orthography2ipa reads the mark and gets closer to the source, for every
   lect code that the lect matching of bifonia.data reaches: each pt-PT and
   pt-BR code of ``orthography2ipa.available_codes()``. The pt-PT codes read
   the pt-PT sources, and the pt-BR codes the pt-BR sources. For each code,
   ``G2P(code).transcribe_word`` must give the source vowel for the marked
   form and another vowel for the unmarked form, and the marked output must be
   no further from the source (edit distance) than the unmarked output. For a
   code with a packaged o2i-lexicons file (pt-PT and pt-BR, as tugaphone reads
   them), the same engine with ``packaged_lexicon(code, patches=True)`` must
   give the source vowel for the marked form, and its marked output must be no
   further from the source than its unmarked output. The sources describe
   pt-PT and pt-BR, not their sub-lects, so a sub-lect code is checked only
   when its parent passes with the plain engine. A sub-lect spec that reads
   the unmarked word differently from its parent may encode a trait of that
   sub-lect, and no source here says that the trait is wrong.

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

# Words whose cited source lists a second reading of the bare form with another
# vowel, which the unified lexicon does not carry.
SECOND_READINGS = {
    ("ateia", "pt-BR"): ("pt.wiktionary ateia, Brasil: adjective and noun /ɐˈtɛj.ɐ/, "
                         "verb forms of atear /ɐˈtej.ɐ/"),
}

# A pt-BR ⟨-eia⟩ word that is also an ⟨-ear⟩ verb form has a second reading. The
# sources show the two vowels for ateia only; ideia, estreia and traqueia stay out
# by analogy with it, which leaves their output as it was.
EAR_VERB_RULE = ("an -ear verb inserts i after the stem e in its stressed present forms (Priberam "
                 "https://dicionario.priberam.org/conjugar/passear: passeio, passeia, passeiam; "
                 "https://dicionario.priberam.org/conjugar/atear: ateia), so an -eia word can also "
                 "be a verb form; pt.wiktionary https://pt.wiktionary.org/wiki/ateia (Brasil) gives "
                 "the noun and adjective /ɐˈtɛj.ɐ/ and the verb forms /ɐˈtej.ɐ/, another vowel")

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


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def distance(ipa: str, readings_: list) -> int:
    """The edit distance from *ipa* to the nearest source reading."""
    ipa = unicodedata.normalize("NFD", ipa).replace(".", "")
    return min(edit_distance(ipa, unicodedata.normalize("NFD", r).replace(".", "")) for r in readings_)


def load_sources(unified: pathlib.Path, words: set):
    """Return ({lect: {word: [(source label, ipa), ...]}} for *words*,
    {⟨-ear⟩ headword: source label})."""
    sources = {lect: defaultdict(list) for lect in LECTS}
    ear_verbs = {w[:-3] + "ear" for w in words if w.endswith("eia")}
    verb_sources = {}
    with open(unified, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            headword = nfc(row["word"])
            if headword in ear_verbs:
                url = row["infopedia_url"] or row["michaelis_url"] or row["wiktionary_url"]
                if url or headword not in verb_sources:
                    verb_sources.setdefault(headword, url or row["region"])
                    if url and not verb_sources[headword].startswith("http"):
                        verb_sources[headword] = url
            if headword not in words or not row["ipa_narrow"]:
                continue
            word = headword
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
    return sources, verb_sources


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
                    help="write one TSV line per candidate and lect code with the verdict")
    args = ap.parse_args()

    import o2i_lexicons
    from orthography2ipa import G2P, Pipeline, available_codes, packaged_lexicon, resolve
    from orthography2ipa.registry import _ALIASES  # orthography2ipa has no public alias list

    codes = {lect: [c for c in available_codes() if c == lect or c.startswith(lect + "-")]
             for lect in LECTS}
    g2p = {c: G2P(c) for cs in codes.values() for c in cs}
    g2p_lexicon = {c: G2P(c, pipeline=Pipeline(lexicon=packaged_lexicon(c, patches=True)))
                   for c in o2i_lexicons.available_codes() if c in g2p}
    print(f"lect codes: {sum(len(cs) for cs in codes.values())} "
          f"({', '.join(f'{lect}: {len(cs)}' for lect, cs in codes.items())}); "
          f"with a packaged lexicon: {sorted(g2p_lexicon)}")

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
        dropped[(reason.split(":")[0], "*")] += 1
        report.append((word, form, "*", "drop", reason, "", "", "", "", ""))
    print(f"pairs past the structural and check-1 filters: {len(evaluated)}")

    sources, verb_sources = load_sources(args.unified, {w for w, _ in evaluated})
    evidence = defaultdict(dict)
    for (word, form), maps in evaluated.items():
        marks = [m for w, m in zip(word, form) if w != m]
        vowels = sorted({MARK_VOWEL.get(m, "?") for m in marks})
        for lect in LECTS:
            entries = [(label, r) for label, ipa in sources[lect].get(word, []) for r in readings(ipa)]
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
            verb = word[:-3] + "ear"
            reason = None
            if "?" in vowels:
                reason = "check 2: the mark encodes no vowel quality"
            elif not entries:
                reason = "check 2: no transcription for the lect"
            elif "w" in vowels and len(w_counts) > 1:
                reason = f"check 2: the readings disagree {src_values}"
            elif not agreeing:
                reason = f"check 2: the lect says another vowel {src_values}"
            elif len(agreeing) < len(entries) and word not in ARCHAIC_COMPETITORS:
                reason = f"check 2: the readings disagree {src_values}"
            elif (word, lect) in SECOND_READINGS:
                reason = f"check 2: a second reading: {SECOND_READINGS[(word, lect)]}"
            elif lect == "pt-BR" and word.endswith("eia") and verb in verb_sources:
                reason = (f"check 2: also a form of the verb {verb} ({verb_sources[verb]}); "
                          f"{EAR_VERB_RULE}")
            source_cell = " | ".join(f"{label}: {r}" for label, r in entries)
            if reason:
                dropped[(reason.split(":")[0], lect)] += 1
                report.append((word, form, lect, "drop", reason, source_cell,
                               g2p[lect].transcribe_word(word), g2p[lect].transcribe_word(form), "", ""))
                continue
            agreeing_readings = [r for _, r in agreeing]
            parent_gain = False
            for code in codes[lect]:
                unmarked = g2p[code].transcribe_word(word)
                marked = g2p[code].transcribe_word(form)
                lex_unmarked = lex_marked = ""
                if code != lect and not parent_gain:
                    reason = ("check 3: the parent lect shows no gain, and no source "
                              "describes the sub-lect")
                elif not says(marked):
                    reason = "check 3: o2i misreads the marked form"
                elif says(unmarked):
                    reason = "check 3: o2i already says the vowel unmarked"
                elif distance(marked, agreeing_readings) > distance(unmarked, agreeing_readings):
                    reason = "check 3: the marked output is further from the source"
                else:
                    parent_gain = parent_gain or code == lect
                if not reason and code in g2p_lexicon:
                    lex_unmarked = g2p_lexicon[code].transcribe_word(word)
                    lex_marked = g2p_lexicon[code].transcribe_word(form)
                    if not says(lex_marked):
                        reason = "check 3: with the packaged lexicon, o2i misreads the marked form"
                    elif distance(lex_marked, agreeing_readings) > distance(lex_unmarked, agreeing_readings):
                        reason = ("check 3: with the packaged lexicon, the marked output is "
                                  "further from the source")
                if reason:
                    dropped[(reason.split(":")[0], code)] += 1
                    verdict = "drop"
                else:
                    reason, verdict = "all three checks pass", "keep"
                    kept[(word, form)][code] = lect
                    evidence[(word, form)][lect] = agreeing
                report.append((word, form, code, verdict, reason, source_cell,
                               unmarked, marked, lex_unmarked, lex_marked))
                reason = None

    rows = []
    all_codes = [c for cs in codes.values() for c in cs]
    # An orthography2ipa sub-lect alias (pt-BR-x-sao-paulo for pt-BR-x-sp) names
    # the same spec, so it takes the same rows. Matched by prefix alone, it would
    # take the rows of its parent, which its spec may misread.
    aliases = {code: sorted(a for a in _ALIASES if a.startswith(code[:5] + "-x-") and resolve(a) == code)
               for code in all_codes}
    for (word, form), per_code in sorted(kept.items()):
        maps = evaluated[(word, form)]
        clause = norm_clause(word, form)
        norm = "; ".join([f"tugalex {AO_MAPS[m]}: {form} → {word}" for m in maps]
                         + ([clause] if clause else []))
        if word in ARCHAIC_COMPETITORS:
            norm += f"; {ARCHAIC_COMPETITORS[word]}"
        lects = [lect for lect in LECTS if lect in evidence[(word, form)]]
        sourced = {lect: (list(dict.fromkeys(label for label, _ in evidence[(word, form)][lect])),
                          list(dict.fromkeys(r for _, r in evidence[(word, form)][lect])))
                   for lect in lects}
        rows.append({
            "word": word,
            "form": form,
            "lects": "|".join(name for c in all_codes if c in per_code for name in [c, *aliases[c]]),
            "norm_source": norm,
            "pronunciation_source": "; ".join(f"{lect}: {' | '.join(sourced[lect][0])}" for lect in lects),
            "transcription": "; ".join(f"{lect}: {' | '.join(sourced[lect][1])}" for lect in lects),
        })
    seen = Counter((r["word"], code) for r in rows for code in r["lects"].split("|"))
    assert all(n == 1 for n in seen.values()), [k for k, n in seen.items() if n > 1]
    per_code = Counter(code for r in rows for code in r["lects"].split("|"))
    # bifonia.data gives a tag the rows of the longest lect code in the table
    # that is the tag or its parent. A measured code with no row would fall
    # back to its parent's rows, which it did not pass.
    empty = [c for c in all_codes if not per_code[c]]
    if empty:
        sys.exit(f"lect codes with no row would take their parent's rows: {empty}")

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(["word", "form", "lect", "verdict", "reason", "source readings",
                             "o2i unmarked", "o2i marked", "o2i+lexicon unmarked", "o2i+lexicon marked"])
            writer.writerows(report)

    print(f"orthography2ipa {importlib.metadata.version('orthography2ipa')}, "
          f"o2i-lexicons {o2i_lexicons.__version__}")
    print(f"rows out: {len(rows)}")
    print("kept per lect code: " + ", ".join(f"{c} {per_code[c]}" for c in all_codes))
    for (stage, where), n in sorted(dropped.items()):
        print(f"drop {stage} @ {where}: {n}")
    check2 = sum(n for (stage, where), n in dropped.items() if stage == "check 2")
    lect_lines = sum(1 for r in report if r[2] != "*")
    print(f"report lines past check 1: {lect_lines} = check-2 drops {check2} + "
          f"per-code verdicts {lect_lines - check2}")


if __name__ == "__main__":
    main()

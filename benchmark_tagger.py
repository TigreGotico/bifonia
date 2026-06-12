"""
Sense-disambiguation benchmark: how well does each approach pick the right
*meaning* (and therefore the right IPA) of a heterophonic homograph?

Four approaches, scored on the train / test / full splits:

    1. most-common  — always predict the most frequent sense per word (train prior)
    2. spaCy        — POS-tag the sentence, map the homograph's POS -> sense
    3. Stanza       — same, with the Stanford neural tagger
    4. rule-based   — bifonia's context + meaning resolver (guess_sense)

A POS tagger can only ever output a part of speech, so when two senses share a
POS (sede thirst/seat are both NOUN; corte cut/court are both nominal; forma
mould/shape; molho sauce/bundle) it maps to the most-common one and gets the
minority sense wrong *by construction*. The rule-based resolver reads meaning
cues and recovers it. That gap is what this dataset measures.

Usage::

    python benchmark_tagger.py                 # full run (spaCy + Stanza, slow)
    python benchmark_tagger.py --limit 5000     # quick sample
    python benchmark_tagger.py --no-taggers     # rule-based + most-common only
    python benchmark_tagger.py --word sede       # restrict to one word
"""
import argparse
import json
import pathlib
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

from bifonia import tokenize, guess_sense
from bifonia.data import POS_SENSES

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "bifonia" / "data" / "corpus.jsonl"
TRAIN = ROOT / "hf" / "train.jsonl"
TEST = ROOT / "hf" / "test.jsonl"
CACHE = ROOT / "scratch" / "tagger_cache.json"   # optional {sentence: {spacy, stanza}}


def _load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _token_pos(tagged, word):
    for tok, pos in tagged:
        if tok.lower() == word:
            return pos
    return None


def _tag_corpus(sentences, words):
    """Return {sentence: {"spacy": POS, "stanza": POS}} for the homograph token.
    Reuses scratch/tagger_cache.json when it covers every sentence; else tags live."""
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        if all(s in cache for s in sentences):
            return cache

    cache = {s: {} for s in sentences}
    try:
        import spacy
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner", "lemmatizer"])
        for i, doc in enumerate(nlp.pipe(sentences, batch_size=512)):
            s = sentences[i]
            cache[s]["spacy"] = _token_pos([(t.text, t.pos_) for t in doc], words[s])
    except Exception as e:
        print(f"  (spaCy unavailable: {e})")
    try:
        import stanza
        nlp_s = stanza.Pipeline("pt", processors="tokenize,pos", verbose=False,
                                tokenize_no_ssplit=True)
        B = 256
        for start in range(0, len(sentences), B):
            batch = sentences[start:start + B]
            docs = nlp_s.bulk_process([stanza.Document([], text=s) for s in batch])
            for s, doc in zip(batch, docs):
                tagged = [(w.text, w.upos) for sent in doc.sentences for w in sent.words]
                cache[s]["stanza"] = _token_pos(tagged, words[s])
    except Exception as e:
        print(f"  (Stanza unavailable: {e})")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="sample N records (quick run)")
    ap.add_argument("--no-taggers", action="store_true", help="skip spaCy/Stanza")
    ap.add_argument("--word", default=None, help="restrict to one homograph")
    args = ap.parse_args()

    full, train, test = _load(DATA), _load(TRAIN), _load(TEST)
    if args.word:
        full = [r for r in full if r["word"] == args.word]
        train = [r for r in train if r["word"] == args.word]
        test = [r for r in test if r["word"] == args.word]
    if args.limit:
        full = full[:args.limit]

    # priors from TRAIN only (no test leakage)
    sense_freq = defaultdict(Counter)
    for r in train:
        sense_freq[r["word"]][r["sense"]] += 1
    most_common = {w: c.most_common(1)[0][0] for w, c in sense_freq.items()}

    def pos_to_sense(word, upos):
        cands = POS_SENSES.get(word, {}).get(upos)
        if cands:
            return sorted(cands, key=lambda s: -sense_freq[word][s])[0]
        return most_common.get(word)

    def rule_sense(r):
        toks = tokenize(r["sentence"].lower())
        return guess_sense(toks, toks.index(r["word"])) if r["word"] in toks else None

    approaches = {
        "most-common": lambda r: most_common.get(r["word"]),
        "rule-based": rule_sense,
    }
    cache = {}
    if not args.no_taggers:
        sents = {r["sentence"]: r["word"] for r in full + train + test}
        print("Tagging corpus (spaCy + Stanza)…")
        cache = _tag_corpus(list(sents), sents)
        approaches = {
            "most-common": approaches["most-common"],
            "spaCy": lambda r: pos_to_sense(r["word"], cache.get(r["sentence"], {}).get("spacy")),
            "Stanza": lambda r: pos_to_sense(r["word"], cache.get(r["sentence"], {}).get("stanza")),
            "rule-based": rule_sense,
        }

    names = list(approaches)
    print(f"\n{'split':<6} {'n':>7} | " + " ".join(f"{a:>12}" for a in names))
    for label, recs in [("train", train), ("test", test), ("full", full)]:
        if not recs:
            continue
        row = []
        for a in names:
            ok = sum(1 for r in recs if approaches[a](r) == r["sense"])
            row.append(f"{ok / len(recs) * 100:>11.2f}%")
        print(f"{label:<6} {len(recs):>7} | " + " ".join(row))

    if not args.no_taggers and not args.word:
        print("\nPer-(word,sense) where POS tagging hits its ceiling (full corpus):")
        hard = [("sede", "thirst"), ("sede", "seat"), ("corte", "cut"), ("corte", "court"),
                ("forma", "mould"), ("forma", "shape"), ("molho", "sauce"), ("molho", "bundle")]
        buckets = defaultdict(list)
        for r in full:
            buckets[(r["word"], r["sense"])].append(r)
        print(f"{'word/sense':<16} {'n':>6} | {'spaCy':>7} {'Stanza':>7} {'rule':>7}")
        for key in hard:
            recs = buckets.get(key, [])
            if not recs:
                continue
            n = len(recs)
            sp = sum(1 for r in recs if approaches['spaCy'](r) == r['sense']) / n * 100
            st = sum(1 for r in recs if approaches['Stanza'](r) == r['sense']) / n * 100
            ru = sum(1 for r in recs if approaches['rule-based'](r) == r['sense']) / n * 100
            print(f"{key[0] + '/' + key[1]:<16} {n:>6} | {sp:>6.1f}% {st:>6.1f}% {ru:>6.1f}%")


if __name__ == "__main__":
    main()

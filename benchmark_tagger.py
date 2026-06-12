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
from bifonia.scoring import guess_pos as _rule_pos, resolve_sense as _rule_resolve
from bifonia.model import SenseModel, NB_PATH, PERCEPTRON_PATH

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

    # corpus-FREE rules: pure scoring path, independent of how guess_sense is wired
    def rule_sense(r):
        toks = tokenize(r["sentence"].lower())
        if r["word"] not in toks:
            return None
        idx = toks.index(r["word"])
        return _rule_resolve(r["word"], toks, idx, _rule_pos(toks, idx))

    nb = SenseModel.load(str(NB_PATH)) if NB_PATH.exists() else None
    perc = SenseModel.load(str(PERCEPTRON_PATH)) if PERCEPTRON_PATH.exists() else None

    def _model_sense(model, r):
        toks = tokenize(r["sentence"].lower())
        if not model or r["word"] not in toks or not model.has(r["word"]):
            return None
        return model.predict(r["word"], toks, toks.index(r["word"]))

    def shipped(r):  # production guess_sense: per-word model routing + rule fallback
        toks = tokenize(r["sentence"].lower())
        return guess_sense(toks, toks.index(r["word"])) if r["word"] in toks else None

    # corpus-free first, then corpus-using (most-common / NB / perceptron), then shipped ensemble
    approaches = {
        "rules(free)": rule_sense,
        "most-common": lambda r: most_common.get(r["word"]),
        "NB": lambda r: _model_sense(nb, r),
        "perceptron": lambda r: _model_sense(perc, r),
        "shipped": shipped,
    }
    cache = {}
    if not args.no_taggers:
        sents = {r["sentence"]: r["word"] for r in full + train + test}
        print("Tagging corpus (spaCy + Stanza)…")
        cache = _tag_corpus(list(sents), sents)
        approaches["spaCy"] = lambda r: pos_to_sense(r["word"], cache.get(r["sentence"], {}).get("spacy"))
        approaches["Stanza"] = lambda r: pos_to_sense(r["word"], cache.get(r["sentence"], {}).get("stanza"))

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

    # ── per-word TEST breakdown: rules(free) vs NB vs perceptron (adoption gate) ──
    if not args.word:
        print("\nPer-word accuracy on TEST  (corpus-free rules vs corpus-using NB / perceptron):")
        by_word = defaultdict(list)
        for r in test:
            by_word[r["word"]].append(r)
        print(f"{'word':<12} {'n':>5} | {'rules':>7} {'NB':>7} {'perc':>7}   best")
        for word in sorted(by_word):
            recs = by_word[word]
            n = len(recs)
            ru = sum(1 for r in recs if rule_sense(r) == r["sense"]) / n * 100
            nbm = sum(1 for r in recs if _model_sense(nb, r) == r["sense"]) / n * 100
            pe = sum(1 for r in recs if _model_sense(perc, r) == r["sense"]) / n * 100
            best = max([("rules", ru), ("NB", nbm), ("perc", pe)], key=lambda x: x[1])[0]
            flag = "" if max(nbm, pe) >= ru - 0.05 else "  ⚠ rules win"
            print(f"{word:<12} {n:>5} | {ru:>6.1f}% {nbm:>6.1f}% {pe:>6.1f}%   {best}{flag}")


if __name__ == "__main__":
    main()

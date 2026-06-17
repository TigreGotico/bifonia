"""
Out-of-distribution benchmark: evaluate every disambiguation approach on real
Portuguese sentences, not synthetic ones.

The OOD set is the public Hugging Face dataset
``TigreGotico/bifonia-pt-homographs-wild`` (real Wikipedia + web sentences, one
meaning labelled per sentence). It is downloaded on demand, so no third-party text
lives in this repository. This is the honest generalisation number — synthetic
benchmarks (``benchmark_tagger.py``) run several points higher because their train
and test sentences share phrasing.

Scores: most-common · rules (no corpus) · Naive-Bayes · perceptron · shipped ensemble.

Usage::

    pip install huggingface_hub
    python benchmark_ood.py
"""
import json
import pathlib
from collections import Counter, defaultdict

from bifonia import tokenize, guess_sense, proper_flags
from bifonia.data import POS_SENSES
from bifonia.scoring import guess_pos as _rule_pos, resolve_sense as _rule_resolve
from bifonia.model import SenseModel, NB_PATH, PERCEPTRON_PATH

ROOT = pathlib.Path(__file__).parent
TRAIN = ROOT / "hf" / "train.jsonl"
HF_REPO = "TigreGotico/bifonia-pt-homographs-wild"
TAG_CACHE = ROOT / "scratch" / "ood_tagger_cache.json"


def _load_ood():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, "test.jsonl", repo_type="dataset")
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _tag(recs):
    """{sentence: {spacy, stanza}} for the homograph token; cached to scratch/."""
    import warnings
    warnings.filterwarnings("ignore")
    words = {r["sentence"]: r["word"] for r in recs}
    if TAG_CACHE.exists():
        cache = json.loads(TAG_CACHE.read_text(encoding="utf-8"))
        if all(s in cache for s in words):
            return cache
    cache = {s: {} for s in words}
    sents = list(words)

    def tok_pos(tagged, w):
        for t, p in tagged:
            if t.lower() == w:
                return p
        return None

    try:
        import spacy
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner", "lemmatizer"])
        for i, doc in enumerate(nlp.pipe(sents, batch_size=512)):
            cache[sents[i]]["spacy"] = tok_pos([(t.text, t.pos_) for t in doc], words[sents[i]])
    except Exception as e:
        print(f"  (spaCy unavailable: {e})")
    try:
        import stanza
        nlp_s = stanza.Pipeline("pt", processors="tokenize,pos", verbose=False,
                                tokenize_no_ssplit=True)
        for k in range(0, len(sents), 256):
            batch = sents[k:k + 256]
            for s, doc in zip(batch, nlp_s.bulk_process(
                    [stanza.Document([], text=s) for s in batch])):
                tagged = [(w.text, w.upos) for sent in doc.sentences for w in sent.words]
                cache[s]["stanza"] = tok_pos(tagged, words[s])
    except Exception as e:
        print(f"  (Stanza unavailable: {e})")
    if TAG_CACHE.parent.exists():
        TAG_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def main():
    recs = _load_ood()
    train = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]

    # majority/frequency baseline from the full bundled corpus (covers every
    # roster word, not just the 27-word train split — otherwise new words have
    # no most-common baseline and the column under-reports).
    from bifonia.corpus import iter_records
    freq = defaultdict(Counter)
    for w, s, _ in iter_records():
        freq[w][s] += 1
    for r in train:
        freq[r["word"]][r["sense"]] += 1
    most_common = {w: c.most_common(1)[0][0] for w, c in freq.items()}

    def pos_to_sense(word, upos):
        cands = POS_SENSES.get(word, {}).get(upos)
        if cands:
            return sorted(cands, key=lambda s: -freq[word][s])[0]
        return most_common.get(word)

    print("tagging OOD sentences (spaCy + Stanza)…")
    tags = _tag(recs)

    nb = SenseModel.load(str(NB_PATH))
    perc = SenseModel.load(str(PERCEPTRON_PATH))

    def toks(r):
        t = tokenize(r["sentence"].lower())
        # tokens may carry trailing punctuation or a leading clitic hyphen — match
        # on the stripped form and normalise the target slot for the scorer.
        i = next((j for j, w in enumerate(t) if w.strip(".,;:!?-") == r["word"]), None)
        if i is None:
            return (None, None, False)
        pf = proper_flags(r["sentence"])
        pr = pf[i] if i < len(pf) else False
        t = list(t); t[i] = r["word"]
        return (t, i, pr)

    def rules(r):
        t, i, pr = toks(r)
        return _rule_resolve(r["word"], t, i, _rule_pos(t, i, proper=pr)) if t else None

    def model(m, r):
        t, i, pr = toks(r)
        return m.predict(r["word"], t, i) if t and m.has(r["word"]) else None

    def shipped(r):
        t, i, pr = toks(r)
        return guess_sense(t, i, proper=pr) if t else None

    appr = {
        "most-common": lambda r: most_common.get(r["word"]),
        "spaCy": lambda r: pos_to_sense(r["word"], tags.get(r["sentence"], {}).get("spacy")),
        "Stanza": lambda r: pos_to_sense(r["word"], tags.get(r["sentence"], {}).get("stanza")),
        "rules(free)": rules,
        "NB": lambda r: model(nb, r),
        "perceptron": lambda r: model(perc, r),
        "shipped": shipped,
    }

    print(f"OOD set: {len(recs)} real sentences, {len({r['word'] for r in recs})} words\n")
    print(f"{'approach':<14} accuracy")
    for name, fn in appr.items():
        ok = sum(1 for r in recs if fn(r) == r["sense"])
        print(f"{name:<14} {ok / len(recs) * 100:6.2f}%  ({ok}/{len(recs)})")

    print("\nper-word (n | rules | NB | perceptron | shipped):")
    byw = defaultdict(list)
    for r in recs:
        byw[r["word"]].append(r)
    for w in sorted(byw):
        rs = byw[w]
        n = len(rs)

        def acc(fn):
            return sum(1 for r in rs if fn(r) == r["sense"]) / n * 100
        print(f"  {w:<12} {n:>3} | {acc(rules):5.0f} | {acc(lambda r: model(nb, r)):5.0f} | "
              f"{acc(lambda r: model(perc, r)):5.0f} | {acc(shipped):5.0f}")


if __name__ == "__main__":
    main()

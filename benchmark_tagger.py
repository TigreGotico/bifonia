"""
Compare the rule-based scorer against a POS tagger on the bifonia corpus.

Usage::

    python benchmark_tagger.py [--tagger tugatagger|stanza] [--word forma]

The script prints per-word and overall accuracy for both the rule-based scorer
and the external tagger, so you can see where each approach wins or loses.

Both taggers output UDEP tags.  We map them to our 4-class set:
    PROPN → NOUN   (e.g. "Porto" as city name)
    AUX   → VERB   (auxiliary verbs that may be tagged separately)

Limitation: external taggers see the *un-diacritized* corpus form (the
ambiguous spelling), so they rely on the same context signals our scorer uses.
The comparison is a fair apples-to-apples test of context-based disambiguation.
"""

import argparse
from collections import defaultdict

from bifonia import _DIACRITIZED_TO_BASE
from bifonia.corpus import CORPUS
from bifonia.scoring import guess_pos


def _normalize(sent: str) -> str:
    """Strip diacritized homograph variants so index lookup works correctly."""
    s = sent.lower()
    for d, b in _DIACRITIZED_TO_BASE.items():
        s = s.replace(d, b)
    return s


_TAG_MAP = {
    "ADP": "ADP",
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "PROPN": "NOUN",
    "AUX": "VERB",
}


def _majority_accuracy(word_filter=None):
    """Baseline: always predict the most frequent POS for each word."""
    wrong = defaultdict(list)
    total = correct = 0
    majority = {}  # word → most-common POS label
    for word, entries in CORPUS.items():
        if word_filter and word != word_filter:
            continue
        counts = {pos: len(sents) for pos, sents in entries.items()}
        majority[word] = max(counts, key=counts.get)

    for word, entries in CORPUS.items():
        if word_filter and word != word_filter:
            continue
        pred = majority[word]
        for pos, sentences in entries.items():
            for sent in sentences:
                ws = sent.lower().split()
                if word not in ws:
                    continue
                total += 1
                if pred == pos:
                    correct += 1
                else:
                    wrong[word].append((pos, pred, sent[:60]))
    return correct, total, wrong, majority


def _rule_accuracy(word_filter=None):
    wrong = defaultdict(list)
    total = correct = 0
    for word, entries in CORPUS.items():
        if word_filter and word != word_filter:
            continue
        for pos, sentences in entries.items():
            for sent in sentences:
                ws = _normalize(sent).split()
                try:
                    idx = ws.index(word)
                except ValueError:
                    continue
                pred = guess_pos(ws, idx)
                total += 1
                if pred == pos:
                    correct += 1
                else:
                    wrong[word].append((pos, pred, sent[:60]))
    return correct, total, wrong


def _tagger_accuracy(tag_fn, word_filter=None):
    wrong = defaultdict(list)
    total = correct = 0
    for word, entries in CORPUS.items():
        if word_filter and word != word_filter:
            continue
        for pos, sentences in entries.items():
            for sent in sentences:
                tagged = tag_fn(sent)
                tokens_lower = [t.lower() for t, _ in tagged]
                try:
                    idx = tokens_lower.index(word)
                except ValueError:
                    continue
                raw_pos = tagged[idx][1]
                pred = _TAG_MAP.get(raw_pos, "NOUN")
                total += 1
                if pred == pos:
                    correct += 1
                else:
                    wrong[word].append((pos, pred, sent[:60]))
    return correct, total, wrong


def _load_brill():
    from tugatagger import TugaTagger
    t = TugaTagger(engine="brill")

    def tag(sent):
        return t.tag(sent)

    return tag, "brill_postagger"


def _load_spacy_tagger():
    from tugatagger import TugaTagger
    t = TugaTagger(engine="spacy")

    def tag(sent):
        return t.tag(sent)

    return tag, "spaCy(pt_core_news_lg)"


def _load_stanza():
    import stanza
    nlp = stanza.Pipeline("pt", processors="tokenize,pos", tokenize_pretokenized=False,
                          logging_level="ERROR")

    def tag(sent):
        doc = nlp(sent)
        result = []
        for sentence in doc.sentences:
            for word in sentence.words:
                result.append((word.text, word.upos))
        return result

    return tag, "Stanza(pt)"


def _print_table(r_wrong, taggers_results, r_total_per_word, r_c, r_t,
                 maj_wrong=None, maj_c=0, maj_t=0, majority=None):
    """
    taggers_results: list of (label, s_c, s_t, s_wrong)
    maj_wrong: per-word wrong list from majority baseline (optional)
    """
    col_w = 8
    show_maj = maj_wrong is not None
    hdr = f"{'Word':<15} {'Rule':>{col_w}}"
    if show_maj:
        hdr += f" {'Majority':>{col_w}}"
    for label, _, _, _ in taggers_results:
        short = label.split("(")[0]  # "spaCy(pt...)" → "spaCy"
        hdr += f" {short:>{col_w}}"
    print("\n" + hdr)
    sep_w = 15 + (1 + col_w) * (1 + len(taggers_results) + (1 if show_maj else 0))
    print("-" * sep_w)

    all_words = sorted(r_total_per_word.keys())
    for word in all_words:
        wt = r_total_per_word.get(word, 0)
        rf = len(r_wrong.get(word, []))
        r_a = (wt - rf) / wt * 100 if wt else 0
        maj_label = f" ({majority[word]})" if majority and word in majority else ""
        row = f"{word:<15} {r_a:>{col_w-1}.1f}%"
        if show_maj:
            mf = len(maj_wrong.get(word, []))
            m_a = (wt - mf) / wt * 100 if wt else 0
            row += f" {m_a:>{col_w-1}.1f}%"
        for _, _, _, s_wrong in taggers_results:
            sf = len(s_wrong.get(word, []))
            s_a = (wt - sf) / wt * 100 if wt else 0
            row += f" {s_a:>{col_w-1}.1f}%"
        if majority and word in majority:
            row += f"  ← majority={majority[word]}"
        print(row)

    print("-" * sep_w)
    r_a = r_c / r_t * 100 if r_t else 0
    row = f"{'OVERALL':<15} {r_a:>{col_w-1}.1f}%"
    if show_maj:
        m_a = maj_c / maj_t * 100 if maj_t else 0
        row += f" {m_a:>{col_w-1}.1f}%"
    for _, s_c, s_t, _ in taggers_results:
        s_a = s_c / s_t * 100 if s_t else 0
        row += f" {s_a:>{col_w-1}.1f}%"
    print(row)

    print(f"\nCorpus: {r_t} sentences across {len(CORPUS)} ambiguous words")
    print(f"Rule-based scorer : {r_c}/{r_t} = {r_a:.2f}%")
    if show_maj:
        print(f"Majority baseline : {maj_c}/{maj_t} = {m_a:.2f}%")
    for label, s_c, s_t, _ in taggers_results:
        s_a = s_c / s_t * 100 if s_t else 0
        print(f"{label:<22}: {s_c}/{s_t} = {s_a:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rule-based vs external POS tagger.")
    parser.add_argument("--tagger", default="all",
                        choices=["all", "brill", "spacy", "stanza"],
                        help="External tagger(s) to compare against (default: all)")
    parser.add_argument("--word", default=None,
                        help="Restrict to one word (e.g. --word forma)")
    parser.add_argument("--errors", action="store_true",
                        help="Print first 5 error sentences for each word")
    args = parser.parse_args()

    filter_ = args.word

    print("Computing majority-class baseline...")
    maj_c, maj_t, maj_wrong, majority = _majority_accuracy(filter_)

    print("Computing rule-based accuracy...")
    r_c, r_t, r_wrong = _rule_accuracy(filter_)

    # per-word total count
    r_total_per_word = {}
    for word, entries in CORPUS.items():
        if filter_ and word != filter_:
            continue
        wt = 0
        for pos, sents in entries.items():
            for sent in sents:
                ws = sent.lower().split()
                if word in ws:
                    wt += 1
        r_total_per_word[word] = wt

    def _try_load(name):
        try:
            if name == "brill":
                return _load_brill()
            if name == "spacy":
                return _load_spacy_tagger()
            if name == "stanza":
                return _load_stanza()
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            return None, None

    if args.tagger == "all":
        candidates = ["brill", "spacy", "stanza"]
    else:
        candidates = [args.tagger]

    taggers_results = []
    for name in candidates:
        print(f"Loading {name}...")
        tag_fn, label = _try_load(name)
        if tag_fn is None:
            continue
        print(f"  Running {label} on {r_t} sentences...")
        s_c, s_t, s_wrong = _tagger_accuracy(tag_fn, filter_)
        taggers_results.append((label, s_c, s_t, s_wrong))

    _print_table(r_wrong, taggers_results, r_total_per_word, r_c, r_t,
                 maj_wrong=maj_wrong, maj_c=maj_c, maj_t=maj_t, majority=majority)

    if args.errors:
        print("\n=== Rule-based errors ===")
        for word, errs in sorted(r_wrong.items()):
            print(f"\n{word}:")
            for true_pos, pred, sent in errs[:5]:
                print(f"  true={true_pos} pred={pred}  {sent}")
        for label, _, _, s_wrong in taggers_results:
            print(f"\n=== {label} errors ===")
            for word, errs in sorted(s_wrong.items()):
                print(f"\n{word}:")
                for true_pos, pred, sent in errs[:5]:
                    print(f"  true={true_pos} pred={pred}  {sent}")


if __name__ == "__main__":
    main()

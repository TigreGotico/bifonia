"""
Parallel corpus generation, bucketed by MEANING (`sense`).

Usage
-----
Generate for senses that need more sentences:
    python corpus_gen.py                            # all targets
    python corpus_gen.py --word sede --sense thirst # one sense
    python corpus_gen.py --word sobre               # all senses of a word

Output is written to staged/<word>__<sense>.txt for human review, then:
    python corpus_gen.py --merge staged/<word>__<sense>.txt

The bucket key is `sense` (a meaning slug); `pos` and `ipa` for each record are
looked up from heterophonic_homographs.csv. Each prompt file carries a
`# sense: <slug>` header that assigns it to a sense (filenames are cosmetic).
"""

import argparse
import asyncio
import os
import json as _json
import re
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Targets: (word, sense, target_count). The bucket is topped up to target.
# ---------------------------------------------------------------------------
TARGETS = [
    ("acerto", "settlement", 1000), ("acerto", "adjust", 1000),
    ("acordo", "agreement", 1000),  ("acordo", "wake", 1000),
    ("cerro", "hill", 1000),        ("cerro", "shut", 1000),
    ("choro", "weeping", 1000),     ("choro", "weep", 1000),
    ("colher", "spoon", 1000),      ("colher", "harvest", 1000),
    ("começo", "beginning", 1000),  ("começo", "begin", 1000),
    ("conserto", "repair", 1000),   ("conserto", "mend", 1000),
    ("coro", "choir", 1000),        ("coro", "blush", 1000),
    ("corte", "court", 1000),       ("corte", "cut", 1000),
    ("forma", "mould", 1000),       ("forma", "shape", 1000),
    ("gosto", "taste", 1000),       ("gosto", "like", 1000),
    ("gozo", "enjoyment", 1000),    ("gozo", "enjoy", 1000),
    ("jogo", "game", 1000),         ("jogo", "play", 1000),
    ("molho", "sauce", 1000),       ("molho", "bundle", 1000),
    ("olho", "eye", 1000),          ("olho", "look", 1000),
    ("para", "purpose", 1000),      ("para", "stop", 1000),
    # 3-way word → smaller per-sense target
    ("pelo", "by_the", 700),        ("pelo", "hair", 700), ("pelo", "peel", 700),
    ("peso", "weight", 1000),       ("peso", "weigh", 1000),
    ("porto", "harbour", 1000),     ("porto", "carry", 1000),
    ("posto", "station", 1000),     ("posto", "post", 1000),
    ("rego", "furrow", 1000),       ("rego", "water", 1000),
    ("seco", "dry", 1000),          ("seco", "dry_vb", 1000),
    ("sede", "seat", 1000),         ("sede", "thirst", 1000),
    # 3-way word; "sail" (nautical) is rare → modest target to avoid repetition
    ("sobre", "about", 700),        ("sobre", "sail", 150), ("sobre", "leftover", 700),
    ("tola", "head", 1000),         ("tola", "foolish", 1000),
    ("torre", "tower", 1000),       ("torre", "roast", 1000),
    ("transtorno", "disorder", 1000), ("transtorno", "upset", 1000),
    # noun/verb -o alternation words (closed-o noun vs open-ɔ 1sg verb)
    ("torno", "lathe", 1000),       ("torno", "turn", 1000),
    ("troco", "change", 1000),      ("troco", "exchange", 1000),
    ("toco", "stump", 1000),        ("toco", "play", 1000),
    ("rogo", "plea", 1000),         ("rogo", "beg", 1000),
    ("contorno", "contour", 1000),  ("contorno", "circumvent", 1000),
    ("entorno", "surroundings", 1000), ("entorno", "spill", 1000),
    # 3-way: adjective (addled) + noun (cuttlefish) share closed-o; verb is open-ɔ
    ("choco", "addled", 700),       ("choco", "cuttlefish", 700), ("choco", "hatch", 700),
    # second wave of noun(closed)/verb(open) pairs — IPA from infopedia
    ("almoço", "lunch", 1000),  ("almoço", "dine", 1000),
    ("rolo", "roll", 1000),  ("rolo", "tumble", 1000),
    ("soco", "punch", 1000),  ("soco", "strike", 1000),
    ("esforço", "effort", 1000),  ("esforço", "strive", 1000),
    ("conforto", "comfort", 1000),  ("conforto", "soothe", 1000),
    ("aborto", "abortion", 1000),  ("aborto", "abort", 1000),
    ("adorno", "adornment", 1000),  ("adorno", "adorn", 1000),
    ("reforço", "reinforcement", 1000),  ("reforço", "reinforce", 1000),
    ("soldo", "pay", 1000),  ("soldo", "weld", 1000),
    ("esboço", "sketch", 1000),  ("esboço", "outline", 1000),
    ("governo", "government", 1000),  ("governo", "govern", 1000),
    ("emprego", "job", 1000),  ("emprego", "employ", 1000),
    ("selo", "stamp", 1000),  ("selo", "seal", 1000),
    ("gelo", "ice", 1000),  ("gelo", "freeze", 1000),
    ("zelo", "zeal", 1000),  ("zelo", "tend", 1000),
    ("cerco", "siege", 1000),  ("cerco", "surround", 1000),
    ("erro", "error", 1000),  ("erro", "err", 1000),
    # wave 3
    ("sopro", "breath", 1000),  ("sopro", "blow", 1000),
    ("forro", "lining", 1000),  ("forro", "line", 1000),
    ("dobro", "double", 1000),  ("dobro", "fold", 1000),
    ("abono", "allowance", 1000),  ("abono", "vouch", 1000),
    ("logro", "deceit", 1000),  ("logro", "deceive", 1000),
    ("topo", "summit", 1000),  ("topo", "bump_into", 1000),
    ("jorro", "jet", 1000),  ("jorro", "gush", 1000),
    ("golfo", "gulf", 1000),  ("golfo", "spew", 1000),
    ("colmo", "culm", 1000),  ("colmo", "thatch", 1000),
    ("fosso", "ditch", 1000),  ("fosso", "root_up", 1000),
    ("toldo", "awning", 1000),  ("toldo", "cloud_over", 1000),
    ("arrojo", "boldness", 1000),  ("arrojo", "hurl", 1000),
    ("despojo", "spoils", 1000),  ("despojo", "strip", 1000),
    ("cobro", "cessation", 1000),  ("cobro", "collect", 1000),
    ("domo", "dome", 1000),  ("domo", "tame", 1000),
    ("sobro", "cork_oak", 1000),  ("sobro", "be_left_over", 1000),
    ("retorno", "return", 1000),  ("retorno", "go_back", 1000),
    ("desgosto", "sorrow", 1000),  ("desgosto", "dislike", 1000),
    ("desconforto", "discomfort", 1000),  ("desconforto", "discomfit", 1000),
    ("desdobro", "unfolding", 1000),  ("desdobro", "unfold", 1000),
    ("redobro", "redoubling", 1000),  ("redobro", "redouble", 1000),
    ("reboco", "plaster", 1000),  ("reboco", "tow", 1000),
    ("decoro", "decorum", 1000),  ("decoro", "memorize", 1000),
    ("estofo", "stuffing", 1000),  ("estofo", "upholster", 1000),
    ("destroço", "wreckage", 1000),  ("destroço", "wreck", 1000),
    ("desafogo", "relief", 1000),  ("desafogo", "relieve", 1000),
    ("arroto", "belch", 1000),  ("arroto", "burp", 1000),
    ("consolo", "consolation", 1000),  ("consolo", "console", 1000),
    ("desconsolo", "disconsolation", 1000),  ("desconsolo", "dishearten", 1000),
    ("desacordo", "disagreement", 1000),  ("desacordo", "disagree", 1000),
    ("engodo", "bait", 1000),  ("engodo", "lure", 1000),
    ("escorço", "foreshortening", 1000),  ("escorço", "foreshorten", 1000),
    ("desaforo", "insolence", 1000),  ("desaforo", "affront", 1000),
    ("abrolho", "caltrop", 1000),  ("abrolho", "sprout", 1000),
    ("rola", "turtledove", 1000),  ("rola", "rolls", 1000),
    ("solto", "loose", 1000),  ("solto", "release", 1000),
    ("encosto", "backrest", 1000),  ("encosto", "lean", 1000),
    ("apelo", "appeal", 1000),  ("apelo", "call_out", 1000),
    ("desprezo", "contempt", 1000),  ("desprezo", "despise", 1000),
    ("enredo", "plot", 1000),  ("enredo", "entangle", 1000),
    ("desenredo", "denouement", 1000),  ("desenredo", "disentangle", 1000),
    ("espeto", "skewer", 1000),  ("espeto", "stab", 1000),
    ("aperto", "squeeze", 1000),  ("aperto", "tighten", 1000),
    ("desvelo", "devotion", 1000),  ("desvelo", "unveil", 1000),
    ("repelo", "hair_pull", 1000),  ("repelo", "pluck", 1000),
    ("arrepelo", "hair_pulling", 1000),  ("arrepelo", "snatch", 1000),
    ("congelo", "freezing", 1000),  ("congelo", "freeze", 1000),
    ("arremesso", "throw", 1000),  ("arremesso", "fling", 1000),
    ("arremedo", "imitation", 1000),  ("arremedo", "mimic", 1000),
    ("despego", "detachment", 1000),  ("despego", "detach", 1000),
    ("desapego", "indifference", 1000),  ("desapego", "let_go", 1000),
    ("apego", "attachment", 1000),  ("apego", "cling", 1000),
    ("desemprego", "noun", 1000),  ("desemprego", "verb", 1000),
    ("desmantelo", "dismantling", 1000),  ("desmantelo", "dismantle", 1000),
    ("atropelo", "trampling", 1000),  ("atropelo", "run_over", 1000),
    ("degelo", "thaw", 1000),  ("degelo", "thaw_out", 1000),
    ("desgelo", "defrosting", 1000),  ("desgelo", "defrost", 1000),
    ("desespero", "despair", 1000),  ("desespero", "despair_at", 1000),
    ("escabelo", "stool", 1000),  ("escabelo", "dishevel", 1000),
    ("novelo", "yarn_ball", 1000),  ("novelo", "narrate", 1000),
    ("relevo", "relief_terrain", 1000),  ("relevo", "emphasize", 1000),
    ("aceno", "nod", 1000),  ("aceno", "beckon", 1000),
    ("sopeso", "heft", 1000),  ("sopeso", "weigh_up", 1000),
    ("empeno", "warping", 1000),  ("empeno", "warp", 1000),
    ("desempeno", "straightening", 1000),  ("desempeno", "straighten", 1000),
    ("sossego", "calm", 1000),  ("sossego", "soothe", 1000),
    ("desassossego", "disquiet", 1000),  ("desassossego", "disturb", 1000),
    ("esmero", "meticulousness", 1000),  ("esmero", "perfect", 1000),
    ("azedo", "sour", 1000),  ("azedo", "turn_sour", 1000),
    ("desempeço", "riddance", 1000),  ("desempeço", "free_up", 1000),
    ("desemperro", "unjamming", 1000),  ("desemperro", "unjam", 1000),
    ("emperro", "jam", 1000),  ("emperro", "stick", 1000),
    ("tempero", "seasoning", 1000),  ("tempero", "season", 1000),
    ("bola", "ball", 1000),         ("bola", "loaf", 1000),
    ("cor", "colour", 1000),        ("cor", "by_heart", 1000),
    ("lobo", "wolf", 1000),         ("lobo", "lobe", 1000),
    ("polo", "pole", 1000),         ("polo", "fledgling", 1000),
    ("renovo", "shoot", 1000),      ("renovo", "renew", 1000),
    ("soma", "sum", 1000),          ("soma", "add", 1000),
    ("força", "strength", 1000),    ("força", "force", 1000),
]

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "pt-PT"
_CORPUS_JSONL = Path(__file__).parent.parent / "bifonia" / "data" / "corpus.jsonl"

# European-Portuguese word characters, for standalone-token matching.
_PT_WORD = "a-zA-ZáéíóúàâêôãõçÁÉÍÓÚÀÂÊÔÃÕÇ"


def _has_token(word: str, line: str) -> bool:
    """True if `word` appears as a whole token (not a substring) in `line`."""
    return re.search(rf"(?<![{_PT_WORD}]){re.escape(word.lower())}(?![{_PT_WORD}])",
                     line.lower()) is not None


def _prompt_sense(path: Path) -> str | None:
    """Read the `# sense: <slug>` header that assigns a prompt to a sense."""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#\s*sense:\s*(\S+)", line)
        if m:
            return m.group(1)
    return None


def _strip_meta(text: str) -> str:
    """Drop `#` header lines so the model never sees the sense tag."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def _load_prompts(word: str, sense: str) -> list[str]:
    """All prompt templates whose `# sense:` header matches (word, sense)."""
    out = []
    for f in sorted(_PROMPTS_DIR.glob(f"{word}_*.prompt")):
        if _prompt_sense(f) == sense:
            out.append(_strip_meta(f.read_text(encoding="utf-8")))
    if out:
        return out
    return [
        f"Gera {{n}} frases em Português Europeu onde \"{word}\" é usado no sentido "
        f"'{sense}' (contexto: {{existing_sample}}).\nApenas uma frase por linha:"
    ]


def _load_existing(word: str, sense: str) -> set:
    """Existing sentences for this (word, sense) bucket from the corpus."""
    try:
        from bifonia.corpus import CORPUS
        return set(CORPUS.get(word, {}).get(sense, []))
    except Exception:
        return set()


def _sample(existing: set, n: int = 5) -> str:
    sample = list(existing)[:n]
    return "; ".join(f'"{s[:50]}"' for s in sample) if sample else "(none yet)"


def _clean(raw: str, word: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line or line.startswith(("#", "-", "*", "```")):
            continue
        if re.match(r"^\d+[\.\)]", line):
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Must contain the headword as a STANDALONE TOKEN (rejects plurals,
        # inflections, and diacritized variants like "pára"/"pêlo").
        if not _has_token(word, line):
            continue
        if len(line.split()) < 4:
            continue
        lines.append(line)
    return lines


# Local Gemma server (no API keys, not rate-limited) — see workspace policy.
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://192.168.1.200:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF")
# Provider: "haiku" (Claude Haiku via the claude CLI), "agentpipe" (fan out over
# several free coding-agent CLIs), or "gemma" (local server).
GEN_PROVIDER = os.environ.get("GEN_PROVIDER", "haiku")
_HAIKU_FANOUT = int(os.environ.get("HAIKU_FANOUT", "3"))
# agentpipe provider fan-out (GEN_PROVIDER=agentpipe). FREE coding-agent CLIs only —
# diverse providers give more varied phrasing than one model. Override via
# GEN_AGENTS=...; add a paid provider (e.g. claude-haiku) ONLY as a fallback when the
# free ones are rate-limited.
_AGENTS = os.environ.get("GEN_AGENTS", "opencode-free,kilo,mimo").split(",")


async def _generate_one(provider: str, word: str, existing: set, n: int, cwd: str,
                        template: str, model: str | None = None) -> list[str]:
    """Ask one agent to generate n sentences; return cleaned list."""
    from agentpipe import Agent
    prompt = template.replace("{n}", str(n)).replace("{existing_sample}", _sample(existing))
    try:
        raw = await Agent(provider, model=model).generate(prompt, cwd=cwd)
    except Exception as e:
        print(f"  [{provider}] ERROR: {e}", file=sys.stderr)
        return []
    return _clean(raw, word)


async def _generate_one_gemma(word: str, existing: set, n: int,
                              template: str, temperature: float) -> list[str]:
    """Generate via the local Gemma chat-completions endpoint."""
    prompt = template.replace("{n}", str(n)).replace("{existing_sample}", _sample(existing))
    body = {"model": LLM_MODEL, "temperature": temperature, "max_tokens": 1400,
            "messages": [{"role": "user", "content": prompt}]}

    def _call():
        req = urllib.request.Request(
            LLM_ENDPOINT + "/v1/chat/completions",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return _json.loads(r.read())["choices"][0]["message"]["content"]

    try:
        raw = await asyncio.to_thread(_call)
    except Exception as e:
        print(f"  [gemma] ERROR: {e}", file=sys.stderr)
        return []
    return _clean(raw, word)


async def generate_batch(word: str, sense: str, target: int, cwd: str) -> list[str]:
    """Fan out across all prompts for this sense; collect new unique sentences."""
    existing = _load_existing(word, sense)
    needed = max(0, target - len(existing))
    if needed <= 0:
        print(f"  {word}/{sense}: already at target ({len(existing)}≥{target}), skipping")
        return []

    prompts = _load_prompts(word, sense)
    print(f"  {word}/{sense}: need {needed} more, {len(prompts)} prompt(s) via {GEN_PROVIDER}")

    tasks = []
    if GEN_PROVIDER == "haiku":
        agents = [("claude-haiku", None)] * _HAIKU_FANOUT
        n_each = max(20, needed // len(prompts) // len(agents) + 15)
        for template in prompts:
            for provider, model in agents:
                tasks.append(_generate_one(provider, word, existing, n_each, cwd, template, model))
    elif GEN_PROVIDER == "agentpipe":
        # Fan out across several free coding-agent CLIs via agentpipe — diverse
        # providers give more varied phrasing than a single model. Set GEN_AGENTS
        # to override the comma-separated provider list.
        agents = [(a.strip(), None) for a in _AGENTS if a.strip()]
        n_each = min(40, max(20, needed // len(prompts) // len(agents) + 15))
        for template in prompts:
            for provider, model in agents:
                tasks.append(_generate_one(provider, word, existing, n_each, cwd, template, model))
    elif GEN_PROVIDER == "gemma":
        per_call = min(40, max(20, needed // len(prompts) // 3 + 10))
        for template in prompts:
            for temp in (0.8, 0.95, 1.1, 1.2):
                tasks.append(_generate_one_gemma(word, existing, per_call, template, temp))
    else:
        raise SystemExit(f"unknown GEN_PROVIDER={GEN_PROVIDER!r} (use haiku, agentpipe or gemma)")

    results = await asyncio.gather(*tasks)
    all_sents, seen = [], set(s.lower() for s in existing)
    for batch in results:
        for s in batch:
            if s.lower() not in seen:
                seen.add(s.lower())
                all_sents.append(s)
    print(f"    → {len(all_sents)} new unique sentences collected")
    return all_sents


def save_staged(word: str, sense: str, sentences: list[str], staged_dir: Path) -> Path:
    staged_dir.mkdir(exist_ok=True)
    path = staged_dir / f"{word}__{sense}.txt"      # double underscore: sense may contain "_"
    path.write_text("".join(s + "\n" for s in sentences), encoding="utf-8")
    print(f"    staged → {path}  ({len(sentences)} sentences)")
    return path


def _pos_ipa(word: str, sense: str) -> tuple[str | None, str | None]:
    try:
        from bifonia.data import HOMOGRAPHS, SENSE_POS
        return SENSE_POS.get(word, {}).get(sense), HOMOGRAPHS.get(word, {}).get(sense)
    except Exception:
        return None, None


def merge_staged(staged_file: Path):
    """Append a staged file's sentences to corpus.jsonl (globally deduped)."""
    import json
    word, _, sense = staged_file.stem.partition("__")
    if not sense:
        print(f"Cannot parse word__sense from filename {staged_file.name}")
        return

    sentences = [l.strip() for l in staged_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = [s for s in sentences if not _has_token(word, s)]
    if bad:
        print(f"  skipping {len(bad)} sentences without a standalone '{word}' token")
    sentences = [s for s in sentences if _has_token(word, s)]
    if not sentences:
        print("No sentences to merge.")
        return

    # global dedup: a sentence may live in only one bucket
    seen = set()
    if _CORPUS_JSONL.exists():
        for line in _CORPUS_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["sentence"].lower())

    pos, ipa = _pos_ipa(word, sense)
    if ipa is None:
        print(f"  unknown (word, sense) = ({word}, {sense}); not in CSV")
        return
    # A staged line is admitted only if the rule engine independently reads it as
    # the claimed sense — token-presence alone would let mislabelled sentences in.
    from bifonia import tokenize, disambiguate, _DIACRITIZED_TO_BASE

    def _reads_as_sense(s):
        toks = tokenize(s.lower())
        idx = next((i for i, t in enumerate(toks)
                    if t.strip(".,;:!?-") == word
                    or _DIACRITIZED_TO_BASE.get(t.strip(".,;:!?-")) == word), None)
        if idx is None:
            return False
        try:
            return disambiguate(toks, idx) == ipa
        except Exception:
            return False

    added = rejected = 0
    with _CORPUS_JSONL.open("a", encoding="utf-8") as fh:
        for s in sentences:
            key = s.strip().lower()
            if key in seen:
                continue
            if not _reads_as_sense(s):
                rejected += 1
                continue
            seen.add(key)
            fh.write(json.dumps({"word": word, "sense": sense, "pos": pos,
                                 "ipa": ipa, "sentence": s}, ensure_ascii=False) + "\n")
            added += 1
    print(f"Merged {added}/{len(sentences)} ({word}/{sense}); "
          f"{rejected} rejected (rule disagreement)")


async def main_async(word_filter, sense_filter, cwd):
    staged_dir = Path(__file__).parent.parent / "staged"
    targets = [t for t in TARGETS
               if (word_filter is None or t[0] == word_filter)
               and (sense_filter is None or t[1] == sense_filter)]
    if not targets:
        print("No matching targets.")
        return
    for word, sense, target in targets:
        print(f"\n── {word} / {sense} ──")
        sentences = await generate_batch(word, sense, target, cwd)
        if sentences:
            save_staged(word, sense, sentences, staged_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default=None)
    parser.add_argument("--sense", default=None)
    parser.add_argument("--merge", default=None, help="Staged file to merge into corpus")
    parser.add_argument("--cwd", default=str(Path(__file__).parent.parent))
    args = parser.parse_args()

    if args.merge:
        merge_staged(Path(args.merge))
        return
    asyncio.run(main_async(args.word, args.sense, args.cwd))


if __name__ == "__main__":
    main()

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
]

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "pt-PT"
_CORPUS_JSONL = Path(__file__).parent / "bifonia" / "data" / "corpus.jsonl"

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
# Provider: "haiku" (Claude Haiku via the claude CLI) or "gemma" (local server).
GEN_PROVIDER = os.environ.get("GEN_PROVIDER", "haiku")
# Claude Haiku via the Claude Code CLI ONLY (never opencode / third-party agents).
_HAIKU_FANOUT = int(os.environ.get("HAIKU_FANOUT", "3"))


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
    elif GEN_PROVIDER == "gemma":
        per_call = min(40, max(20, needed // len(prompts) // 3 + 10))
        for template in prompts:
            for temp in (0.8, 0.95, 1.1, 1.2):
                tasks.append(_generate_one_gemma(word, existing, per_call, template, temp))
    else:
        raise SystemExit(f"unknown GEN_PROVIDER={GEN_PROVIDER!r} (use haiku or gemma)")

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
    added = 0
    with _CORPUS_JSONL.open("a", encoding="utf-8") as fh:
        for s in sentences:
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            fh.write(json.dumps({"word": word, "sense": sense, "pos": pos,
                                 "ipa": ipa, "sentence": s}, ensure_ascii=False) + "\n")
            added += 1
    print(f"Merged {added}/{len(sentences)} new sentences ({word}/{sense}) into corpus.jsonl")


async def main_async(word_filter, sense_filter, cwd):
    staged_dir = Path(__file__).parent / "staged"
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
    parser.add_argument("--cwd", default=str(Path(__file__).parent))
    args = parser.parse_args()

    if args.merge:
        merge_staged(Path(args.merge))
        return
    asyncio.run(main_async(args.word, args.sense, args.cwd))


if __name__ == "__main__":
    main()

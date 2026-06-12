"""
Parallel corpus generation via free coding agents (opencode-free + antigravity-flash-low).

Usage
-----
Generate for words that need more sentences:
    python corpus_gen.py                         # all priority words
    python corpus_gen.py --word sede --pos VERB  # specific word/POS
    python corpus_gen.py --word sobre            # all POS for a word

Output is written to staged/<word>_<pos>.txt for human review before merging.

After review, run:
    python corpus_gen.py --merge staged/<word>_<pos>.txt
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Targets: words that need more sentences.
# Format: (word, pos, current_count, target_count, notes)
# ---------------------------------------------------------------------------
TARGETS = [
    # ── Two-way words: 555 per POS → ~30k total ─────────────────────────────
    ("acerto",     "NOUN", 290, 1000),
    ("acerto",     "VERB", 280, 1000),
    ("acordo",     "NOUN", 292, 1000),
    ("acordo",     "VERB", 287, 1000),
    ("cerro",      "NOUN", 290, 1000),
    ("cerro",      "VERB", 281, 1000),
    ("choro",      "NOUN", 288, 1000),
    ("choro",      "VERB", 281, 1000),
    ("colher",     "NOUN", 357, 1000),
    ("colher",     "VERB", 359, 1000),
    ("começo",     "NOUN", 356, 1000),
    ("começo",     "VERB", 358, 1000),
    ("conserto",   "NOUN", 358, 1000),
    ("conserto",   "VERB", 359, 1000),
    # coro: 2 prompt files (choir, blush) → meaning-balanced generation
    ("coro",       "NOUN", 293, 1000),
    ("coro",       "VERB", 281, 1000),
    # corte: NOUN = royal court only (fem, closed o)
    #        VERB = cut/incision (masc, open o) + subjunctive of cortar (open o)
    ("corte",      "NOUN", 266, 1000),
    ("corte",      "VERB", 321, 1000),
    # forma: NOUN = mould/fôrma (closed o): forma de bolo/pão/sapato
    #        VERB = figura/modo/maneira/formato/condição física/formatura militar
    #               (open o, common) + 3sg of formar (open o)
    ("forma",      "NOUN", 319, 1000),
    ("forma",      "VERB", 317, 1000),
    ("gosto",      "NOUN", 287, 1000),
    ("gosto",      "VERB", 286, 1000),
    ("gozo",       "NOUN", 320, 1000),
    ("gozo",       "VERB", 320, 1000),
    # jogo: 2 prompt files (game, play)
    ("jogo",       "NOUN", 294, 1000),
    ("jogo",       "VERB", 280, 1000),
    # molho: NOUN = sauce only (closed o)
    #        VERB = bundle/sheaf (open o) + 1sg molhar (open o)
    ("molho",      "NOUN", 288, 1000),
    ("molho",      "VERB", 292, 1000),
    ("olho",       "NOUN", 286, 1000),
    ("olho",       "VERB", 281, 1000),
    ("para",       "ADP",  319, 1000),
    ("para",       "VERB", 362, 1000),
    # pelo: 3-way → 400 per POS
    ("pelo",       "ADP",  287, 700),
    ("pelo",       "NOUN", 339, 700),
    ("pelo",       "VERB", 289, 700),
    ("peso",       "NOUN", 286, 1000),
    ("peso",       "VERB", 284, 1000),
    ("porto",      "NOUN", 280, 1000),
    ("porto",      "VERB", 281, 1000),
    # posto: NOUN = cargo / posto de gasolina (closed o); the participle of
    #        pôr ("foi posto") shares this closed reading — not a heterophone.
    #        VERB = 1sg of postar ("eu posto") — the only open-o reading.
    ("posto",      "NOUN", 385, 1000),
    ("posto",      "VERB", 285, 1000),
    ("rego",       "NOUN", 290, 1000),
    ("rego",       "VERB", 291, 1000),
    ("seco",       "ADJ",  228, 1000),
    ("seco",       "VERB", 217, 1000),
    # sede: NOUN = headquarters/seat only (open ɛ)
    #       VERB = thirst+desire (closed e) — both senses are nouns; "sede" is
    #              NOT a verb form (not from ceder → that conjugates to "cede")
    ("sede",       "NOUN", 310, 1000),
    ("sede",       "VERB", 287, 1000),
    # sobre: 3-way → 400 per POS
    #   ADP  = preposition (closed o)
    #   NOUN = nautical "vela alta de um navio" (closed o, homophone of ADP)
    #          — rare term: modest target to avoid model repetition/hallucination
    #   VERB = pres. subjunctive / imperative of sobrar (open o)
    ("sobre",      "ADP",  292, 700),
    ("sobre",      "NOUN",   0, 150),
    ("sobre",      "VERB",  90, 700),
    # tola: NOUN = head/skull (colloquial) + hardwood (open ɔ)
    #       ADJ  = foolish/silly + substantivized "a tola" (closed o)
    ("tola",       "NOUN",   0, 1000),
    ("tola",       "ADJ",  430, 1000),
    ("torre",      "NOUN", 293, 1000),
    ("torre",      "VERB", 281, 1000),
    ("transtorno", "NOUN", 319, 1000),
    ("transtorno", "VERB", 320, 1000),
]

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "pt-PT"

# European-Portuguese word characters, for standalone-token matching.
_PT_WORD = "a-zA-ZáéíóúàâêôãõçÁÉÍÓÚÀÂÊÔÃÕÇ"


def _has_token(word: str, line: str) -> bool:
    """True if `word` appears as a whole token (not a substring) in `line`."""
    return re.search(rf"(?<![{_PT_WORD}]){re.escape(word.lower())}(?![{_PT_WORD}])",
                     line.lower()) is not None


def _load_prompts(word: str, pos: str) -> list[str]:
    """Return all .prompt files for this word/POS (one per meaning, sorted by name)."""
    files = sorted(_PROMPTS_DIR.glob(f"{word}_{pos}*.prompt"))
    if files:
        return [f.read_text(encoding="utf-8") for f in files]
    # fallback generic template
    return [
        f"Gera {{n}} frases em Português Europeu onde \"{word}\" é usado como {pos} "
        f"(contexto: {{existing_sample}}).\nApenas uma frase por linha:"
    ]


def _load_existing(word: str, pos: str) -> set:
    """Return the set of existing sentences for this word/POS from the corpus."""
    try:
        from bifonia.corpus import CORPUS
        return set(CORPUS.get(word, {}).get(pos, []))
    except Exception:
        return set()


def _sample(existing: set, n: int = 5) -> str:
    sample = list(existing)[:n]
    return "; ".join(f'"{s[:50]}"' for s in sample) if sample else "(none yet)"


async def _generate_one(provider: str, word: str, pos: str,
                         existing: set, n: int, cwd: str,
                         template: str | None = None,
                         model: str | None = None) -> list[str]:
    """Ask one agent to generate n sentences; return cleaned list."""
    from agentpipe import Agent
    if template is None:
        template = _load_prompts(word, pos)[0]
    prompt = template.replace("{n}", str(n)).replace(
        "{existing_sample}", _sample(existing)
    )
    try:
        raw = await Agent(provider, model=model).generate(prompt, cwd=cwd)
    except Exception as e:
        print(f"  [{provider}] ERROR: {e}", file=sys.stderr)
        return []

    lines = []
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        # Drop lines that look like meta-commentary or numbering
        if not line or line.startswith(("#", "-", "*", "```")):
            continue
        if re.match(r"^\d+[\.\)]", line):
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Must contain the target word as a STANDALONE TOKEN (not a substring:
        # rejects plurals "colheres", inflections "formam", and accented
        # disambiguating variants "pára"/"pêlo" that aren't the headword).
        if not _has_token(word, line):
            continue
        # Sanity: at least 4 words
        if len(line.split()) < 4:
            continue
        lines.append(line)
    return lines


import os
import json as _json
import urllib.request

# Local Gemma server (no API keys, not rate-limited) — see workspace policy.
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://192.168.1.200:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF")
# Provider for generation: "haiku" (Claude Haiku via free agents), "gemma"
# (local server), or "agents" (legacy free coding agents).
GEN_PROVIDER = os.environ.get("GEN_PROVIDER", "haiku")

# Claude Haiku via the Claude Code CLI ONLY (never opencode / third-party agents).
# How many parallel claude-haiku calls to fan out per meaning.
_HAIKU_FANOUT = int(os.environ.get("HAIKU_FANOUT", "3"))


async def _live_haiku_agents() -> list:
    """Claude Haiku reached only through the claude CLI provider."""
    return [("claude-haiku", None)] * _HAIKU_FANOUT


async def _generate_one_gemma(word: str, pos: str, existing: set, n: int,
                              template: str, temperature: float) -> list[str]:
    """Generate via the local Gemma chat-completions endpoint; return cleaned list."""
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

    lines = []
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line or line.startswith(("#", "-", "*", "```")):
            continue
        if re.match(r"^\d+[\.\)]", line):
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if not _has_token(word, line):
            continue
        if len(line.split()) < 4:
            continue
        lines.append(line)
    return lines


async def generate_batch(word: str, pos: str,
                          current: int, target: int, cwd: str) -> list[str]:
    """Fan out across all meaning-prompts; balance by meaning."""
    existing = _load_existing(word, pos)
    needed = max(0, target - len(existing))
    if needed <= 0:
        print(f"  {word}/{pos}: already at target ({len(existing)}≥{target}), skipping")
        return []

    prompts = _load_prompts(word, pos)
    print(f"  {word}/{pos}: need {needed} more, {len(prompts)} meaning(s) via {GEN_PROVIDER}")

    tasks = []
    if GEN_PROVIDER == "haiku":
        # Claude Haiku via whichever free agents are live (not rate-limited).
        agents = await _live_haiku_agents()
        n_per_meaning = max(20, needed // len(prompts) // len(agents) + 15)
        for template in prompts:
            for provider, model in agents:
                tasks.append(_generate_one(provider, word, pos, existing,
                                           n_per_meaning, cwd, template, model))
    elif GEN_PROVIDER == "gemma":
        # Several Gemma calls per meaning at varied temperature for diversity.
        per_call = min(40, max(20, needed // len(prompts) // 3 + 10))
        for template in prompts:
            for temp in (0.8, 0.95, 1.1, 1.2):
                tasks.append(_generate_one_gemma(word, pos, existing, per_call, template, temp))
    else:
        n_per_meaning = max(20, needed // len(prompts) // 2 + 15)
        for template in prompts:
            tasks.append(_generate_one("opencode-free", word, pos, existing, n_per_meaning, cwd, template))
            tasks.append(_generate_one("antigravity-flash-low", word, pos, existing, n_per_meaning, cwd, template))
    results = await asyncio.gather(*tasks)

    all_sents = []
    seen = set(s.lower() for s in existing)
    for batch in results:
        for s in batch:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                all_sents.append(s)

    print(f"    → {len(all_sents)} new unique sentences collected")
    return all_sents


def save_staged(word: str, pos: str, sentences: list[str], staged_dir: Path):
    staged_dir.mkdir(exist_ok=True)
    path = staged_dir / f"{word}_{pos}.txt"
    with path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"    staged → {path}  ({len(sentences)} sentences)")
    return path


_CORPUS_JSONL = Path(__file__).parent / "bifonia" / "data" / "corpus.jsonl"


def _ipa_for(word: str, pos: str) -> str | None:
    try:
        from bifonia.data import HOMOGRAPHS
        return HOMOGRAPHS.get(word, {}).get(pos)
    except Exception:
        return None


def merge_staged(staged_file: Path):
    """Append a staged file's sentences to corpus.jsonl (deduped, with IPA)."""
    import json
    name = staged_file.stem          # e.g. "sobre_ADP"
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        print(f"Cannot parse word/POS from filename {staged_file.name}")
        return
    word, pos = parts

    sentences = [l.strip() for l in staged_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not sentences:
        print("No sentences to merge.")
        return

    # reject any sentence where the headword is not a standalone token
    bad = [s for s in sentences if not _has_token(word, s)]
    if bad:
        print(f"  skipping {len(bad)} sentences without a standalone '{word}' token")
    sentences = [s for s in sentences if _has_token(word, s)]

    # existing (word,pos,sentence) keys for dedup
    seen = set()
    if _CORPUS_JSONL.exists():
        with _CORPUS_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                seen.add((r["word"], r["pos"], r["sentence"].lower()))

    ipa = _ipa_for(word, pos)
    added = 0
    with _CORPUS_JSONL.open("a", encoding="utf-8") as fh:
        for s in sentences:
            key = (word, pos, s.lower())
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps({"word": word, "pos": pos, "ipa": ipa,
                                 "sentence": s}, ensure_ascii=False) + "\n")
            added += 1
    print(f"Merged {added}/{len(sentences)} new sentences ({word}/{pos}) into corpus.jsonl")


async def main_async(word_filter: str | None, pos_filter: str | None, cwd: str):
    staged_dir = Path(__file__).parent / "staged"
    targets = [t for t in TARGETS
               if (word_filter is None or t[0] == word_filter)
               and (pos_filter is None or t[1] == pos_filter)]

    if not targets:
        print("No matching targets.")
        return

    for word, pos, current, target, *_ in targets:
        print(f"\n── {word} / {pos} ──")
        sentences = await generate_batch(word, pos, current, target, cwd)
        if sentences:
            save_staged(word, pos, sentences, staged_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default=None)
    parser.add_argument("--pos", default=None)
    parser.add_argument("--merge", default=None, help="Staged file to merge into corpus")
    parser.add_argument("--cwd", default=str(Path(__file__).parent))
    args = parser.parse_args()

    if args.merge:
        merge_staged(Path(args.merge))
        return

    asyncio.run(main_async(args.word, args.pos, args.cwd))


if __name__ == "__main__":
    main()

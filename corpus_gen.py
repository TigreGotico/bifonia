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
    # Three-way words: under-represented POS
    ("sede",  "VERB", 60,  250, "3rd-person of sedear: to HOLD/HOST (meetings, events, sessions). Examples: 'O comité sede as suas reuniões em Lisboa', 'A empresa sede as suas conferências no auditório'. NOT the imperative of ser (be). NOT thirst. The subject is always an organisation holding an event somewhere."),
    ("posto", "VERB", 60,  250, "1st-person of postar: 'eu posto fotos', 'posto o pacote', 'posto encomendas'"),
    # Three-way words: top up all POS
    ("sobre", "ADP",  135, 250, "preposition about/over/on: 'falou sobre', 'livro sobre', 'reflexão sobre'"),
    ("sobre", "NOUN", 113, 250, "envelope: 'meteu no sobre', 'fechou o sobre', 'endereçou o sobre'"),
    ("sobre", "VERB", 114, 250, "3rd-person of sobrar (to be left over): 'sobra comida', 'sobre dinheiro', 'sempre sobre pão'"),
    ("pelo",  "ADP",  144, 250, "por+o: 'passou pelo parque', 'foi feito pelo governo', 'pelo menos'"),
    ("pelo",  "NOUN", 123, 250, "body hair/fur: 'o pelo do gato', 'perdeu pelo', 'pelo longo'"),
    ("pelo",  "VERB", 128, 250, "1st-person of pelar (to peel/skin): 'pelo as batatas', 'pelo o frango', 'pelo laranjas'"),
    # Two-way words: under 420 total
    ("para",  "ADP",  199, 280, "preposition to/for/towards: 'vou para casa', 'é para ti', 'para quê'"),
    ("para",  "VERB", 183, 280, "3rd-person of parar (to stop): 'o carro para', 'o coração para', 'quando para de chover'"),
    ("corte", "NOUN", 213, 280, "cut or royal court: 'um corte na mão', 'a corte do rei', 'corte de cabelo'"),
    ("corte", "VERB", 190, 280, "subjunctive of cortar: 'espero que corte', 'para que corte', 'antes que corte'"),
    ("forma", "NOUN", 212, 280, "baking mold ONLY (closed-o): 'a forma do bolo', 'untar a forma', 'forma redonda de tarte'"),
    ("forma", "VERB", 195, 280, "3rd-person of formar OR noun meaning way/manner (open-o): 'desta forma', 'forma equipas', 'de que forma'"),
    ("gozo",  "NOUN", 212, 280, "enjoyment / legal possession: 'no gozo das férias', 'gozo de direitos', 'pleno gozo'"),
    ("gozo",  "VERB", 196, 280, "1st-person of gozar: 'gozo de boa saúde', 'gozo as férias', 'gozo muito'"),
    ("transtorno", "NOUN", 207, 280, "inconvenience/disorder: 'que transtorno!', 'transtorno mental', 'causou um transtorno'"),
    ("transtorno", "VERB", 196, 280, "1st-person of transtornar: 'transtorno os planos', 'transtorno tudo quando'"),
]

PROMPT_TEMPLATE = """\
Generate {n} natural European Portuguese sentences where the word "{word}" is used as {pos_desc}.

Rules:
- Exactly one sentence per line, no numbering, no bullets, no explanations
- Each sentence must contain the word "{word}" (lowercase, no accents on it)
- The word "{word}" must unambiguously be used as {pos_desc}
- Use varied vocabulary, registers, and sentence lengths (6-20 words)
- Do NOT repeat sentences already in this list: {existing_sample}
- {extra_notes}

Output ONLY the sentences, one per line:"""

POS_DESC = {
    "NOUN": 'a NOUN (substantivo)',
    "VERB": 'a VERB (verbo)',
    "ADP":  'a preposition/ADP (preposição)',
    "ADJ":  'an adjective/ADJ (adjetivo)',
}


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


async def _generate_one(provider: str, word: str, pos: str, notes: str,
                         existing: set, n: int, cwd: str) -> list[str]:
    """Ask one agent to generate n sentences; return cleaned list."""
    from agentpipe import Agent
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        word=word,
        pos_desc=POS_DESC[pos],
        extra_notes=notes,
        existing_sample=_sample(existing),
    )
    try:
        raw = await Agent(provider).generate(prompt, cwd=cwd)
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
        # Must contain the target word (case-insensitive)
        if word.lower() not in line.lower():
            continue
        # Sanity: at least 4 words
        if len(line.split()) < 4:
            continue
        lines.append(line)
    return lines


async def generate_batch(word: str, pos: str, notes: str,
                          current: int, target: int, cwd: str) -> list[str]:
    """Fan out to both free agents and merge unique results."""
    existing = _load_existing(word, pos)
    needed = max(0, target - current)
    if needed <= 0:
        print(f"  {word}/{pos}: already at target ({current}), skipping")
        return []

    n_each = max(30, needed // 2 + 20)
    print(f"  {word}/{pos}: need {needed} more → asking each agent for {n_each}")

    results = await asyncio.gather(
        _generate_one("opencode-free",        word, pos, notes, existing, n_each, cwd),
        _generate_one("antigravity-flash-low", word, pos, notes, existing, n_each, cwd),
    )

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


def merge_staged(staged_file: Path):
    """Read a staged file and append its sentences to the appropriate extra_*.py."""
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

    extra_file = Path(__file__).parent / "bifonia" / "data" / f"extra_{word}.py"
    if not extra_file.exists():
        print(f"No extra file found at {extra_file}")
        return

    src = extra_file.read_text(encoding="utf-8")

    # Find the insertion point: the list for this POS
    # Pattern: "POS": [   ...   ]
    pattern = rf'"{pos}":\s*\['
    m = re.search(pattern, src)
    if not m:
        print(f'No "{pos}" list found in {extra_file}')
        return

    # Find the closing bracket of this list
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
        i += 1
    insert_pos = i - 1  # just before the closing ]

    new_entries = "\n".join(f'        "{s}",' for s in sentences)
    new_src = src[:insert_pos] + "\n" + new_entries + "\n    " + src[insert_pos:]
    extra_file.write_text(new_src, encoding="utf-8")
    print(f"Merged {len(sentences)} sentences into {extra_file}")


async def main_async(word_filter: str | None, pos_filter: str | None, cwd: str):
    staged_dir = Path(__file__).parent / "staged"
    targets = [t for t in TARGETS
               if (word_filter is None or t[0] == word_filter)
               and (pos_filter is None or t[1] == pos_filter)]

    if not targets:
        print("No matching targets.")
        return

    for word, pos, current, target, notes in targets:
        print(f"\n── {word} / {pos} ──")
        sentences = await generate_batch(word, pos, notes, current, target, cwd)
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

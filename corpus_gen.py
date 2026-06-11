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
    # ── Two-way words: bring all to ≥280 per POS ────────────────────────────
    ("acerto",  "NOUN", 222, 320, "accuracy/correctness/settlement: 'um acerto de contas', 'o acerto do diagnóstico', 'com acerto'"),
    ("acerto",  "VERB", 200, 320, "1st-person of acertar (to get right/hit): 'acerto no alvo', 'acerto sempre nas previsões', 'acerto a resposta'"),
    ("acordo",  "NOUN", 222, 320, "agreement/accord: 'um acordo de paz', 'chegaram a um acordo', 'o acordo foi assinado'"),
    ("acordo",  "VERB", 213, 320, "1st-person of acordar (to wake/agree): 'acordo cedo', 'acordo com o plano', 'acordo todos os dias às sete'"),
    ("cerro",   "NOUN", 220, 320, "hill/mound: 'no topo do cerro', 'o cerro da aldeia', 'subiu o cerro'"),
    ("cerro",   "VERB", 205, 320, "1st-person of cerrar (to close/clench): 'cerro os olhos', 'cerro o punho', 'cerro a porta devagar'"),
    ("choro",   "NOUN", 212, 320, "crying/weeping: 'um choro de bebé', 'o choro da criança', 'conteve o choro'"),
    ("choro",   "VERB", 203, 320, "1st-person of chorar (to cry): 'choro de alegria', 'choro sempre neste filme', 'choro sozinho'"),
    ("colher",  "NOUN", 222, 320, "spoon: 'uma colher de sopa', 'mexeu com a colher', 'a colher de pau'"),
    ("colher",  "VERB", 216, 320, "to harvest/collect: 'colher fruta', 'vou colher amoras', 'começou a colher os legumes'"),
    ("começo",  "NOUN", 215, 320, "beginning/start: 'no começo do ano', 'um começo difícil', 'desde o começo'"),
    ("começo",  "VERB", 203, 320, "1st-person of começar (to start): 'começo agora', 'começo o trabalho cedo', 'começo a perceber'"),
    ("conserto", "NOUN", 225, 320, "repair/fix: 'o conserto da máquina', 'em conserto', 'levou ao conserto'"),
    ("conserto", "VERB", 225, 320, "1st-person of consertar (to repair): 'conserto bicicletas', 'conserto o computador', 'conserto o estrago'"),
    ("coro",    "NOUN", 227, 320, "choir/chorus: 'o coro da catedral', 'cantou no coro', 'um coro de vozes'"),
    ("coro",    "VERB", 206, 320, "1st-person of corar (to blush/flush): 'coro de vergonha', 'coro facilmente', 'coro quando me elogiam'"),
    ("gosto",   "NOUN", 216, 320, "taste/liking: 'um gosto refinado', 'ao gosto de cada um', 'gosto pessoal'"),
    ("gosto",   "VERB", 218, 320, "1st-person of gostar (to like): 'gosto de música', 'gosto muito', 'gosto do teu trabalho'"),
    ("jogo",    "NOUN", 226, 320, "game/play: 'um jogo de futebol', 'o jogo terminou', 'o jogo das palavras'"),
    ("jogo",    "VERB", 203, 320, "1st-person of jogar (to play/gamble): 'jogo futebol', 'jogo aos dados', 'jogo bem sob pressão'"),
    ("molho",   "NOUN", 224, 320, "sauce/bundle: 'o molho de tomate', 'molho de chaves', 'preparou o molho'"),
    ("molho",   "VERB", 226, 320, "1st-person of molhar (to wet/soak): 'molho o pão no café', 'molho os pés na praia', 'molho a esponja'"),
    ("olho",    "NOUN", 219, 320, "eye: 'o olho direito', 'com o olho vivo', 'olho azul'"),
    ("olho",    "VERB", 199, 320, "1st-person of olhar (to look): 'olho pela janela', 'olho para o céu', 'olho com atenção'"),
    ("peso",    "NOUN", 217, 320, "weight: 'o peso do saco', 'de peso', 'o peso da responsabilidade'"),
    ("peso",    "VERB", 210, 320, "1st-person of pesar (to weigh): 'peso os ingredientes', 'peso 70 kg', 'peso tudo antes de decidir'"),
    ("porto",   "NOUN", 211, 320, "port/harbour OR port wine: 'no porto de Lisboa', 'um copo de porto', 'atracou no porto'"),
    ("porto",   "VERB", 209, 320, "1st-person of portar (to carry/behave): 'porto-me bem', 'porto as malas', 'porto a bandeira'"),
    ("rego",    "NOUN", 223, 320, "irrigation ditch/furrow: 'o rego de rega', 'abriu o rego', 'o rego do campo'"),
    ("rego",    "VERB", 223, 320, "1st-person of regar (to water/irrigate): 'rego as plantas', 'rego o jardim de manhã', 'rego os canteiros'"),
    ("torre",   "NOUN", 225, 320, "tower: 'a torre do castelo', 'torre de controlo', 'a torre de Belém'"),
    ("torre",   "VERB", 200, 320, "1st-person of torrar (to toast/roast): 'torre o pão', 'torre café', 'torre as amêndoas'"),
    # ── Three-way words already near target — small top-up ──────────────────
    ("pelo",  "NOUN", 286, 350, "body hair/fur: 'o pelo do gato', 'perdeu pelo', 'tem pelo comprido'. The animal subject typically OWNS the fur."),
    ("para",  "VERB", 318, 380, "3rd-person of parar (to stop): 'o carro para', 'para de chover', 'o motor para'. Use a concrete stoppable subject in the sentence."),
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

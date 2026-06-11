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
    # ── Two-way words: bring all to 465 per POS (≈930/word → 25k total) ────
    ("acerto",     "NOUN", 290, 465, "accuracy/correctness/settlement: 'um acerto de contas', 'o acerto do diagnóstico', 'com acerto'"),
    ("acerto",     "VERB", 280, 465, "1st-person of acertar (to get right/hit): 'acerto no alvo', 'acerto sempre nas previsões', 'acerto a resposta'"),
    ("acordo",     "NOUN", 292, 465, "agreement/accord: 'um acordo de paz', 'chegaram a um acordo', 'o acordo foi assinado'"),
    ("acordo",     "VERB", 287, 465, "1st-person of acordar (to wake/agree): 'acordo cedo', 'acordo com o plano', 'acordo todos os dias às sete'"),
    ("cerro",      "NOUN", 290, 465, "hill/mound: 'no topo do cerro', 'o cerro da aldeia', 'subiu o cerro'"),
    ("cerro",      "VERB", 281, 465, "1st-person of cerrar (to close/clench): 'cerro os olhos', 'cerro o punho', 'cerro a porta devagar'"),
    ("choro",      "NOUN", 288, 465, "crying/weeping: 'um choro de bebé', 'o choro da criança', 'conteve o choro'"),
    ("choro",      "VERB", 281, 465, "1st-person of chorar (to cry): 'choro de alegria', 'choro sempre neste filme', 'choro sozinho'"),
    ("colher",     "NOUN", 357, 465, "spoon: 'uma colher de sopa', 'mexeu com a colher', 'a colher de pau'"),
    ("colher",     "VERB", 359, 465, "to harvest/collect: 'colher fruta', 'vou colher amoras', 'começou a colher os legumes'"),
    ("começo",     "NOUN", 356, 465, "beginning/start: 'no começo do ano', 'um começo difícil', 'desde o começo'"),
    ("começo",     "VERB", 358, 465, "1st-person of começar (to start): 'começo agora', 'começo o trabalho cedo', 'começo a perceber'"),
    ("conserto",   "NOUN", 358, 465, "repair/fix: 'o conserto da máquina', 'em conserto', 'levou ao conserto'"),
    ("conserto",   "VERB", 359, 465, "1st-person of consertar (to repair): 'conserto bicicletas', 'conserto o computador', 'conserto o estrago'"),
    ("coro",       "NOUN", 293, 465, "choir/chorus: 'o coro da catedral', 'cantou no coro', 'um coro de vozes'"),
    ("coro",       "VERB", 281, 465, "1st-person of corar (to blush/flush): 'coro de vergonha', 'coro facilmente', 'coro quando me elogiam'"),
    ("corte",      "NOUN", 266, 465, "cut/slash OR royal court: 'o corte na pele', 'a corte do rei', 'corte orçamental'"),
    ("corte",      "VERB", 321, 465, "subjunctive/imperative of cortar (to cut): 'que o médico corte', 'corte o papel', 'espero que corte'"),
    ("forma",      "NOUN", 319, 465, "shape/form/mold: 'a forma do bolo', 'de forma clara', 'em boa forma'"),
    ("forma",      "VERB", 317, 465, "3rd-person of formar (to form/shape): 'forma uma equipa', 'a nuvem forma', 'forma novos profissionais'"),
    ("gosto",      "NOUN", 287, 465, "taste/liking: 'um gosto refinado', 'ao gosto de cada um', 'gosto pessoal'"),
    ("gosto",      "VERB", 286, 465, "1st-person of gostar (to like): 'gosto de música', 'gosto muito', 'gosto do teu trabalho'"),
    ("gozo",       "NOUN", 320, 465, "enjoyment/pleasure OR mockery: 'o gozo das férias', 'em gozo de licença', 'que gozo!'"),
    ("gozo",       "VERB", 320, 465, "1st-person of gozar (to enjoy/mock): 'gozo das minhas férias', 'gozo com os amigos', 'gozo de boa saúde'"),
    ("jogo",       "NOUN", 294, 465, "game/play: 'um jogo de futebol', 'o jogo terminou', 'o jogo das palavras'"),
    ("jogo",       "VERB", 280, 465, "1st-person of jogar (to play/gamble): 'jogo futebol', 'jogo aos dados', 'jogo bem sob pressão'"),
    ("molho",      "NOUN", 288, 465, "sauce/bundle: 'o molho de tomate', 'molho de chaves', 'preparou o molho'"),
    ("molho",      "VERB", 292, 465, "1st-person of molhar (to wet/soak): 'molho o pão no café', 'molho os pés na praia', 'molho a esponja'"),
    ("olho",       "NOUN", 286, 465, "eye: 'o olho direito', 'com o olho vivo', 'olho azul'"),
    ("olho",       "VERB", 281, 465, "1st-person of olhar (to look): 'olho pela janela', 'olho para o céu', 'olho com atenção'"),
    ("para",       "ADP",  319, 465, "preposition to/for: 'vou para casa', 'comprei para ti', 'útil para todos'"),
    ("para",       "VERB", 362, 465, "3rd-person of parar (to stop): 'o carro para', 'para de chover', 'o motor para'. Use a concrete stoppable subject."),
    ("pelo",       "ADP",  287, 350, "preposition por+o: 'passou pelo parque', 'foi elogiado pelo professor', 'andou pelo rio'"),
    ("pelo",       "NOUN", 339, 350, "body hair/fur: 'o pelo do gato', 'perdeu pelo', 'tem pelo comprido'"),
    ("pelo",       "VERB", 289, 350, "1st-person of pelar (to peel/skin): 'pelo as batatas', 'pelo os pêssegos', 'pelo a cenoura'"),
    ("peso",       "NOUN", 286, 465, "weight: 'o peso do saco', 'de peso', 'o peso da responsabilidade'"),
    ("peso",       "VERB", 284, 465, "1st-person of pesar (to weigh): 'peso os ingredientes', 'peso 70 kg', 'peso tudo antes de decidir'"),
    ("porto",      "NOUN", 280, 465, "port/harbour OR port wine: 'no porto de Lisboa', 'um copo de porto', 'atracou no porto'"),
    ("porto",      "VERB", 281, 465, "1st-person of portar (to carry/behave): 'porto-me bem', 'porto as malas', 'porto a bandeira'"),
    ("posto",      "NOUN", 385, 465, "post/position/petrol station: 'um posto de trabalho', 'posto de saúde', 'o posto de gasolina'"),
    ("posto",      "VERB", 285, 465, "past participle of pôr (placed/put): 'foi posto à disposição', 'o livro posto na mesa', 'tendo sido posto em prática'"),
    ("rego",       "NOUN", 290, 465, "irrigation ditch/furrow: 'o rego de rega', 'abriu o rego', 'o rego do campo'"),
    ("rego",       "VERB", 291, 465, "1st-person of regar (to water/irrigate): 'rego as plantas', 'rego o jardim de manhã', 'rego os canteiros'"),
    ("seco",       "ADJ",  228, 465, "dry (adjective): 'o clima seco', 'pão seco', 'tempo seco e frio'"),
    ("seco",       "VERB", 217, 465, "1st-person of secar (to dry): 'seco o cabelo', 'seco a louça', 'seco as mãos na toalha'"),
    ("sede",       "NOUN", 310, 465, "thirst OR headquarters: 'tenho sede', 'a sede da empresa', 'matar a sede'"),
    ("sede",       "VERB", 287, 465, "3rd-person of ceder (to yield/cede): 'o governo sede terreno', 'a empresa sede os direitos', 'nunca sede às pressões'"),
    ("sobre",      "ADP",  292, 350, "preposition about/on: 'falou sobre o tema', 'um livro sobre Portugal', 'debruçou-se sobre o problema'"),
    ("sobre",      "NOUN", 291, 350, "surplus/leftover: 'o sobre do orçamento', 'ficou um sobre', 'aproveitou o sobre'"),
    ("sobre",      "VERB", 290, 350, "3rd-person of sobrar (to be left over): 'sobre dinheiro', 'sobre tempo', 'não sobre nada no fim'"),
    ("tola",       "NOUN", 0, 465, "colloquial for head/skull OR a type of lightweight African hardwood (tola wood, Pterygopodium oxyphyllum): 'bater com a tola', 'meter na tola', 'partir a tola', 'a tola à roda', 'dói-me a tola', 'madeira de tola', 'ripas de tola'. Mix both meanings. NEVER use to mean fool — that is the ADJ reading."),
    ("tola",       "ADJ",  370, 465, "foolish/silly (adj or substantivized adj): 'uma ideia tola', 'a rapariga é tola', 'comportou-se como uma tola', 'és uma tola'"),
    ("torre",      "NOUN", 293, 465, "tower: 'a torre do castelo', 'torre de controlo', 'a torre de Belém'"),
    ("torre",      "VERB", 281, 465, "1st-person of torrar (to toast/roast): 'torre o pão', 'torre café', 'torre as amêndoas'"),
    ("transtorno", "NOUN", 319, 465, "disorder/disruption: 'um transtorno mental', 'causou transtorno', 'sem transtorno'"),
    ("transtorno", "VERB", 320, 465, "1st-person of transtornar (to disrupt/upset): 'transtorno os planos', 'transtorno a rotina', 'transtorno tudo'"),
]

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "pt-PT"


def _load_prompt(word: str, pos: str) -> str:
    """Load the .prompt file for this word/POS pair."""
    path = _PROMPTS_DIR / f"{word}_{pos}.prompt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # fallback: generic template (should not be needed once all .prompt files exist)
    return (
        f"Gera {{n}} frases em Português Europeu onde \"{word}\" é usado como {pos} "
        f"(contexto: {{existing_sample}}).\nApenas uma frase por linha:"
    )


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
                         existing: set, n: int, cwd: str) -> list[str]:
    """Ask one agent to generate n sentences; return cleaned list."""
    from agentpipe import Agent
    template = _load_prompt(word, pos)
    prompt = template.replace("{n}", str(n)).replace(
        "{existing_sample}", _sample(existing)
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


async def generate_batch(word: str, pos: str,
                          current: int, target: int, cwd: str) -> list[str]:
    """Fan out to both free agents and merge unique results."""
    existing = _load_existing(word, pos)
    needed = max(0, target - len(existing))
    if needed <= 0:
        print(f"  {word}/{pos}: already at target ({len(existing)}≥{target}), skipping")
        return []

    n_each = max(30, needed // 2 + 20)
    print(f"  {word}/{pos}: need {needed} more → asking each agent for {n_each}")

    results = await asyncio.gather(
        _generate_one("opencode-free",        word, pos, existing, n_each, cwd),
        _generate_one("antigravity-flash-low", word, pos, existing, n_each, cwd),
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

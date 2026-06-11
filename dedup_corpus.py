"""
Deduplicate sentences across ALL corpus group files (grp_*.py + extra_*.py).

For each (word, pos) pair, collect all sentences from all files, keep only first
occurrence, and write the deduplicated set back into the extra_*.py files (which
are the editable ones). The grp_*.py files are left untouched.
"""
import ast
import pathlib
from collections import defaultdict

DATA = pathlib.Path(__file__).parent / "bifonia" / "data"


def _load(path):
    try:
        return ast.literal_eval(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return {}


def _flatten(data: dict) -> dict[tuple, list]:
    """Flatten nested {word: {pos: [...]}} OR {pos: [...]} into {(word, pos): [...]}."""
    result = defaultdict(list)
    for k, v in data.items():
        if isinstance(v, dict):
            # nested: {word: {pos: [sents]}}
            for pos, sents in v.items():
                result[(k, pos)].extend(sents)
        elif isinstance(v, list):
            # flat: {pos: [sents]} — word unknown, skip (shouldn't happen in our files)
            pass
    return result


# ── Step 1: collect all sentences per (word, pos) from all files ──────────────
seen: dict[tuple, set] = defaultdict(set)

for fpath in sorted(DATA.glob("grp_*.py")):
    flat = _flatten(_load(fpath))
    for key, sents in flat.items():
        for s in sents:
            seen[key].add(s.strip().lower())

# ── Step 2: rewrite extra_*.py files, keeping only sentences not already seen ─
total_removed = 0
for fpath in sorted(DATA.glob("extra_*.py")):
    data = _load(fpath)
    if not data:
        continue

    word = next(iter(data))  # outer key is the word name
    pos_dict = data[word]

    new_pos_dict = {}
    file_changed = False
    for pos, sents in pos_dict.items():
        key = (word, pos)
        deduped = []
        for s in sents:
            norm = s.strip().lower()
            if norm not in seen[key]:
                seen[key].add(norm)
                deduped.append(s)
        removed = len(sents) - len(deduped)
        if removed:
            print(f"  {fpath.name}  {pos}: removed {removed} duplicates ({len(sents)}→{len(deduped)})")
            total_removed += removed
            file_changed = True
        new_pos_dict[pos] = deduped

    if not file_changed:
        continue

    # Reconstruct with original nesting
    lines = ["{\n", f'  "{word}": {{\n']
    for pos, sents in new_pos_dict.items():
        lines.append(f'    "{pos}": [\n')
        for s in sents:
            escaped = s.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'      "{escaped}",\n')
        lines.append("    ],\n")
    lines.append("  }\n}\n")
    fpath.write_text("".join(lines), encoding="utf-8")

print(f"\nTotal duplicates removed: {total_removed}")

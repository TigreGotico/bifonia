"""Write the score tables of docs/benchmarks.md from benchmarks/results.json.

Each table sits between ``<!-- results:NAME -->`` and ``<!-- /results:NAME -->``.
Run ``python scripts/benchmark_table.py`` after a change to results.json.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results.json"
PAGE = ROOT / "docs" / "benchmarks.md"
BLOCK = re.compile(r"(<!-- results:(\w+) -->\n).*?(<!-- /results:\2 -->)", re.S)


def _bold(text, on):
    return f"**{text}**" if on else text


def render(table):
    head = "| " + " | ".join(table["columns"]) + " |\n" + table["align"] + "\n"
    parts, rows = [], []
    for row in table["rows"]:
        bold = set(row.get("bold", []))
        cells = [_bold(row["label"], 0 in bold)]
        for col, (fmt, value) in enumerate(zip(table["format"], row["scores"]), 1):
            cell = _bold(fmt.format(value), col in bold)
            if col == 1 and "note" in row:
                cell += " " + row["note"]
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |\n")
    split = table.get("split_at", len(rows))
    parts.append(head + "".join(rows[:split]))
    if split < len(rows):
        cont = dict(table, columns=[table["columns"][0] + " (continued)"] + table["columns"][1:])
        parts.append("| " + " | ".join(cont["columns"]) + " |\n" + table["align"] + "\n"
                     + "".join(rows[split:]))
    return "\n".join(parts)


def build_page(page_text, results):
    def sub(m):
        return m.group(1) + render(results["tables"][m.group(2)]) + m.group(3)
    return BLOCK.sub(sub, page_text)


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    PAGE.write_text(build_page(PAGE.read_text(encoding="utf-8"), results), encoding="utf-8")


if __name__ == "__main__":
    main()

"""The score tables of docs/benchmarks.md match benchmarks/results.json."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("benchmark_table", ROOT / "scripts" / "benchmark_table.py")
benchmark_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_table)


def test_page_tables_match_results():
    page = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    assert benchmark_table.build_page(page, results) == page


def test_every_results_table_is_on_the_page():
    page = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    on_page = {m.group(2) for m in benchmark_table.BLOCK.finditer(page)}
    assert on_page == set(results["tables"])


def test_rules_ood_score():
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    rules = next(r for r in results["tables"]["ood"]["rows"] if r["label"] == "rules (no model)")
    assert rules["scores"] == [88.0]

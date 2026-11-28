"""The score tables of docs/benchmarks.md and docs/index.md match benchmarks/results.json."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("benchmark_table", ROOT / "scripts" / "benchmark_table.py")
benchmark_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_table)


RESULTS = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))


def test_page_tables_match_results():
    for path in benchmark_table.PAGES:
        page = path.read_text(encoding="utf-8")
        assert benchmark_table.build_page(page, RESULTS) == page, path.name


def test_every_results_table_is_on_a_page():
    on_pages = [m.group(2) for path in benchmark_table.PAGES
                for m in benchmark_table.BLOCK.finditer(path.read_text(encoding="utf-8"))]
    assert sorted(on_pages) == sorted(RESULTS["tables"])


def test_the_support_table_reads_the_rules_scores():
    page = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "| 95.2% (gold), 88.0% (wild) |" in page


def test_rules_ood_score():
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    rules = next(r for r in results["tables"]["ood"]["rows"] if r["label"] == "rules (no model)")
    assert rules["scores"] == [88.0]

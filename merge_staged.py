"""
Batch-merge all staged/*.txt files into the corresponding extra_*.py corpus files.

Usage:
    python merge_staged.py            # merge all staged files
    python merge_staged.py --dry-run  # preview what would be merged
"""
import argparse
from pathlib import Path
from corpus_gen import merge_staged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    staged_dir = Path(__file__).parent / "staged"
    files = sorted(staged_dir.glob("*.txt"))
    if not files:
        print("No staged files found.")
        return

    print(f"Found {len(files)} staged files:")
    for f in files:
        n = sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
        print(f"  {f.name}  ({n} sentences)")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        return

    print()
    for f in files:
        merge_staged(f)

    print("\nDone. Run `python dataset.py` to regenerate dataset.csv / dataset.json.")


if __name__ == "__main__":
    main()

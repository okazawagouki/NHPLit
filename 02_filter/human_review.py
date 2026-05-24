"""
02_filter/human_review.py
=========================
Interactive human review of 'pending' items in 01_extracted_db.xlsx.

For each pending paper, shows the title, journal/year, AI confidence,
and abstract, then prompts you to judge:
    p  → pass
    f  → fail
    s  → skip (leave as pending)
    q  → quit and save

Results are written back to 01_extracted_db.xlsx immediately on quit
or after each paper if --autosave is set.

Usage
-----
    python 02_filter/human_review.py            # review all pending
    python 02_filter/human_review.py --autosave # save after every decision
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIVIDER = "─" * 72


def load_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: Database not found: {path}")
    return pd.read_excel(path, dtype=str)


def save_db(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False)


def show_paper(idx: int, total: int, row: pd.Series) -> None:
    """Print one paper to the terminal."""
    print(f"\n{DIVIDER}")
    print(f"  [{idx}/{total}]  PMID: {row.get('pmid', 'N/A')}")
    print(DIVIDER)

    title = row.get("title") or "No title"
    title = title if isinstance(title, str) else "No title"
    print(textwrap.fill(f"TITLE   : {title}", width=72,
                        subsequent_indent=" " * 10))

    journal = row.get("journal") or ""
    year    = row.get("year") or ""
    journal = journal if isinstance(journal, str) else ""
    year    = year if isinstance(year, str) else ""
    print(f"JOURNAL : {journal}  ({year})")

    ai_conf = row.get("ai_confidence", "")
    ai_rel  = row.get("ai_relevant", "")
    print(f"AI      : relevant={ai_rel}  confidence={ai_conf}")

    abstract = row.get("abstract") or ""
    abstract = abstract if isinstance(abstract, str) else "No abstract"
    abstract = " ".join(abstract.split())
    print()
    print(f"  {abstract}")

    print()


def prompt_choice() -> str:
    """Prompt until a valid key is entered."""
    while True:
        raw = input("  [p]ass / [f]ail / [s]kip / [q]uit : ").strip().lower()
        if raw in ("p", "f", "s", "q"):
            return raw
        print("  → Please enter p, f, s, or q.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Human review of pending papers in 01_extracted_db.xlsx"
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config", default=repo_root / "config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--autosave", action="store_true",
        help="Save to xlsx after every decision (safer but slower)"
    )
    args = parser.parse_args()

    cfg     = yaml.safe_load(open(args.config))
    db_path = repo_root / cfg["paths"]["filtered_db"]

    df = load_db(db_path)

    pending_mask = df["flag"] == "pending"
    pending_idx  = df.index[pending_mask].tolist()
    total        = len(pending_idx)

    if total == 0:
        print("No pending items found. Nothing to review.")
        sys.exit(0)

    print(f"\nFound {total} pending paper(s) to review.")
    print("Keys: [p]ass  [f]ail  [s]kip  [q]uit\n")

    n_pass = n_fail = n_skip = 0

    for count, row_idx in enumerate(pending_idx, start=1):
        row = df.loc[row_idx]
        show_paper(count, total, row)

        choice = prompt_choice()

        if choice == "p":
            df.at[row_idx, "flag"] = "pass"
            n_pass += 1
            print("  → Marked as PASS")
        elif choice == "f":
            df.at[row_idx, "flag"] = "fail"
            n_fail += 1
            print("  → Marked as FAIL")
        elif choice == "s":
            n_skip += 1
            print("  → Skipped (still pending)")
        elif choice == "q":
            print("\n  Quitting early — saving progress...")
            save_db(df, db_path)
            print(f"  Saved → {db_path}")
            print(f"  Session: pass={n_pass}  fail={n_fail}  skip={n_skip}  "
                  f"remaining={total - count}")
            sys.exit(0)

        if args.autosave and choice in ("p", "f"):
            save_db(df, db_path)

    # All pending reviewed
    save_db(df, db_path)
    print(f"\n{DIVIDER}")
    print(f"  Review complete!  pass={n_pass}  fail={n_fail}  skip={n_skip}")
    print(f"  Saved → {db_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()

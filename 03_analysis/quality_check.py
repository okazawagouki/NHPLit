"""
03_analysis/quality_check.py
=============================
Random spot-check of AI annotations in 03_annotated_db.xlsx.

Shows brain_areas and functions for randomly sampled papers.
You can confirm or overwrite each field. Corrections are saved back
to 03_annotated_db.xlsx.

For each paper:
  - Press Enter         → keep current annotation
  - Type new terms      → semicolon-separated, replaces current annotation
  - Type "other"        → marks brain_areas as other
  - Type "unknown"      → marks functions as unknown
  - Type "s"            → skip this paper entirely
  - Type "q"            → quit and save

Usage
-----
    python 03_analysis/quality_check.py           # 20 random papers
    python 03_analysis/quality_check.py -n 50
    python 03_analysis/quality_check.py --seed 7  # reproducible sample
    python 03_analysis/quality_check.py --errors  # review only ann_error rows
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd
import json
import yaml

DIVIDER = "─" * 72
INDENT  = "  "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: Database not found: {path}")
    return pd.read_excel(path, dtype=str)


def save_db(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False)


def show_vocabulary(candidates: dict) -> None:
    """Print the vocabulary compactly as a two-column reference."""
    print(f"\n{INDENT}── Brain areas ──────────────────────────────────────────────")
    areas = candidates["brain_areas"]
    for i in range(0, len(areas), 6):
        print(INDENT + "  ".join(f"{a:<12}" for a in areas[i:i+6]))
    print(f"\n{INDENT}── Functions ────────────────────────────────────────────────")
    funcs = candidates["functions"]
    for i in range(0, len(funcs), 4):
        print(INDENT + "  ".join(f"{f:<20}" for f in funcs[i:i+4]))
    print()


def show_paper(idx: int, total: int, row: pd.Series) -> None:
    print(f"\n{DIVIDER}")
    print(f"  [{idx}/{total}]  PMID: {row.get('pmid', 'N/A')}  "
          f"|  Year: {row.get('year', '?')}  "
          f"|  Error: {row.get('ann_error', 'False')}")
    print(DIVIDER)

    title = row.get("title") or "No title"
    title = title if isinstance(title, str) else "No title"
    print(textwrap.fill(f"TITLE   : {title}", width=72,
                        subsequent_indent=" " * 10))

    journal = row.get("journal") or ""
    journal = journal if isinstance(journal, str) else ""
    print(f"JOURNAL : {journal}")

    abstract = row.get("abstract") or ""
    abstract = abstract if isinstance(abstract, str) else "No abstract"
    abstract = " ".join(abstract.split())
    print()
    print(f"  {abstract}")
    print()

    print(f"  BRAIN AREAS : {row.get('brain_areas', '')}")
    print(f"  FUNCTIONS   : {row.get('functions', '')}")
    print()


def prompt_field(label: str, current: str) -> str | None:
    """
    Prompt for one field. Returns:
      None           → keep current (user pressed Enter)
      str            → new value typed by user
      "SKIP"         → user typed s
      "QUIT"         → user typed q
    """
    raw = input(f"  {label} [Enter=keep, s=skip, q=quit]: ").strip()
    if raw == "":
        return None
    if raw.lower() == "s":
        return "SKIP"
    if raw.lower() == "q":
        return "QUIT"
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spot-check annotations in 03_annotated_db.xlsx."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("-n", type=int, default=20, metavar="N",
                        help="Number of papers to sample (default: 20)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--errors", action="store_true",
                        help="Review only rows where ann_error=True")
    parser.add_argument("--vocab", action="store_true",
                        help="Show full vocabulary before starting")
    parser.add_argument("--autosave", action="store_true",
                        help="Save after every correction")
    args = parser.parse_args()

    cfg       = yaml.safe_load(open(args.config))
    db_path   = repo_root / cfg["paths"]["annotated_db"]
    cand_path = repo_root / cfg["paths"]["candidates"]

    df         = load_db(db_path)
    candidates = json.loads(cand_path.read_text())

    # Build sample pool
    if args.errors:
        pool = df[df["ann_error"].str.lower() == "true"]
        print(f"\nReviewing {len(pool)} error rows.")
    else:
        pool = df

    total_pool = len(pool)
    if total_pool == 0:
        print("No papers to review.")
        sys.exit(0)

    n      = min(args.n, total_pool)
    sample = pool.sample(n=n, random_state=args.seed)

    print(f"\nSpot-checking {n} randomly sampled paper(s) from {total_pool}.")
    print("For each field: Enter=keep, type new value=replace, s=skip paper, q=quit.")
    print("Values are semicolon-separated (e.g. 'V1; MT; LIP').\n")

    if args.vocab:
        show_vocabulary(candidates)

    n_changed = 0
    changes   = []

    for count, (row_idx, row) in enumerate(sample.iterrows(), start=1):
        show_paper(count, n, row)

        # --- brain_areas ---
        old_areas = row.get("brain_areas", "")
        resp = prompt_field("brain_areas", old_areas)
        if resp == "QUIT":
            print("\n  Quitting — saving progress...")
            save_db(df, db_path)
            print(f"  Saved → {db_path}")
            print(f"  Session: changed={n_changed}  remaining={n - count}")
            sys.exit(0)
        if resp == "SKIP":
            print("  → Skipped")
            continue
        if resp is not None:
            df.at[row_idx, "brain_areas"] = resp
            # Clear error flag if manually corrected
            df.at[row_idx, "ann_error"] = "False"
            changes.append((row.get("pmid", "?"), "brain_areas", old_areas, resp))
            n_changed += 1
            print(f"  → brain_areas updated")

        # --- functions ---
        old_funcs = row.get("functions", "")
        resp = prompt_field("functions ", old_funcs)
        if resp == "QUIT":
            print("\n  Quitting — saving progress...")
            save_db(df, db_path)
            print(f"  Saved → {db_path}")
            print(f"  Session: changed={n_changed}  remaining={n - count}")
            sys.exit(0)
        if resp == "SKIP":
            print("  → Skipped")
            continue
        if resp is not None:
            df.at[row_idx, "functions"] = resp
            df.at[row_idx, "ann_error"] = "False"
            changes.append((row.get("pmid", "?"), "functions", old_funcs, resp))
            n_changed += 1
            print(f"  → functions updated")

        if args.autosave and n_changed:
            save_db(df, db_path)

    save_db(df, db_path)
    print(f"\n{DIVIDER}")
    print(f"  QC complete!  checked={n}  fields changed={n_changed}")
    if changes:
        print(f"\n  Corrections:")
        for pmid, field, old, new in changes:
            print(f"    PMID {pmid} [{field}]")
            print(f"      was : {old}")
            print(f"      now : {new}")
    print(f"\n  Saved → {db_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()

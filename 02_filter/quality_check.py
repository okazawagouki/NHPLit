"""
02_filter/quality_check.py
==========================
Random spot-check of AI-flagged pass/fail items to assess AI quality.

Randomly samples papers from 'pass' and/or 'fail' flags and shows them
one by one. You can correct the flag if the AI got it wrong.

Keys during review:
    p  → set to pass
    f  → set to fail
    s  → skip (keep current flag)
    q  → quit and save

Usage
-----
    python 02_filter/quality_check.py                  # 20 random from pass+fail
    python 02_filter/quality_check.py -n 50            # sample 50
    python 02_filter/quality_check.py --flag pass      # only check pass items
    python 02_filter/quality_check.py --flag fail      # only check fail items
    python 02_filter/quality_check.py --seed 42        # reproducible sample
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

DIVIDER = "─" * 72


def load_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: Database not found: {path}")
    return pd.read_excel(path, dtype=str)


def save_db(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False)


def show_paper(idx: int, total: int, row: pd.Series) -> None:
    print(f"\n{DIVIDER}")
    print(f"  [{idx}/{total}]  PMID: {row.get('pmid', 'N/A')}  "
          f"| Current flag: {row.get('flag', '?').upper()}")
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


def prompt_choice(current_flag: str) -> str:
    while True:
        raw = input("  [p]ass / [f]ail / [s]kip / [q]uit : ").strip().lower()
        if raw in ("p", "f", "s", "q"):
            return raw
        print("  → Please enter p, f, s, or q.")


def main():
    parser = argparse.ArgumentParser(
        description="Spot-check AI pass/fail decisions for quality control."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config", default=repo_root / "config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "-n", type=int, default=20, metavar="N",
        help="Number of papers to sample (default: 20)"
    )
    parser.add_argument(
        "--flag", choices=["pass", "fail", "both"], default="both",
        help="Which flags to sample from (default: both)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--autosave", action="store_true",
        help="Save after every correction"
    )
    args = parser.parse_args()

    cfg     = yaml.safe_load(open(args.config))
    db_path = repo_root / cfg["paths"]["filtered_db"]
    df      = load_db(db_path)

    # Build pool
    if args.flag == "both":
        pool = df[df["flag"].isin(["pass", "fail"])]
    else:
        pool = df[df["flag"] == args.flag]

    total_pool = len(pool)
    if total_pool == 0:
        print(f"No items with flag='{args.flag}' found.")
        sys.exit(0)

    n = min(args.n, total_pool)
    sample = pool.sample(n=n, random_state=args.seed)

    print(f"\nSpot-checking {n} randomly sampled paper(s) "
          f"(pool: {total_pool} pass/fail items).")
    if args.flag == "both":
        n_pass_pool = (pool["flag"] == "pass").sum()
        n_fail_pool = (pool["flag"] == "fail").sum()
        print(f"Pool breakdown: pass={n_pass_pool}  fail={n_fail_pool}")
    print("Keys: [p]ass  [f]ail  [s]kip  [q]uit\n")

    n_changed = n_skip = 0
    changes = []  # track what changed for summary

    for count, (row_idx, row) in enumerate(sample.iterrows(), start=1):
        show_paper(count, n, row)
        current = row.get("flag", "")
        choice  = prompt_choice(current)

        if choice == "q":
            print("\n  Quitting early — saving progress...")
            save_db(df, db_path)
            print(f"  Saved → {db_path}")
            print(f"  Session: changed={n_changed}  skip={n_skip}  "
                  f"remaining={n - count}")
            sys.exit(0)
        elif choice == "s":
            n_skip += 1
            print(f"  → Kept as {current.upper()}")
        else:
            new_flag = "pass" if choice == "p" else "fail"
            if new_flag != current:
                df.at[row_idx, "flag"] = new_flag
                changes.append((row.get("pmid", "?"), current, new_flag))
                n_changed += 1
                print(f"  → Changed {current.upper()} → {new_flag.upper()}")
            else:
                n_skip += 1
                print(f"  → Confirmed as {new_flag.upper()} (no change)")

            if args.autosave and new_flag != current:
                save_db(df, db_path)

    save_db(df, db_path)
    print(f"\n{DIVIDER}")
    print(f"  QC complete!  checked={n}  changed={n_changed}  skip={n_skip}")
    if changes:
        print(f"\n  Corrections made:")
        for pmid, old, new in changes:
            print(f"    PMID {pmid}: {old.upper()} → {new.upper()}")
    print(f"\n  Saved → {db_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()

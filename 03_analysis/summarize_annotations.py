"""
03_analysis/summarize_annotations.py
=====================================
Quick summary of annotation progress in 03_annotated_db.xlsx.
Run anytime to check how many papers have been annotated.

Usage
-----
    python 03_analysis/summarize_annotations.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg       = yaml.safe_load(open(repo_root / "config.yaml"))

    in_path  = repo_root / cfg["paths"]["filtered_db2"]
    out_path = repo_root / cfg["paths"]["annotated_db"]

    print(f"\n{'='*60}")
    print(f"  Annotation progress summary")
    print(f"{'='*60}")

    # Load input (02_filtered_db)
    if not in_path.exists():
        print(f"  Input not found: {in_path.name}")
        print("  Run 02_filter/build_filtered_db.py first.\n")
        return
    df_in = pd.read_excel(in_path, dtype=str)
    print(f"  02_filtered_db total papers : {len(df_in)}")

    # Load output (03_annotated_db)
    if not out_path.exists():
        print(f"  03_annotated_db not found — nothing annotated yet.")
        print("  Run 03_analysis/annotate_papers.py to start.\n")
        return
    df_out = pd.read_excel(out_path, dtype=str)

    n_total      = len(df_in)
    n_annotated  = len(df_out)
    n_remaining  = n_total - n_annotated
    n_error      = (df_out["ann_error"].str.lower() == "true").sum() \
                   if "ann_error" in df_out.columns else 0
    n_ok         = n_annotated - n_error
    pct          = f"{100 * n_annotated // n_total}%" if n_total else "-"

    print(f"  Annotated so far            : {n_annotated} / {n_total}  ({pct})")
    print(f"    clean                     : {n_ok}")
    print(f"    ann_error                 : {n_error}")
    print(f"  Remaining                   : {n_remaining}")

    # Per-year breakdown
    annotated_pmids = set(df_out["pmid"].tolist())
    per_year_total  = Counter(df_in["year"].tolist())
    per_year_done   = Counter(
        row["year"] for _, row in df_out.iterrows()
        if row.get("ann_error", "False").lower() != "true"
    )
    per_year_error  = Counter(
        row["year"] for _, row in df_out.iterrows()
        if row.get("ann_error", "False").lower() == "true"
    )

    all_years = sorted(per_year_total.keys(), reverse=True)
    print(f"\n  Per-year progress:")
    print(f"    {'Year':>6}  {'Total':>7}  {'Done':>6}  {'Error':>6}  {'Left':>6}  {'Done%':>6}")
    print(f"    {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for yr in all_years:
        total = per_year_total[yr]
        done  = per_year_done[yr]
        err   = per_year_error[yr]
        left  = total - done - err
        pct_y = f"{100 * (done + err) // total}%" if total else "-"
        print(f"    {yr:>6}  {total:>7}  {done:>6}  {err:>6}  {left:>6}  {pct_y:>6}")

    # Most common brain areas and functions
    if n_annotated > 0:
        area_counts = Counter()
        func_counts = Counter()
        for _, row in df_out.iterrows():
            for a in str(row.get("brain_areas", "") or "").split(";"):
                a = a.strip()
                if a and a not in ("other", "error", ""):
                    area_counts[a] += 1
            for f in str(row.get("functions", "") or "").split(";"):
                f = f.strip()
                if f and f not in ("unknown", "error", ""):
                    func_counts[f] += 1

        print(f"\n  Top 15 brain areas:")
        for area, n in area_counts.most_common(15):
            bar = "█" * (n * 30 // max(area_counts.values()))
            print(f"    {n:5d}  {area:<28}  {bar}")

        print(f"\n  Top 15 functions:")
        for func, n in func_counts.most_common(15):
            bar = "█" * (n * 30 // max(func_counts.values()))
            print(f"    {n:5d}  {func:<28}  {bar}")

    print()


if __name__ == "__main__":
    main()

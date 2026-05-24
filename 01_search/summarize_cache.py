"""
01_search/summarize_cache.py
============================
Quick summary of what is currently in the raw cache and filter progress.
Run this anytime to check progress.

Usage
-----
    python 01_search/summarize_cache.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg       = yaml.safe_load(open(repo_root / "config.yaml"))
    cache_dir = repo_root / cfg["paths"]["raw_cache"]
    db_path   = repo_root / cfg["paths"]["filtered_db"]
    log_path  = repo_root / cfg["paths"]["pmid_log"]

    files = list(cache_dir.rglob("*.json"))
    print(f"\n{'='*60}")
    print(f"  Raw cache summary")
    print(f"{'='*60}")
    print(f"  Total papers cached : {len(files)}")

    if not files:
        print("  (cache is empty — run fetch_pubmed.py first)\n")
        return

    # Count cached papers per year from directory structure
    # pmid -> folder_year mapping (used to align filter results)
    pmid_to_folder_year = {}
    cached_per_year = Counter()
    missing_abstract = 0
    journals = Counter()
    for fp in files:
        yr = fp.parent.name  # raw_cache/<year>/<pmid>.json
        pmid_to_folder_year[fp.stem] = yr
        cached_per_year[yr] += 1
        try:
            rec = json.loads(fp.read_text())
            journals[rec.get("journal", "?")] += 1
            if not rec.get("abstract"):
                missing_abstract += 1
        except Exception:
            pass

    print(f"  Missing abstract    : {missing_abstract}")

    # Load filter results — use folder year (not JSON year) for consistency with --years flag
    filtered_per_year = defaultdict(Counter)  # folder_year -> {flag -> count}
    if db_path.exists():
        try:
            import pandas as pd
            df = pd.read_excel(db_path, dtype=str)
            for _, row in df.iterrows():
                pmid = str(row.get("pmid", ""))
                flag = str(row.get("flag", "?"))
                yr   = pmid_to_folder_year.get(pmid, "?")
                filtered_per_year[yr][flag] += 1
        except Exception:
            pass

    # Per-year table
    all_years = sorted(set(cached_per_year.keys()) | set(filtered_per_year.keys()), reverse=True)
    print(f"\n  Per-year progress:")
    print(f"    {'Year':>6}  {'Cached':>7}  {'Pass':>6}  {'Fail':>6}  {'Pending':>8}  {'Error':>6}  {'Done%':>6}")
    print(f"    {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")
    for yr in all_years:
        cached  = cached_per_year.get(yr, 0)
        f       = filtered_per_year.get(yr, {})
        n_pass  = f.get("pass", 0)
        n_fail  = f.get("fail", 0)
        n_pend  = f.get("pending", 0)
        n_err   = f.get("error", 0)
        n_done  = n_pass + n_fail + n_pend + n_err
        pct     = f"{100*n_done//cached}%" if cached else "-"
        print(f"    {yr:>6}  {cached:>7}  {n_pass:>6}  {n_fail:>6}  {n_pend:>8}  {n_err:>6}  {pct:>6}")

    total_filtered = sum(sum(v.values()) for v in filtered_per_year.values())
    print(f"\n  Total filtered: {total_filtered} / {len(files)}")
    if total_filtered:
        all_flags = Counter()
        for fc in filtered_per_year.values():
            all_flags.update(fc)
        print(f"  pass={all_flags['pass']}  fail={all_flags['fail']}  pending={all_flags['pending']}  error={all_flags['error']}")

    print(f"\n  Top 10 journals:")
    for j, n in journals.most_common(10):
        print(f"    {n:5d}  {j}")

    if log_path.exists():
        import csv
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        query_counts = Counter(r["query_label"] for r in rows)
        print(f"\n  Top 10 most productive queries:")
        for label, n in query_counts.most_common(10):
            print(f"    {n:5d}  {label}")

    print()


if __name__ == "__main__":
    main()

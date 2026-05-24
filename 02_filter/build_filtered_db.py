"""
02_filter/build_filtered_db.py
==============================
Stage 3: Convert 01_extracted_db.xlsx → 02_filtered_db.xlsx.

Steps
-----
1. Keep only flag == 'pass' papers.
2. Remove papers with no abstract (empty or whitespace only).
3. Deduplicate bioRxiv / arXiv preprints:
   a. Same title (normalised) exists in a later year → remove preprint.
   b. Same first author AND overlapping last author exist within ±2 years
      in a non-preprint journal → remove preprint.
4. Report what was removed and why.

Usage
-----
    python 02_filter/build_filtered_db.py
    python 02_filter/build_filtered_db.py --dry-run   # report only, no file written
"""

import argparse
import re
import string
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PREPRINT_JOURNALS = re.compile(
    r"biorxiv|medrxiv|arxiv|preprint", re.IGNORECASE
)


def is_preprint(journal: str) -> bool:
    if not isinstance(journal, str):
        return False
    return bool(PREPRINT_JOURNALS.search(journal))


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = title.translate(str.maketrans("", "", string.punctuation))
    title = " ".join(title.split())
    return title


def parse_authors(authors_str: str) -> list[str]:
    """Split 'Last FM; Last FM; ...' into a list of normalised names."""
    if not isinstance(authors_str, str):
        return []
    return [a.strip().lower() for a in authors_str.split(";") if a.strip()]


def first_author(authors: list[str]) -> str:
    return authors[0] if authors else ""


def last_author(authors: list[str]) -> str:
    return authors[-1] if authors else ""


def safe_int(val) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------
def find_preprint_duplicates(df: pd.DataFrame) -> dict[int, str]:
    """
    Returns {row_index: reason_string} for preprint rows to drop.

    Rules
    -----
    A preprint is dropped if, among the remaining pass papers, there exists
    a non-preprint paper where:
      (a) normalised titles match exactly, OR
      (b) first author AND last author both match AND |year_diff| <= 2
    """
    preprint_mask = df["journal"].apply(is_preprint)
    preprints     = df[preprint_mask]
    published     = df[~preprint_mask]

    # Pre-compute lookup structures on published set
    pub_norm_titles = {
        normalize_title(row["title"]): idx
        for idx, row in published.iterrows()
        if isinstance(row.get("title"), str)
    }

    to_drop: dict[int, str] = {}

    for idx, row in preprints.iterrows():
        year_pre   = safe_int(row.get("year"))
        norm_title = normalize_title(row.get("title", ""))
        authors    = parse_authors(row.get("authors", ""))
        fa         = first_author(authors)
        la         = last_author(authors)

        # Rule (a): exact normalised title match in published set
        if norm_title and norm_title in pub_norm_titles:
            pub_idx  = pub_norm_titles[norm_title]
            pub_year = safe_int(df.at[pub_idx, "year"])
            # Only remove preprint if published paper is same or later year
            if year_pre is None or pub_year is None or pub_year >= year_pre:
                pub_title = df.at[pub_idx, "title"]
                to_drop[idx] = (
                    f"title match → PMID {df.at[pub_idx, 'pmid']} "
                    f"({pub_year}): '{pub_title[:60]}'"
                )
                continue

        # Rule (b): first+last author match; preprint must be before published (up to 4 years prior)
        if not fa:
            continue
        for pub_idx, pub_row in published.iterrows():
            pub_year    = safe_int(pub_row.get("year"))
            if year_pre is not None and pub_year is not None:
                if not (0 <= pub_year - year_pre <= 4):
                    continue
            pub_authors = parse_authors(pub_row.get("authors", ""))
            pub_fa      = first_author(pub_authors)
            pub_la      = last_author(pub_authors)
            if fa == pub_fa and la == pub_la and fa != "":
                to_drop[idx] = (
                    f"author match (1st='{fa}', last='{la}') within 4 yr → "
                    f"PMID {pub_row.get('pmid', '?')} ({pub_year})"
                )
                break

    return to_drop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build 02_filtered_db.xlsx from 01_extracted_db.xlsx."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config", default=repo_root / "config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be removed without writing output file"
    )
    args = parser.parse_args()

    cfg      = yaml.safe_load(open(args.config))
    in_path  = repo_root / cfg["paths"]["filtered_db"]
    out_path = repo_root / cfg["paths"]["filtered_db2"]

    # ------------------------------------------------------------------
    # Step 1: load and filter to pass only
    # ------------------------------------------------------------------
    print(f"Loading: {in_path}")
    df_all = pd.read_excel(in_path, dtype=str)
    print(f"  Total rows in 01_extracted_db: {len(df_all)}")

    df = df_all[df_all["flag"] == "pass"].copy().reset_index(drop=True)
    print(f"  Pass papers:                   {len(df)}")

    # ------------------------------------------------------------------
    # Step 2: remove papers with no abstract
    # ------------------------------------------------------------------
    has_abstract = df["abstract"].apply(
        lambda x: isinstance(x, str) and x.strip() != ""
    )
    n_no_abstract = (~has_abstract).sum()
    print(f"  No abstract (removed):         {n_no_abstract}")
    df = df[has_abstract].reset_index(drop=True)

    n_preprints = df["journal"].apply(is_preprint).sum()
    print(f"  of which preprints:            {n_preprints}")

    # ------------------------------------------------------------------
    # Step 3: find duplicate preprints
    # ------------------------------------------------------------------
    to_drop = find_preprint_duplicates(df)
    print(f"\nPreprint duplicates to remove:  {len(to_drop)}")

    if to_drop:
        print("\n  Dropped preprints:")
        for idx, reason in to_drop.items():
            pmid  = df.at[idx, "pmid"]
            title = str(df.at[idx, "title"])[:60]
            year  = df.at[idx, "year"]
            print(f"    PMID {pmid} ({year}) '{title}' — {reason}")

    # ------------------------------------------------------------------
    # Step 4: build output
    # ------------------------------------------------------------------
    df_out = df.drop(index=list(to_drop.keys())).reset_index(drop=True)
    print(f"\nFinal count in 02_filtered_db: {len(df_out)}")

    if args.dry_run:
        print("\n--dry-run: no file written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

"""
01_search/fetch_pubmed.py
=========================
Stage 1: fetch paper metadata from PubMed using a single combined query
         covering all keyword combinations and a date range from config.yaml.

Usage
-----
    python 01_search/fetch_pubmed.py                     # current year only (safe default)
    python 01_search/fetch_pubmed.py --all               # full range from config.yaml
    python 01_search/fetch_pubmed.py --years 2020 2024  # explicit subset of years
    python 01_search/fetch_pubmed.py --dry-run           # print queries only

Output
------
    data/raw_cache/<YEAR>/<PMID>.json   one file per unique paper, sorted by year
    data/pmid_log.csv            running log of all fetched PMIDs

Dependencies
------------
    pip install biopython pyyaml tqdm
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import date, datetime
from itertools import product
from pathlib import Path

import yaml
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def build_combined_query(cfg: dict) -> str:
    """
    Combine all keyword combinations and additional queries into a single
    PubMed OR query, e.g.:
        ((macaque[tiab]) AND (neuron[tiab])) OR ((macaque[tiab]) AND (neural[tiab])) OR ...
    Exclusion terms are appended once as a NOT clause.
    """
    kw = cfg["keywords"]

    exclude_clause = ""
    if kw.get("exclude"):
        parts = " OR ".join(kw["exclude"])
        exclude_clause = f" NOT ({parts})"

    parts = []

    # species × signal combinations
    for species_term, signal_term in product(kw["species"], kw["signal"]):
        parts.append(f'(({species_term}) AND ({signal_term}))')

    # free-form additional queries
    for aq in (kw.get("additional_queries") or []):
        parts.append(f'({aq})')

    combined = " OR ".join(parts)
    return f'({combined}){exclude_clause}'


def add_year_filter(query: str, year: int) -> str:
    return f'({query}) AND ("{year}/01/01"[dp] : "{year}/12/31"[dp])'


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

def search_pmids(query: str, max_results: int, retries: int) -> list[str]:
    """Run esearch and return a list of PMIDs."""
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="pubmed", term=query,
                                    retmax=max_results, usehistory="n")
            record = Entrez.read(handle)
            handle.close()
            return record["IdList"]
        except Exception as e:
            log.warning(f"esearch attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return []


def fetch_records_batch(pmids: list[str], retries: int) -> dict[str, dict]:
    """
    Fetch full metadata for a batch of PMIDs in a single efetch call.
    Returns a dict of {pmid: parsed_record}; failed PMIDs are omitted.
    """
    id_str = ",".join(pmids)
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(db="pubmed", id=id_str,
                                   rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            result = {}
            for article in records["PubmedArticle"]:
                pmid = str(article["MedlineCitation"]["PMID"])
                result[pmid] = parse_article(article, pmid)
            return result
        except Exception as e:
            log.warning(f"efetch batch attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return {}


def parse_article(article: dict, pmid: str) -> dict:
    """Extract the fields we care about from a parsed PubMed XML record."""
    medline = article.get("MedlineCitation", {})
    art     = medline.get("Article", {})

    # Title
    title = str(art.get("ArticleTitle", ""))

    # Abstract (may be structured with multiple sections)
    abstract_parts = []
    abstract_obj = art.get("Abstract", {})
    for part in abstract_obj.get("AbstractText", []):
        label = getattr(part, "attributes", {}).get("Label", "")
        text  = str(part)
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(abstract_parts)

    # Authors
    authors = []
    for a in art.get("AuthorList", []):
        last  = str(a.get("LastName", ""))
        first = str(a.get("ForeName", ""))
        authors.append(f"{last} {first}".strip())

    # Journal + year
    journal_info = art.get("Journal", {})
    journal = str(journal_info.get("Title", ""))
    pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
    year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

    # MeSH terms
    mesh_list = medline.get("MeshHeadingList", [])
    mesh_terms = [str(m["DescriptorName"]) for m in mesh_list]

    # DOI
    doi = ""
    for id_obj in art.get("ELocationID", []):
        if getattr(id_obj, "attributes", {}).get("EIdType") == "doi":
            doi = str(id_obj)
            break

    return {
        "pmid":     pmid,
        "title":    title,
        "abstract": abstract,
        "authors":  authors,
        "journal":  journal,
        "year":     year,
        "doi":      doi,
        "mesh":     mesh_terms,
    }


# ---------------------------------------------------------------------------
# PMID log helpers
# ---------------------------------------------------------------------------

LOG_FIELDS = ["pmid", "query_label", "year", "date_fetched"]


def load_existing_pmids(log_path: Path) -> set[str]:
    """Return set of PMIDs already in the log (= already cached)."""
    if not log_path.exists():
        return set()
    with open(log_path, newline="") as f:
        return {row["pmid"] for row in csv.DictReader(f)}


def append_pmid_log(log_path: Path, rows: list[dict]) -> None:
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def run_fetch(cfg: dict, years: list[int], dry_run: bool) -> None:
    # Load credentials from .env (never committed to git)
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / "key.env")

    email   = os.environ.get("NCBI_EMAIL", "")
    api_key = os.environ.get("NCBI_API_KEY", "")

    if not email or email == "your_email@example.com":
        raise SystemExit(
            "ERROR: NCBI_EMAIL not set.\n"
            "  Copy example.env to key.env and fill in your email address."
        )

    # Configure Entrez
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    max_records    = cfg["entrez"]["max_records_per_query"]
    batch_size     = cfg["entrez"]["fetch_batch_size"]
    rps            = cfg["entrez"]["requests_per_second"]
    retries        = cfg["entrez"]["retries"]
    delay          = 1.0 / rps

    raw_cache_dir  = Path(cfg["paths"]["raw_cache"])
    pmid_log_path  = Path(cfg["paths"]["pmid_log"])
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    base_query = build_combined_query(cfg)
    log.info(f"Query ({len(base_query)} chars): {base_query[:120]}...")
    log.info(f"Years to fetch: {min(years)}–{max(years)} ({len(years)} year(s))")

    if dry_run:
        log.info("--- DRY RUN: printing query for first year ---")
        print(add_year_filter(base_query, years[0]))
        return

    existing_pmids = load_existing_pmids(pmid_log_path)
    log.info(f"Already cached: {len(existing_pmids)} PMIDs")

    today     = date.today().isoformat()
    total_new = 0

    for year in years:
        log.info(f"=== Year {year} ===")
        full_query = add_year_filter(base_query, year)
        pmids = search_pmids(full_query, max_records, retries)
        time.sleep(delay)
        log.info(f"  PubMed returned {len(pmids)} PMIDs")

        if len(pmids) >= max_records:
            log.warning(
                f"  Hit the max_records_per_query cap ({max_records}) for {year}. "
                "Some results may be missing — consider raising max_records_per_query in config.yaml."
            )

        new_pmids = [p for p in pmids if p not in existing_pmids]
        if not new_pmids:
            log.info(f"  No new PMIDs for {year}, skipping.")
            continue

        year_cache_dir = raw_cache_dir / str(year)
        year_cache_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []

        batches = [new_pmids[i:i + batch_size]
                   for i in range(0, len(new_pmids), batch_size)]

        for batch in tqdm(batches, desc=f"{year}", unit="batch", leave=False):
            records = fetch_records_batch(batch, retries)
            time.sleep(delay)
            for pmid in batch:
                record = records.get(pmid)
                cache_file = year_cache_dir / f"{pmid}.json"
                if record and not cache_file.exists():
                    with open(cache_file, "w") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)

                existing_pmids.add(pmid)
                log_rows.append({
                    "pmid":         pmid,
                    "query_label":  "combined",
                    "year":         year,
                    "date_fetched": today,
                })

        append_pmid_log(pmid_log_path, log_rows)
        log.info(f"  {len(new_pmids)} new PMIDs added for {year}")
        total_new += len(new_pmids)

    log.info(f"Done. Total new PMIDs this run: {total_new}")
    log.info(f"Total unique PMIDs in cache:    {len(existing_pmids)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Fetch monkey ephys paper metadata from PubMed."
    )
    parser.add_argument(
        "--config", default=repo_root / "config.yaml",
        help="Path to config.yaml (default: repo root)"
    )
    parser.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"),
        help="Fetch a specific year range, e.g. --years 2020 2024"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch the full year range defined in config.yaml (1970-present)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print queries without hitting PubMed"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    elif args.all:
        y = cfg["years"]
        years = list(range(y["start"], y["end"] + 1))
    else:
        current_year = datetime.now().year
        years = [current_year]

    run_fetch(cfg, years, args.dry_run)


if __name__ == "__main__":
    main()

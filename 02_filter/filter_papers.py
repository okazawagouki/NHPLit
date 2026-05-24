"""
02_filter/filter_papers.py
==========================
Stage 2: AI-based relevance screening of cached PubMed papers.

For each new PMID in the raw cache (not yet in filtered_db.xlsx), asks Claude
whether the paper reports primary electrophysiology recordings from macaque
monkeys. Results are appended to filtered_db.xlsx with one of three flags:
    pass    — Claude confident it is relevant
    fail    — Claude confident it is not relevant
    pending — low confidence; needs human review

Already-processed PMIDs are skipped, so this script is safe to re-run as the
cache grows. Human edits to flags in filtered_db.xlsx are preserved.

Usage
-----
    python 02_filter/filter_papers.py           # process all new cache entries
    python 02_filter/filter_papers.py --dry-run # print stats without calling Claude
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import anthropic
import openai
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

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
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are screening scientific papers for a macaque electrophysiology database.
Respond with JSON only — no extra text.
"""

USER_TEMPLATE = """\
Title: {title}

Abstract: {abstract}

Does this paper report PRIMARY electrophysiology recordings (single-unit, \
multi-unit, or LFP) from macaque monkeys (in vivo)?
Exclude: review articles, fMRI/EEG-only studies, behavioral-only studies, \
in-vitro recordings.

Respond with JSON only:
{{"relevant": true or false, "confidence": 0.0 to 1.0}}"""

# ---------------------------------------------------------------------------
# xlsx helpers
# ---------------------------------------------------------------------------
XLSX_COLUMNS = [
    "pmid", "title", "abstract", "authors", "journal",
    "year", "doi", "mesh", "ai_relevant", "ai_confidence", "flag",
]


def load_filtered_db(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_excel(path, dtype=str)
    return pd.DataFrame(columns=XLSX_COLUMNS)


def save_filtered_db(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# LLM call (provider-agnostic)
# ---------------------------------------------------------------------------
async def call_llm(provider: str, client, model: str, prompt: str) -> str:
    """Send a prompt to the configured LLM and return raw text."""
    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip() if response.content else ""
        if not raw:
            raise ValueError(f"Empty response from model (stop_reason={response.stop_reason})")
        return raw
    else:  # openai_compat
        response = await client.chat.completions.create(
            model=model,
            max_tokens=64,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("Empty response from model")
        return raw


async def classify_paper(
    provider: str,
    client,
    record: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Classify one paper. Returns the record enriched with AI fields."""
    prompt = USER_TEMPLATE.format(
        title    = record.get("title", ""),
        abstract = record.get("abstract", ""),
    )

    async with semaphore:
        try:
            raw    = await call_llm(provider, client, model, prompt)
            log.debug(f"PMID {record['pmid']}: raw response = {raw!r}")
            result = json.loads(raw)
            relevant   = bool(result.get("relevant", False))
            confidence = float(result.get("confidence", 0.0))
        except (anthropic.AuthenticationError, openai.AuthenticationError) as e:
            raise SystemExit(f"ERROR: Authentication failed — check your API key in key.env\n({e})")
        except Exception as e:
            raw_repr = repr(raw) if "raw" in locals() else "N/A"
            log.warning(f"PMID {record['pmid']}: LLM call failed ({e!r}), raw={raw_repr}, flagging as error")
            relevant   = False
            confidence = 0.0
            return {**record, "ai_relevant": relevant, "ai_confidence": confidence, "ai_error": True}

    return {**record, "ai_relevant": relevant, "ai_confidence": confidence, "ai_error": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_filter(cfg: dict, dry_run: bool, years: list[int] | None = None) -> None:
    repo_root   = Path(__file__).resolve().parent.parent
    cache_dir   = repo_root / cfg["paths"]["raw_cache"]
    db_path     = repo_root / cfg["paths"]["filtered_db"]
    model       = cfg["filter"]["ai_model"]
    threshold   = cfg["filter"]["confidence_threshold"]
    concurrency = cfg["filter"]["concurrency"]

    # Load existing results — preserve human edits
    # "error" flagged entries are retried on the next run; all others are skipped
    existing_df  = load_filtered_db(db_path)
    error_pmids  = set(existing_df.loc[existing_df["flag"] == "error", "pmid"].tolist())
    done_pmids   = set(existing_df["pmid"].tolist()) - error_pmids
    if error_pmids:
        log.info(f"Will retry {len(error_pmids)} previously errored PMIDs")
        existing_df = existing_df[existing_df["flag"] != "error"]  # remove old error rows
    log.info(f"Already filtered: {len(done_pmids)} PMIDs")

    # Find all cached records not yet filtered
    # If --years specified, restrict to those year subdirectories
    if years:
        search_dirs = [cache_dir / str(y) for y in years if (cache_dir / str(y)).exists()]
        missing = [y for y in years if not (cache_dir / str(y)).exists()]
        if missing:
            log.warning(f"No cache found for years: {missing}")
        all_files = [f for d in search_dirs for f in d.glob("*.json")]
        log.info(f"Filtering years: {years}")
    else:
        all_files = list(cache_dir.rglob("*.json"))

    new_files = [f for f in all_files if f.stem not in done_pmids]
    log.info(f"Cache total: {len(all_files)} | New to filter: {len(new_files)}")

    if not new_files:
        log.info("Nothing new to filter. Exiting.")
        return

    if dry_run:
        log.info("--- DRY RUN: no Claude calls made ---")
        return

    # Load new records
    new_records = []
    for fp in new_files:
        try:
            new_records.append(json.loads(fp.read_text()))
        except Exception as e:
            log.warning(f"Could not read {fp}: {e}")

    # Classify concurrently
    load_dotenv(repo_root / "key.env")
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit("ERROR: ANTHROPIC_API_KEY not set in key.env")
        client = anthropic.AsyncAnthropic(api_key=api_key)

    elif provider == "openai_compat":
        api_key  = os.environ.get("OPENAI_COMPAT_API_KEY", "")
        base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
        if not api_key or not base_url:
            raise SystemExit("ERROR: OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL must be set in key.env")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    else:
        raise SystemExit(f"ERROR: Unknown LLM_PROVIDER '{provider}'. Choose 'anthropic' or 'openai_compat'.")

    log.info(f"Using provider: {provider} | model: {model}")
    semaphore  = asyncio.Semaphore(concurrency)
    chunk_size = cfg["filter"].get("checkpoint_every", 100)

    n_pass = n_fail = n_pending = n_error = 0

    # Process in chunks so results are saved incrementally
    for chunk_start in range(0, len(new_records), chunk_size):
        chunk   = new_records[chunk_start : chunk_start + chunk_size]
        chunk_n = chunk_start // chunk_size + 1
        total_chunks = (len(new_records) + chunk_size - 1) // chunk_size
        log.info(f"Chunk {chunk_n}/{total_chunks} ({len(chunk)} papers)")

        tasks   = [classify_paper(provider, client, rec, model, semaphore) for rec in chunk]
        results = await atqdm.gather(*tasks, desc=f"Chunk {chunk_n}", unit="paper")

        rows = []
        for r in results:
            conf = r["ai_confidence"]
            rel  = r["ai_relevant"]
            if r.get("ai_error", False):
                flag = "error"
                n_error += 1
            elif conf >= threshold and rel:
                flag = "pass"
                n_pass += 1
            elif conf >= threshold and not rel:
                flag = "fail"
                n_fail += 1
            else:
                flag = "pending"
                n_pending += 1

            rows.append({
                "pmid":          r.get("pmid", ""),
                "title":         r.get("title", ""),
                "abstract":      r.get("abstract", ""),
                "authors":       "; ".join(r.get("authors", [])),
                "journal":       r.get("journal", ""),
                "year":          r.get("year", ""),
                "doi":           r.get("doi", ""),
                "mesh":          "; ".join(r.get("mesh", [])),
                "ai_relevant":   r["ai_relevant"],
                "ai_confidence": round(r["ai_confidence"], 3),
                "flag":          flag,
            })

        # Save after every chunk
        new_df      = pd.DataFrame(rows, columns=XLSX_COLUMNS)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        save_filtered_db(existing_df, db_path)
        log.info(f"Checkpoint saved ({chunk_start + len(chunk)}/{len(new_records)} processed)")

    log.info(f"Done. pass={n_pass}  fail={n_fail}  pending={n_pending}  error={n_error}")
    log.info(f"Saved → {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: AI relevance filtering of cached PubMed papers."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config", default=repo_root / "config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without calling the LLM"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, metavar="YEAR",
        help="Only filter papers from these years (e.g. --years 2023 2024)"
    )
    parser.add_argument(
        "--year-range", nargs=2, type=int, metavar=("FROM", "TO"),
        help="Only filter papers in this year range inclusive (e.g. --year-range 2020 2024)"
    )
    args = parser.parse_args()

    years = args.years
    if args.year_range:
        start, end = args.year_range
        extra = list(range(start, end + 1))
        years = list(set(years or []) | set(extra))

    cfg = yaml.safe_load(open(args.config))
    asyncio.run(run_filter(cfg, args.dry_run, years=years))


if __name__ == "__main__":
    main()

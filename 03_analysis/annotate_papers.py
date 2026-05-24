"""
03_analysis/annotate_papers.py
===============================
Stage 4: Annotate each paper in 02_filtered_db.xlsx with brain areas
and cognitive functions, using the vocabulary in candidates_curated.json.

For each paper the LLM chooses zero or more terms from the provided lists.
If nothing fits, it labels the field ["other"] or ["unknown"].

Results are written to 03_annotated_db.xlsx (02_filtered_db.xlsx is never
modified). Already-annotated PMIDs are skipped, so the script is safe to
re-run incrementally.

Usage
-----
    python 03_analysis/annotate_papers.py
    python 03_analysis/annotate_papers.py --dry-run
    python 03_analysis/annotate_papers.py --years 2020 2021
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import openai
import anthropic
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
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are annotating macaque electrophysiology papers for a structured database.
Choose terms strictly from the provided vocabulary lists.
Respond with JSON only — no extra text, no markdown fences.
"""

USER_TEMPLATE = """\
Annotate the following paper using ONLY terms from the vocabulary lists below.

BRAIN AREA VOCABULARY:
{brain_areas}

FUNCTION VOCABULARY:
{functions}

Rules:
- Choose all terms that clearly apply based on the title and abstract.
- You may select multiple terms from each list.
- If no brain area is identifiable, use ["other"].
- If no function is identifiable, use ["unknown"].
- Do NOT invent terms outside the vocabulary.

Title: {title}

Abstract: {abstract}

Respond with JSON only:
{{"brain_areas": [...], "functions": [...]}}
"""


# ---------------------------------------------------------------------------
# xlsx helpers
# ---------------------------------------------------------------------------
def load_db(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_excel(path, dtype=str)
    return pd.DataFrame()


def save_db(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def call_llm(provider: str, client, model: str, prompt: str) -> str:
    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip() if response.content else ""
    else:  # openai_compat
        response = await client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            extra_body={"enable_thinking": False},
        )
        return (response.choices[0].message.content or "").strip()


async def annotate_paper(
    provider: str,
    client,
    model: str,
    record: dict,
    brain_areas_str: str,
    functions_str: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    title    = str(record.get("title", "") or "")
    abstract = " ".join(str(record.get("abstract", "") or "").split())

    prompt = USER_TEMPLATE.format(
        brain_areas = brain_areas_str,
        functions   = functions_str,
        title       = title,
        abstract    = abstract[:800],
    )

    async with semaphore:
        try:
            raw = await call_llm(provider, client, model, prompt)
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rsplit("```", 1)[0].strip()
            result      = json.loads(raw)
            brain_areas = result.get("brain_areas", ["other"])
            functions   = result.get("functions",   ["unknown"])
            if not brain_areas:
                brain_areas = ["other"]
            if not functions:
                functions = ["unknown"]
            error = False
        except Exception as e:
            log.warning(f"PMID {record.get('pmid', '?')}: failed ({e!r})")
            brain_areas = ["error"]
            functions   = ["error"]
            error = True

    return {
        **record,
        "brain_areas": "; ".join(brain_areas),
        "functions":   "; ".join(functions),
        "ann_error":   error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_annotation(cfg: dict, dry_run: bool, years: list[int] | None) -> None:
    repo_root    = Path(__file__).resolve().parent.parent
    in_path      = repo_root / cfg["paths"]["filtered_db2"]
    out_path     = repo_root / cfg["paths"]["annotated_db"]
    cand_path    = repo_root / cfg["paths"]["candidates"]
    model        = cfg["filter"]["ai_model"]
    concurrency  = cfg["filter"]["concurrency"]
    chunk_size   = cfg["filter"].get("checkpoint_every", 100)

    # Load vocabulary
    candidates      = json.loads(cand_path.read_text())
    brain_areas_str = "\n".join(f"  - {a}" for a in candidates["brain_areas"])
    functions_str   = "\n".join(f"  - {f}" for f in candidates["functions"])
    log.info(f"Vocabulary: {len(candidates['brain_areas'])} brain areas, "
             f"{len(candidates['functions'])} functions")

    # Load input
    df_in = pd.read_excel(in_path, dtype=str)
    log.info(f"Input: {len(df_in)} papers from {in_path.name}")

    # Filter by year if requested
    if years:
        df_in = df_in[df_in["year"].astype(str).isin([str(y) for y in years])]
        log.info(f"Year filter {years}: {len(df_in)} papers remaining")

    # Load existing output to skip already-annotated PMIDs
    out_cols = list(df_in.columns) + ["brain_areas", "functions", "ann_error"]
    if out_path.exists():
        df_out      = pd.read_excel(out_path, dtype=str)
        done_pmids  = set(df_out["pmid"].tolist())
        # Remove error rows so they get retried
        error_pmids = set(df_out.loc[df_out.get("ann_error", "False") == "True", "pmid"])
        done_pmids -= error_pmids
        df_out = df_out[~df_out["pmid"].isin(error_pmids)]
        log.info(f"Already annotated: {len(done_pmids)} | Will retry errors: {len(error_pmids)}")
    else:
        df_out     = pd.DataFrame(columns=out_cols)
        done_pmids = set()

    new_records = [
        row.to_dict() for _, row in df_in.iterrows()
        if row["pmid"] not in done_pmids
    ]
    log.info(f"To annotate: {len(new_records)}")

    if not new_records:
        log.info("Nothing new to annotate. Exiting.")
        return

    if dry_run:
        log.info("--- DRY RUN: no LLM calls made ---")
        return

    # Set up client
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
            raise SystemExit("ERROR: OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL must be set")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        raise SystemExit(f"ERROR: Unknown LLM_PROVIDER '{provider}'")

    log.info(f"Provider: {provider} | Model: {model}")
    semaphore = asyncio.Semaphore(concurrency)

    # Process in chunks with checkpointing
    for chunk_start in range(0, len(new_records), chunk_size):
        chunk        = new_records[chunk_start: chunk_start + chunk_size]
        chunk_n      = chunk_start // chunk_size + 1
        total_chunks = (len(new_records) + chunk_size - 1) // chunk_size
        log.info(f"Chunk {chunk_n}/{total_chunks} ({len(chunk)} papers)")

        tasks = [
            annotate_paper(provider, client, model, rec,
                           brain_areas_str, functions_str, semaphore)
            for rec in chunk
        ]
        results = await atqdm.gather(*tasks, desc=f"Chunk {chunk_n}", unit="paper")

        new_rows = pd.DataFrame(results)[out_cols]
        df_out   = pd.concat([df_out, new_rows], ignore_index=True)
        save_db(df_out, out_path)
        log.info(f"Checkpoint saved ({chunk_start + len(chunk)}/{len(new_records)} done)")

    n_error = (df_out["ann_error"] == "True").sum()
    log.info(f"Annotation complete. Total rows: {len(df_out)} | Errors: {n_error}")
    log.info(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Annotate papers with brain areas and functions."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without calling the LLM")
    parser.add_argument("--years", nargs="+", type=int, metavar="YEAR",
                        help="Only annotate papers from these years")
    parser.add_argument("--year-range", nargs=2, type=int, metavar=("FROM", "TO"),
                        help="Only annotate papers in this year range inclusive")
    args = parser.parse_args()

    years = args.years or []
    if args.year_range:
        start, end = args.year_range
        years = list(set(years) | set(range(start, end + 1)))

    cfg = yaml.safe_load(open(args.config))
    asyncio.run(run_annotation(cfg, args.dry_run, years or None))


if __name__ == "__main__":
    main()

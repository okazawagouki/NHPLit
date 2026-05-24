"""
03_analysis/patch_other_areas.py
=================================
Patch pass for papers annotated with brain_areas = "other".

Uses a more permissive prompt that extracts any brain region mentioned
in the abstract, regardless of whether the paper is a standard ephys
study (e.g. technique development, methods papers still name regions).

Updates 03_annotated_db.xlsx in place.

Usage
-----
    python 03_analysis/patch_other_areas.py
    python 03_analysis/patch_other_areas.py --dry-run
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — more permissive than annotate_papers.py
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are extracting brain region mentions from neuroscience paper abstracts.
Respond with JSON only — no extra text, no markdown fences.
"""

USER_TEMPLATE = """\
Extract every brain region or area explicitly mentioned in the abstract below.
The paper may be about technique development, methods, or instrumentation —
that is fine. Simply list any brain areas recorded.

Choose terms from this vocabulary where possible:
{brain_areas}

Rules:
- If the abstract mentions a brain region not in the vocabulary, include it
  as-is (do not force-fit to the vocabulary).
- If truly no brain region is mentioned at all, return ["other"].
- Do NOT invent regions that are not in the text.

Title: {title}

Abstract: {abstract}

Respond with JSON only:
{{"brain_areas": [...]}}
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def call_llm(provider: str, client, model: str,
                   system: str, prompt: str) -> str:
    if provider == "anthropic":
        response = await client.messages.create(
            model=model, max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip() if response.content else ""
    else:
        response = await client.chat.completions.create(
            model=model, max_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            extra_body={"enable_thinking": False},
        )
        return (response.choices[0].message.content or "").strip()


async def patch_one(provider, client, model, record,
                    brain_areas_str, semaphore):
    title    = str(record.get("title", "") or "")
    abstract = " ".join(str(record.get("abstract", "") or "").split())

    prompt = USER_TEMPLATE.format(
        brain_areas = brain_areas_str,
        title       = title,
        abstract    = abstract[:800],
    )

    async with semaphore:
        try:
            raw = await call_llm(provider, client, model, SYSTEM_PROMPT, prompt)
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            # Model sometimes returns a bare list instead of {"brain_areas": [...]}
            if isinstance(result, list):
                brain_areas = result
            else:
                brain_areas = result.get("brain_areas", ["other"])
            if not brain_areas:
                brain_areas = ["other"]
            error = False
        except Exception as e:
            log.warning(f"PMID {record.get('pmid', '?')}: failed ({e!r})")
            brain_areas = ["error"]
            error = True

    return {
        "pmid":        record.get("pmid", ""),
        "brain_areas": "; ".join(brain_areas),
        "ann_error":   error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_patch(cfg: dict, dry_run: bool) -> None:
    repo_root   = Path(__file__).resolve().parent.parent
    db_path     = repo_root / cfg["paths"]["annotated_db"]
    cand_path   = repo_root / cfg["paths"]["candidates"]
    model       = cfg["filter"]["ai_model"]
    concurrency = cfg["filter"]["concurrency"]
    chunk_size  = cfg["filter"].get("checkpoint_every", 100)

    # Load vocabulary (brain areas only)
    candidates      = json.loads(cand_path.read_text())
    brain_areas_str = "\n".join(f"  - {a}" for a in candidates["brain_areas"])

    # Load annotated DB
    if not db_path.exists():
        raise SystemExit(f"ERROR: {db_path} not found — run annotate_papers.py first.")
    df = pd.read_excel(db_path, dtype=str)
    log.info(f"Loaded {len(df)} rows from {db_path.name}")

    # Find rows where brain_areas is exactly "other" or contains only "other"
    def is_other(val):
        if not isinstance(val, str):
            return False
        parts = [v.strip().lower() for v in val.split(";")]
        return all(p in ("other", "") for p in parts) and len(parts) > 0

    other_mask = df["brain_areas"].apply(is_other)
    other_rows = df[other_mask]
    log.info(f"Rows with brain_areas='other': {len(other_rows)}")

    if len(other_rows) == 0:
        log.info("Nothing to patch. Exiting.")
        return

    if dry_run:
        log.info("--- DRY RUN: no LLM calls made ---")
        log.info(f"Would re-process {len(other_rows)} rows.")
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
            raise SystemExit("ERROR: OPENAI_COMPAT_API_KEY / BASE_URL not set")
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        raise SystemExit(f"ERROR: Unknown LLM_PROVIDER '{provider}'")

    log.info(f"Provider: {provider} | Model: {model}")
    semaphore = asyncio.Semaphore(concurrency)
    records   = other_rows.to_dict("records")

    n_improved = 0
    n_still_other = 0
    n_error = 0

    for chunk_start in range(0, len(records), chunk_size):
        chunk        = records[chunk_start: chunk_start + chunk_size]
        chunk_n      = chunk_start // chunk_size + 1
        total_chunks = (len(records) + chunk_size - 1) // chunk_size
        log.info(f"Chunk {chunk_n}/{total_chunks} ({len(chunk)} papers)")

        tasks   = [patch_one(provider, client, model, rec,
                             brain_areas_str, semaphore) for rec in chunk]
        results = await atqdm.gather(*tasks, desc=f"Chunk {chunk_n}", unit="paper")

        for res in results:
            pmid     = res["pmid"]
            new_area = res["brain_areas"]
            err      = res["ann_error"]
            idx      = df.index[df["pmid"] == pmid]
            if len(idx) == 0:
                continue
            i = idx[0]
            df.at[i, "brain_areas"] = new_area
            if not err:
                df.at[i, "ann_error"] = "False"

            if err:
                n_error += 1
            elif new_area.strip().lower() == "other":
                n_still_other += 1
            else:
                n_improved += 1

        df.to_excel(db_path, index=False)
        log.info(f"Checkpoint saved ({chunk_start + len(chunk)}/{len(records)} done)")

    log.info(f"Patch complete.")
    log.info(f"  Improved (no longer 'other') : {n_improved}")
    log.info(f"  Still 'other'                : {n_still_other}")
    log.info(f"  Errors                       : {n_error}")
    log.info(f"Saved → {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-annotate brain_areas for papers currently labeled 'other'."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report count without making LLM calls")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    asyncio.run(run_patch(cfg, args.dry_run))


if __name__ == "__main__":
    main()

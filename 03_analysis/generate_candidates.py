"""
02_filter/generate_candidates.py
=================================
Scan abstracts from 02_filtered_db.xlsx and extract candidate
brain areas and cognitive functions using the configured LLM.

Processes a random sample of papers in batches, aggregates all
candidate terms (with counts), and saves them to:
    data/candidates_raw.json

After running, review the candidates and add curated lists to config.yaml
under the 'annotation:' section.

Usage
-----
    python 02_filter/generate_candidates.py           # sample 600 papers
    python 02_filter/generate_candidates.py -n 200   # smaller/faster sample
    python 02_filter/generate_candidates.py --all    # use all papers (slow)
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import openai
import pandas as pd
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM = """\
You are helping build a macaque electrophysiology literature database.
Extract structured metadata from neuroscience paper abstracts.
Respond with JSON only — no extra text, no markdown fences.
"""

USER_TEMPLATE = """\
Below are {n} paper titles and abstracts from macaque electrophysiology studies.

For each paper extract:
1. brain_areas: list of brain areas/regions studied.
   Use standard abbreviations where possible (e.g. "V1", "V4", "MT", "MST",
   "IT", "PFC", "dlPFC", "vmPFC", "OFC", "ACC", "S1", "S2", "M1", "PMC",
   "SMA", "LIP", "VIP", "AIP", "7a", "hippocampus", "amygdala", "striatum",
   "caudate", "putamen", "thalamus", "SC", "cerebellum", "brainstem").
   Use [] if none mentioned.

2. functions: list of cognitive/behavioral functions or task types studied.
   Examples: "working memory", "attention", "reward", "decision making",
   "motor planning", "motor control", "saccade", "object recognition",
   "learning", "social cognition", "emotion", "pain", "multisensory".
   Use [] if none mentioned.

Return a JSON array with exactly {n} objects, one per paper, in order:
[
  {{"brain_areas": [...], "functions": [...]}},
  ...
]

Papers:
{papers}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate brain area and function lists from abstracts."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("-n", type=int, default=600,
                        help="Number of papers to sample (default: 600)")
    parser.add_argument("--all", action="store_true",
                        help="Use all papers (ignores -n)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=5,
                        help="Papers per LLM call (default: 5)")
    args = parser.parse_args()

    cfg      = yaml.safe_load(open(args.config))
    db_path  = repo_root / cfg["paths"]["filtered_db2"]
    out_path = repo_root / "data" / "candidates_raw.json"

    load_dotenv(repo_root / "key.env")
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip()

    if provider == "openai_compat":
        client = openai.OpenAI(
            api_key=os.environ["OPENAI_COMPAT_API_KEY"],
            base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
        )
    else:
        import anthropic as ant
        client = ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    model = cfg["filter"]["ai_model"]

    # Load DB
    df = pd.read_excel(db_path, dtype=str)
    print(f"Loaded {len(df)} papers from {db_path.name}")

    if args.all:
        sample = df.reset_index(drop=True)
    else:
        n = min(args.n, len(df))
        sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampling {len(sample)} papers (seed={args.seed})")

    all_areas = Counter()
    all_funcs = Counter()
    BATCH     = args.batch
    n_batches = (len(sample) + BATCH - 1) // BATCH

    for b in range(n_batches):
        chunk = sample.iloc[b * BATCH: (b + 1) * BATCH]

        papers_text = ""
        for i, (_, row) in enumerate(chunk.iterrows(), 1):
            title    = str(row.get("title", "") or "")
            abstract = " ".join(str(row.get("abstract", "") or "").split())
            papers_text += f"\n[{i}] Title: {title}\nAbstract: {abstract[:600]}\n"

        prompt = USER_TEMPLATE.format(n=len(chunk), papers=papers_text)

        try:
            if provider == "openai_compat":
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    extra_body={"enable_thinking": False},
                )
                raw = (resp.choices[0].message.content or "").strip()
            else:
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip() if resp.content else ""

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rsplit("```", 1)[0].strip()

            results = json.loads(raw)
            for item in results:
                for a in item.get("brain_areas", []):
                    if a.strip():
                        all_areas[a.strip()] += 1
                for f in item.get("functions", []):
                    if f.strip():
                        all_funcs[f.strip()] += 1

        except Exception as e:
            print(f"  Batch {b + 1} error: {e}")

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"  {b + 1}/{n_batches} batches | "
                  f"areas={len(all_areas)} funcs={len(all_funcs)}")

        time.sleep(0.05)

    print(f"\nUnique brain area candidates : {len(all_areas)}")
    print(f"Unique function candidates   : {len(all_funcs)}")

    out = {
        "brain_areas": dict(all_areas.most_common()),
        "functions":   dict(all_funcs.most_common()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")
    print("Review candidates_raw.json, then add curated lists to config.yaml")


if __name__ == "__main__":
    main()

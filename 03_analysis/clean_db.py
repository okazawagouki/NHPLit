"""
03_analysis/clean_db.py
========================
Stage 5: Clean and normalise journal names from 03_annotated_db.xlsx
and write the result to 04_cleaned_db.xlsx.

Two-pass cleaning:
  1. Automated rules — strips subtitles, parentheticals, leading "The ", title-cases.
  2. Manual mapping — overrides for journals whose names are still inconsistent
     after automated cleaning. Edit JOURNAL_MAP below to add/fix entries.

Re-run this script any time 03_annotated_db.xlsx is updated.

Usage
-----
    python 03_analysis/clean_db.py
    python 03_analysis/clean_db.py --report   # print top 50 journal names after cleaning
    python 03_analysis/clean_db.py --dry-run  # print without writing output file
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Manual journal name mapping
# Add entries here whenever you spot inconsistencies in the report.
# Keys are matched AFTER automated cleaning (case-insensitive).
# Values are the canonical full name you want in the database.
# ---------------------------------------------------------------------------
JOURNAL_MAP = {
    # Neuroscience flagships
    "journal of neuroscience": "Journal of Neuroscience",
    "journal of neuroscience research": "Journal of Neuroscience Research",
    "nature neuroscience": "Nature Neuroscience",
    "nature communications": "Nature Communications",
    "nature methods": "Nature Methods",
    "neuron": "Neuron",
    "current biology": "Current Biology",
    "plos one": "PLOS ONE",
    "plos biology": "PLOS Biology",
    "elife": "eLife",
    "science": "Science",
    "proceedings of the national academy of sciences of the united states of america":
        "PNAS",
    "pnas": "PNAS",

    # Physiology & systems neuro
    "journal of neurophysiology": "Journal of Neurophysiology",
    "cerebral cortex": "Cerebral Cortex",
    "cerebral cortex (new york, n.y.)": "Cerebral Cortex",
    "european journal of neuroscience": "European Journal of Neuroscience",
    "experimental brain research": "Experimental Brain Research",
    "brain research": "Brain Research",
    "neuroscience": "Neuroscience",
    "neuroscience letters": "Neuroscience Letters",
    "neuroscience & biobehavioral reviews": "Neuroscience & Biobehavioral Reviews",
    "behavioural brain research": "Behavioural Brain Research",
    "brain": "Brain",
    "brain research bulletin": "Brain Research Bulletin",

    # Cognitive / systems
    "journal of cognitive neuroscience": "Journal of Cognitive Neuroscience",
    "cognitive neuroscience": "Cognitive Neuroscience",
    "frontiers in neuroscience": "Frontiers in Neuroscience",
    "frontiers in systems neuroscience": "Frontiers in Systems Neuroscience",
    "frontiers in neural circuits": "Frontiers in Neural Circuits",
    "frontiers in computational neuroscience": "Frontiers in Computational Neuroscience",

    # Neuro imaging / methods (sometimes appear)
    "neuroimage": "NeuroImage",
    "journal of neuroscience methods": "Journal of Neuroscience Methods",
    "acs chemical neuroscience": "ACS Chemical Neuroscience",

    # Primate-specific
    "journal of comparative neurology": "Journal of Comparative Neurology",
    "journal of primatology": "Journal of Primatology",
}

# ---------------------------------------------------------------------------
# Automated cleaning
# ---------------------------------------------------------------------------
# Patterns to strip from the end of journal names (e.g. year/city qualifiers)
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")


def clean_journal(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return name

    # 1. Strip trailing parentheticals: "Cerebral Cortex (New York, N.Y. : 1991)"
    name = _PAREN_RE.sub("", name).strip()

    # 2. Truncate at " : " — removes subtitles
    if " : " in name:
        name = name.split(" : ")[0].strip()

    # 3. Strip leading "The " / "the "
    if name.lower().startswith("the "):
        name = name[4:]

    # 4. Title-case (preserves ALL-CAPS acronyms like "PLOS")
    name = name.strip().title()

    # 5. Restore common lower-case words that title() over-capitalises
    for word in ("and", "of", "the", "in", "for", "a", "an", "&"):
        name = re.sub(rf"\b{word.title()}\b", word, name)
    # But keep first word capitalised
    if name:
        name = name[0].upper() + name[1:]

    # 6. Manual map (case-insensitive lookup)
    mapped = JOURNAL_MAP.get(name.lower())
    if mapped:
        name = mapped

    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Clean journal names and write 04_cleaned_db.xlsx."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("--report", action="store_true",
                        help="Print top 50 journal names after cleaning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report without writing output file")
    args = parser.parse_args()

    cfg      = yaml.safe_load(open(args.config))
    in_path  = repo_root / cfg["paths"]["annotated_db"]
    out_path = repo_root / cfg["paths"]["cleaned_db"]

    print(f"Loading {in_path.name}...")
    df = pd.read_excel(in_path, dtype=str)
    print(f"  {len(df)} papers")

    # Apply cleaning
    df["journal"] = df["journal"].apply(clean_journal)

    before = df["journal"].nunique()
    print(f"  Unique journal names after cleaning: {before}")

    if args.report or args.dry_run:
        counts = Counter(df["journal"].dropna().tolist())
        print(f"\nTop 50 journal names:")
        for j, n in counts.most_common(50):
            print(f"  {n:5d}  {j}")

    if args.dry_run:
        print("\n--dry-run: no file written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

# NHPLit — Macaque Electrophysiology Literature Database

A reproducible, AI-assisted database of macaque electrophysiology literature built from PubMed metadata.

## Pipeline overview

```
PubMed → raw cache → 01_extracted_db → 02_filtered_db → 03_annotated_db → report.html
  (1)        (2)           (3)                (4)               (5)             (6)
```

## Quick start

```bash
pip install -r requirements.txt

# Credentials (never committed to git)
cp example.env key.env   # fill in NCBI email, API keys
```

---

## Stage 1 — Fetch (`01_search/`)

Query PubMed with species × signal keyword combinations defined in `config.yaml` and cache each paper's metadata (title, abstract, authors, journal, year, DOI, MeSH terms) as individual JSON files under `data/raw_cache/`. Fetching is done one year at a time so partial re-runs are cheap and result caps are avoided. Already-cached PMIDs are skipped, so it is safe to re-run.

| Script | Description |
|---|---|
| `fetch_pubmed.py` | Fetch metadata from PubMed and cache as JSON |
| `summarize_cache.py` | Show per-year fetch progress and AI filter results |

```bash
python 01_search/fetch_pubmed.py                        # current year
python 01_search/fetch_pubmed.py --all                  # full range in config.yaml
python 01_search/fetch_pubmed.py --years 2023 2024      # specific years
python 01_search/fetch_pubmed.py --years 2010 2020      # year range
python 01_search/fetch_pubmed.py --dry-run              # preview only

python 01_search/summarize_cache.py                     # check progress
```

---

## Stage 2 — AI Filter (`02_filter/`)

Each cached paper is sent to an LLM (configured in `config.yaml`) with a prompt asking whether it reports primary in-vivo electrophysiology recordings from macaques. The model returns a relevance judgement and a confidence score. Papers above the confidence threshold are labelled **pass** or **fail**; uncertain ones are labelled **pending** for human review. Results are saved incrementally to `01_extracted_db.xlsx` so the script can be interrupted and resumed.

After AI filtering, `build_filtered_db.py` produces a clean `02_filtered_db.xlsx` by keeping only passed papers and deduplicating bioRxiv/arXiv preprints that were later published in a journal.

| Script | Description |
|---|---|
| `filter_papers.py` | AI screening: labels each paper pass / fail / pending |
| `human_review.py` | Interactive terminal review of `pending` items one by one |
| `quality_check.py` | Random spot-check of AI pass/fail decisions; correct if wrong |
| `build_filtered_db.py` | Build `02_filtered_db.xlsx`: pass-only, preprints deduped |

```bash
python 02_filter/filter_papers.py                       # filter all new cache entries
python 02_filter/filter_papers.py --years 2023 2024
python 02_filter/filter_papers.py --dry-run

python 02_filter/human_review.py                        # review pending items
python 02_filter/human_review.py --autosave

python 02_filter/quality_check.py                       # spot-check 20 random pass/fail
python 02_filter/quality_check.py -n 50
python 02_filter/quality_check.py --flag pass           # only check AI-passed items
python 02_filter/quality_check.py --flag fail

python 02_filter/build_filtered_db.py                   # build 02_filtered_db.xlsx
python 02_filter/build_filtered_db.py --dry-run
```

---

## Stage 3 — Annotate (`03_analysis/`)

Each passed paper is annotated with the brain regions recorded and cognitive functions studied, chosen from a controlled vocabulary stored in `data/candidates_curated.json`. The vocabulary is generated semi-automatically: `generate_candidates.py` samples abstracts and proposes candidate terms via LLM; you then review and trim the JSON file before running annotation. This controlled vocabulary ensures consistent, queryable labels across all papers.

`annotate_papers.py` sends each paper's title and abstract to the LLM with the full vocabulary list and asks it to select all applicable terms. Results are added as new columns to `03_annotated_db.xlsx`. Papers where no brain region could be identified are labelled `other`; `patch_other_areas.py` re-processes these with a more permissive prompt that extracts any region mentioned in the text even for methods or technique papers.

| Script | Description |
|---|---|
| `generate_candidates.py` | Sample abstracts and propose brain area / function vocabulary via LLM |
| `annotate_papers.py` | Annotate each paper with brain areas + functions from the vocabulary |
| `patch_other_areas.py` | Re-annotate `other` papers with a permissive region-extraction prompt |
| `quality_check.py` | Random spot-check of annotations; edit interactively if wrong |
| `summarize_annotations.py` | Show annotation progress, top regions, and top functions |
| `clean_db.py` | Normalise journal names → `04_cleaned_db.xlsx` |
| `build_report.py` | Generate `report.html` — self-contained interactive public dashboard |

```bash
# Build vocabulary (run once, then manually curate data/candidates_curated.json)
python 03_analysis/generate_candidates.py               # sample 600 papers
python 03_analysis/generate_candidates.py -n 200

# Annotate
python 03_analysis/annotate_papers.py                   # annotate all
python 03_analysis/annotate_papers.py --years 2023 2024
python 03_analysis/annotate_papers.py --year-range 2010 2020
python 03_analysis/annotate_papers.py --dry-run

# Patch papers with no region identified
python 03_analysis/patch_other_areas.py
python 03_analysis/patch_other_areas.py --dry-run


# Quality check
python 03_analysis/quality_check.py                     # 20 random papers
python 03_analysis/quality_check.py -n 50
python 03_analysis/quality_check.py --errors            # only ann_error rows
python 03_analysis/quality_check.py --vocab             # show vocabulary first

# Progress summary
python 03_analysis/summarize_annotations.py

# Clean journal names
python 03_analysis/clean_db.py                          # → 04_cleaned_db.xlsx
python 03_analysis/clean_db.py --report                 # preview top 50 names
python 03_analysis/clean_db.py --dry-run                # preview without writing

# Build report (auto-uses 04_cleaned_db.xlsx if present)
python 03_analysis/build_report.py                      # → report.html
python 03_analysis/build_report.py --top 30 --since 1990
```

---

## Repository layout

```
NHPLit/
├── config.yaml                  ← all settings: keywords, years, paths, models
├── key.env                      ← API keys (gitignored)
├── requirements.txt
├── report.html                  ← public dashboard (regenerated by build_report.py)
│
├── 01_search/
│   ├── fetch_pubmed.py
│   └── summarize_cache.py
│
├── 02_filter/
│   ├── filter_papers.py
│   ├── human_review.py
│   ├── quality_check.py
│   └── build_filtered_db.py
│
├── 03_analysis/
│   ├── generate_candidates.py
│   ├── annotate_papers.py
│   ├── patch_other_areas.py
│   ├── quality_check.py
│   ├── summarize_annotations.py
│   └── build_report.py
│
└── data/
    ├── raw_cache/               ← one JSON per PMID, by year/  [gitignored]
    ├── pmid_log.csv             ← fetch log
    ├── candidates_curated.json  ← curated vocabulary for annotation
    ├── 01_extracted_db.xlsx     ← AI-screened papers (pass/fail/pending)
    ├── 02_filtered_db.xlsx      ← pass-only, preprints deduped
    └── 03_annotated_db.xlsx     ← brain areas + functions annotated
```

---

## Configuration (`config.yaml`)

| Section | Purpose |
|---|---|
| `entrez` | PubMed API settings (batch size, rate limits) |
| `years` | Year range to fetch |
| `paths` | File paths for all inputs and outputs |
| `keywords` | Species and signal terms; additional queries; exclusions |
| `filter` | AI model, confidence threshold, concurrency |

---

## Data columns

**01_extracted_db / 02_filtered_db:**

| Column | Description |
|---|---|
| `pmid` | PubMed ID |
| `title` | Article title |
| `abstract` | Full abstract |
| `authors` | Semicolon-separated author list |
| `journal` | Journal name |
| `year` | Publication year |
| `doi` | DOI |
| `mesh` | MeSH descriptors |
| `ai_relevant` | AI relevance judgement (true/false) |
| `ai_confidence` | AI confidence score (0–1) |
| `flag` | `pass` / `fail` / `pending` / `error` |

**03_annotated_db** adds:

| Column | Description |
|---|---|
| `brain_areas` | Semicolon-separated brain regions (from vocabulary) |
| `functions` | Semicolon-separated cognitive functions (from vocabulary) |
| `ann_error` | True if annotation LLM call failed |

---

## Keyword strategy

Queries are built as `(species term) AND (signal term)` for every combination defined in `config.yaml`, fetched one year at a time to avoid PubMed result caps. MeSH-based additional queries supplement the title/abstract searches. See `config.yaml` for the full lists and instructions.

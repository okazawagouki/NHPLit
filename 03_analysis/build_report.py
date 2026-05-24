"""
03_analysis/build_report.py
============================
Generate a self-contained interactive HTML report from 04_cleaned_db.xlsx.

Charts included:
  1. Total papers per year
  2. Top brain regions recorded
  3. Top functions studied
  4. Top journals (by paper count)
  5. Journal trends over time
  6. Brain region × Function co-occurrence heatmap
  7. Brain region × Year heatmap (research focus over time)

Output: report.html (path set in config.yaml)
Host on GitHub Pages or open locally in any browser.

Usage
-----
    python 03_analysis/build_report.py
    python 03_analysis/build_report.py --top 25   # show top N in bar charts / heatmaps
    python 03_analysis/build_report.py --since 1990  # restrict year range
"""

import argparse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml

# ---------------------------------------------------------------------------
# Colour palette — clean scientific blues/teals
# ---------------------------------------------------------------------------
PRIMARY   = "#2C5F8A"
SECONDARY = "#3A8C7E"
ACCENT    = "#E07B39"
BG        = "#FFFFFF"
GRID      = "#E8EDF2"
TEXT      = "#1A2733"
SUBTEXT   = "#5A7080"

PALETTE = [
    "#2C5F8A", "#3A8C7E", "#E07B39", "#7B5EA7",
    "#C0453A", "#4E9E6B", "#C4962A", "#5589B3",
    "#6BB89E", "#E8A070",
]

# ---------------------------------------------------------------------------
# Selected journals for trend chart
# ---------------------------------------------------------------------------
SELECTED_JOURNALS = [
    "Nature Neuroscience",
    "Neuron",
    "PNAS",
    "Cell",
    "Science",
    "Nature",
    "Nature Communications",
    "Journal of Neuroscience",
    "Current Biology",
    "eLife",
]

# ---------------------------------------------------------------------------
# Anatomical hierarchy for brain region sorting
# Regions are ordered from posterior (sensory) to anterior (frontal/motor),
# with subcortical structures at the end.
# ---------------------------------------------------------------------------
REGION_HIERARCHY = [
    # --- Visual cortex (early → late) ---
    "visual cortex", "striate cortex", "extrastriate cortex",
    "LGN", "V1", "V2", "V3", "V4", "V4t", "V5", "MT", "MST", "FST", "STP",
    # --- Inferotemporal ---
    "inferotemporal cortex", "temporal cortex",
    "TEO", "TE", "IT", "AIT", "PIT",
    # --- Parietal ---
    "parietal cortex", "somatosensory cortex",
    "S1", "S2", "PE", "PF", "PG", "7a", "7b", "LIP", "VIP", "AIP", "MIP", "CIP",
    # --- Superior temporal ---
    "superior temporal sulcus", "STG", "STS", "STSd", "STSv",
    # --- Frontal / motor ---
    "motor cortex", "premotor cortex",
    "M1", "PMd", "PMv", "PMC", "SMA", "pre-SMA", "CMA",
    "frontal eye field", "FEF", "SEF",
    # --- Prefrontal ---
    "prefrontal cortex", "orbitofrontal cortex", "anterior cingulate cortex", "cingulate cortex",
    "dlPFC", "vlPFC", "PFC", "LPFC", "MPFC", "OFC", "vmPFC",
    "ACC", "dACC", "PCC",
    # --- Temporal / medial temporal ---
    "medial temporal lobe", "hippocampus", "entorhinal cortex",
    "EC", "perirhinal cortex", "parahippocampal cortex",
    "amygdala",
    # --- Subcortical ---
    "striatum", "caudate", "putamen", "GPe", "GPi", "globus pallidus",
    "subthalamic nucleus",
    "thalamus", "MD", "pulvinar",
    "superior colliculus", "SC",
    "cerebellum",
    "brainstem",
    "basal ganglia",
    "SNr", "SNc", "substantia nigra",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, Helvetica Neue, Arial, sans-serif",
              color=TEXT, size=13),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID,
                    font_size=12, font_family="Inter, Arial, sans-serif"),
)

# Default axis style — applied via update_xaxes/update_yaxes, not in layout dict
AXIS_STYLE = dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
DEFAULT_MARGIN = dict(l=60, r=30, t=50, b=60)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def split_field(series: pd.Series) -> list[str]:
    """Explode a semicolon-separated column into a flat list of values."""
    out = []
    for val in series.dropna():
        for item in str(val).split(";"):
            item = item.strip()
            if item and item.lower() not in ("other", "unknown", "error", ""):
                out.append(item)
    return out


def top_n_counts(items: list[str], n: int) -> tuple[list[str], list[int]]:
    counts = Counter(items).most_common(n)
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]
    return labels, values


def safe_int(v) -> int | None:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def chart_papers_per_year(df: pd.DataFrame, since: int) -> go.Figure:
    years = [safe_int(y) for y in df["year"]]
    counts = Counter(y for y in years if y and y >= since)
    yr_sorted = sorted(counts.keys())
    fig = go.Figure(go.Bar(
        x=yr_sorted,
        y=[counts[y] for y in yr_sorted],
        marker_color=PRIMARY,
        hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                      title="Papers per Year",
                      xaxis_title="Year",
                      yaxis_title="Number of papers")
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def sort_regions_by_hierarchy(labels: list[str], values: list[int]) -> tuple[list[str], list[int]]:
    """Sort region labels by REGION_HIERARCHY; unknown regions go to the end (alphabetical)."""
    hierarchy_index = {r.lower(): i for i, r in enumerate(REGION_HIERARCHY)}
    n = len(REGION_HIERARCHY)
    paired = sorted(
        zip(labels, values),
        key=lambda lv: (hierarchy_index.get(lv[0].lower(), n), lv[0].lower())
    )
    if paired:
        labels, values = zip(*paired)
        return list(labels), list(values)
    return [], []


def chart_top_regions(df: pd.DataFrame, top: int) -> go.Figure:
    labels, values = top_n_counts(split_field(df["brain_areas"]), top)
    # Sort by anatomical hierarchy instead of by count
    labels, values = sort_regions_by_hierarchy(labels, values)
    # Reverse so the first hierarchy entry appears at the top in horizontal bar
    labels, values = labels[::-1], values[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=SECONDARY,
        hovertemplate="<b>%{y}</b><br>Papers: %{x}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                      title=f"Top {top} Brain Regions Recorded (anatomical order)",
                      xaxis_title="Number of papers",
                      yaxis_title=None,
                      height=max(400, top * 22))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def chart_top_functions(df: pd.DataFrame, top: int) -> go.Figure:
    labels, values = top_n_counts(split_field(df["functions"]), top)
    labels, values = labels[::-1], values[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=ACCENT,
        hovertemplate="<b>%{y}</b><br>Papers: %{x}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                      title=f"Top {top} Functions Studied",
                      xaxis_title="Number of papers",
                      yaxis_title=None,
                      height=max(400, top * 22))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def chart_top_journals(df: pd.DataFrame, top: int) -> go.Figure:
    labels, values = top_n_counts(df["journal"].dropna().tolist(), top)
    labels, values = labels[::-1], values[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color="#7B5EA7",
        hovertemplate="<b>%{y}</b><br>Papers: %{x}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                      title=f"Top {top} Journals",
                      xaxis_title="Number of papers",
                      yaxis_title=None,
                      height=max(400, top * 22))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def chart_journal_trends(df: pd.DataFrame, since: int) -> go.Figure:
    """Plot publication trends for selected journals that appear in the data."""
    present = set(df["journal"].dropna().unique())
    # Keep only selected journals that actually appear in the data, preserving curated order
    journals_to_plot = [j for j in SELECTED_JOURNALS if j in present]

    df2 = df.copy()
    df2["year_int"] = df2["year"].apply(safe_int)
    df2 = df2[df2["year_int"] >= since]

    fig = go.Figure()
    for i, journal in enumerate(journals_to_plot):
        sub    = df2[df2["journal"] == journal]
        counts = Counter(sub["year_int"])
        yrs    = sorted(counts.keys())
        if not yrs:
            continue
        fig.add_trace(go.Scatter(
            x=yrs,
            y=[counts[y] for y in yrs],
            name=journal,
            mode="lines+markers",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{journal}</b><br>%{{x}}: %{{y}} papers<extra></extra>",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                      title="Publication Trends — Selected Journals",
                      xaxis_title="Year",
                      yaxis_title="Papers per year",
                      legend=dict(orientation="v", x=1.01, y=1,
                                  bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def chart_region_function_heatmap(df: pd.DataFrame, top: int) -> go.Figure:
    areas_by_count, vals_by_count = top_n_counts(split_field(df["brain_areas"]), top)
    top_areas, _ = sort_regions_by_hierarchy(areas_by_count, vals_by_count)
    top_funcs, _  = top_n_counts(split_field(df["functions"]),   top)

    area_set = set(top_areas)
    func_set = set(top_funcs)

    # Count co-occurrences per paper
    matrix = defaultdict(int)
    for _, row in df.iterrows():
        areas = [a.strip() for a in str(row.get("brain_areas", "") or "").split(";")
                 if a.strip() in area_set]
        funcs = [f.strip() for f in str(row.get("functions", "") or "").split(";")
                 if f.strip() in func_set]
        for a in areas:
            for f in funcs:
                matrix[(a, f)] += 1

    # Build z matrix (rows = regions, cols = functions)
    z    = [[matrix[(a, f)] for f in top_funcs] for a in top_areas]
    zmax = max((matrix[k] for k in matrix), default=1)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=top_funcs,
        y=top_areas,
        colorscale="Blues",
        zmin=0, zmax=zmax,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Papers: %{z}<extra></extra>",
        colorbar=dict(title="Papers"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"Brain Region × Function Co-occurrence (top {top} each)",
                      height=max(500, top * 22),
                      margin=dict(l=120, r=30, t=60, b=160))
    fig.update_xaxes(**AXIS_STYLE, tickangle=-40, tickfont_size=11)
    fig.update_yaxes(**AXIS_STYLE, tickfont_size=11)
    return fig


def chart_region_year_heatmap(df: pd.DataFrame, since: int, top: int) -> go.Figure:
    df2 = df.copy()
    df2["year_int"] = df2["year"].apply(safe_int)
    df2 = df2[df2["year_int"] >= since]

    areas_by_count, vals_by_count = top_n_counts(split_field(df2["brain_areas"]), top)
    top_areas, _ = sort_regions_by_hierarchy(areas_by_count, vals_by_count)
    area_set      = set(top_areas)
    all_years     = sorted(df2["year_int"].dropna().unique().astype(int))

    # Count papers per year (denominator for normalisation)
    papers_per_year = Counter(df2["year_int"].dropna().astype(int))

    matrix = defaultdict(int)
    for _, row in df2.iterrows():
        yr    = safe_int(row.get("year_int"))
        areas = [a.strip() for a in str(row.get("brain_areas", "") or "").split(";")
                 if a.strip() in area_set]
        for a in areas:
            matrix[(a, yr)] += 1

    # Normalise: percentage of papers that year containing each region
    def pct(a, y):
        n = papers_per_year.get(y, 0)
        return round(100.0 * matrix[(a, y)] / n, 1) if n else 0.0

    z         = [[pct(a, y) for y in all_years] for a in top_areas]
    hover_raw = [[matrix[(a, y)] for y in all_years] for a in top_areas]

    # Build custom hover text matrix
    hover_text = [
        [f"<b>{a}</b><br>{y}: {pct(a,y):.1f}% ({matrix[(a,y)]} papers)"
         for y in all_years]
        for a in top_areas
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=all_years,
        y=top_areas,
        colorscale="Teal",
        zmin=0,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title="% of papers"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"Brain Region Recording Trends Over Time — % of papers per year (top {top} regions)",
                      height=max(500, top * 22),
                      margin=dict(l=120, r=30, t=60, b=60))
    fig.update_xaxes(**AXIS_STYLE, title_text="Year")
    fig.update_yaxes(**AXIS_STYLE, tickfont_size=11)
    return fig


def chart_function_year_heatmap(df: pd.DataFrame, since: int, top: int) -> go.Figure:
    df2 = df.copy()
    df2["year_int"] = df2["year"].apply(safe_int)
    df2 = df2[df2["year_int"] >= since]

    top_funcs, _ = top_n_counts(split_field(df2["functions"]), top)
    func_set      = set(top_funcs)
    all_years     = sorted(df2["year_int"].dropna().unique().astype(int))

    papers_per_year = Counter(df2["year_int"].dropna().astype(int))

    matrix = defaultdict(int)
    for _, row in df2.iterrows():
        yr    = safe_int(row.get("year_int"))
        funcs = [f.strip() for f in str(row.get("functions", "") or "").split(";")
                 if f.strip() in func_set]
        for f in funcs:
            matrix[(f, yr)] += 1

    def pct(f, y):
        n = papers_per_year.get(y, 0)
        return round(100.0 * matrix[(f, y)] / n, 1) if n else 0.0

    hover_text = [
        [f"<b>{f}</b><br>{y}: {pct(f,y):.1f}% ({matrix[(f,y)]} papers)"
         for y in all_years]
        for f in top_funcs
    ]

    z = [[pct(f, y) for y in all_years] for f in top_funcs]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=all_years,
        y=top_funcs,
        colorscale="Oranges",
        zmin=0,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title="% of papers"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"Cognitive Function Trends Over Time — % of papers per year (top {top} functions)",
                      height=max(500, top * 22),
                      margin=dict(l=180, r=30, t=60, b=60))
    fig.update_xaxes(**AXIS_STYLE, title_text="Year")
    fig.update_yaxes(**AXIS_STYLE, tickfont_size=11)
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHP Electrophysiology Literature Database</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
      background: #F4F7FA;
      color: {text};
    }}
    header {{
      background: {primary};
      color: white;
      padding: 40px 48px 32px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.5px;
    }}
    header p {{
      margin: 0;
      font-size: 1rem;
      opacity: 0.85;
      max-width: 680px;
      line-height: 1.6;
    }}
    .meta {{
      margin-top: 18px;
      display: flex;
      gap: 28px;
      flex-wrap: wrap;
    }}
    .meta-item {{
      background: rgba(255,255,255,0.15);
      border-radius: 8px;
      padding: 10px 18px;
      text-align: center;
    }}
    .meta-item .num {{
      font-size: 1.6rem;
      font-weight: 700;
      display: block;
    }}
    .meta-item .lbl {{
      font-size: 0.78rem;
      opacity: 0.85;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px 80px;
    }}
    section {{
      margin-bottom: 48px;
    }}
    section h2 {{
      font-size: 1.2rem;
      font-weight: 600;
      color: {primary};
      margin: 0 0 4px;
      padding-bottom: 8px;
      border-bottom: 2px solid {primary};
    }}
    section p.desc {{
      margin: 8px 0 16px;
      color: {subtext};
      font-size: 0.92rem;
      line-height: 1.6;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      padding: 8px;
      margin-bottom: 24px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}
    @media (max-width: 760px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      header {{ padding: 28px 20px; }}
      main {{ padding: 24px 12px 60px; }}
    }}
    footer {{
      text-align: center;
      padding: 24px;
      font-size: 0.82rem;
      color: {subtext};
      border-top: 1px solid {grid};
    }}
  </style>
</head>
<body>

<header>
  <h1>🐒 NHP Electrophysiology Literature Database (beta)</h1>
  <p>
    A semi-automatically generated database of monkey electrophysiology studies, built by
    automated PubMed search and AI-based screening and annotation. Don't trust it much.
  </p>
  <div class="meta">
    <div class="meta-item">
      <span class="num">{n_papers}</span>
      <span class="lbl">Papers</span>
    </div>
    <div class="meta-item">
      <span class="num">{year_range}</span>
      <span class="lbl">Year range</span>
    </div>
    <div class="meta-item">
      <span class="num">{updated}</span>
      <span class="lbl">Last updated</span>
    </div>
    <div class="meta-item">
      <a href="data/04_cleaned_db.xlsx"
         style="color:white; text-decoration:none;"
         download>
        <span class="num">⬇</span>
        <span class="lbl">Download database</span>
      </a>
    </div>
  </div>
</header>

<main>

  <section>
    <h2>Publication Volume</h2>
    <p class="desc">Total number of monkey electrophysiology papers per year included in this database.</p>
    <div class="card">{fig_years}</div>
  </section>

  <section>
    <h2>Recording Sites &amp; Functions</h2>
    <p class="desc">Brain regions most frequently recorded from (left) and functions most frequently studied (right), across all papers.</p>
    <div class="two-col">
      <div class="card">{fig_regions}</div>
      <div class="card">{fig_functions}</div>
    </div>
  </section>

  <section>
    <h2>Journals</h2>
    <p class="desc">Most productive journals (left) and publication trends for selected journals over time (right).</p>
    <div class="two-col">
      <div class="card">{fig_journals}</div>
      <div class="card">{fig_journal_trends}</div>
    </div>
  </section>

  <section>
    <h2>Brain Region × Function Co-occurrence</h2>
    <p class="desc">
      How often each brain region and function appear together in the same paper.
      Darker cells indicate more papers studying that region–function combination.
    </p>
    <div class="card">{fig_heatmap_rf}</div>
  </section>

  <section>
    <h2>Recording Site Trends Over Time</h2>
    <p class="desc">
      Percentage of papers per year that recorded from each brain region.
      Normalising by yearly paper count removes the effect of overall publication growth.
    </p>
    <div class="card">{fig_heatmap_ry}</div>
  </section>

  <section>
    <h2>Cognitive Function Trends Over Time</h2>
    <p class="desc">
      Percentage of papers per year studying each cognitive function.
      Shows how research interests have shifted across decades.
    </p>
    <div class="card">{fig_heatmap_fy}</div>
  </section>

  <section>
    <h2>Disclaimer</h2>
    <p class="desc">
      This database is generated with AI-assist, based on pubmed search of terms like "monkeys AND neurons" in title/abstract. Also check whether [MH] field contains macaca etc. 
      Papers without abstract are excluded. It is then further filtered with AI, and the target brain regions, functions extracted.
      But perhaps, many relevant papers are missing (particularly those not including species name in title/abstract/MH), and some irrelevant papers may be included.
    </p>
    <p class="desc">
      This database focuses on cellular-level electrophysiology
      (single-unit, multi-unit, and LFP recordings) in macaques and other non-human primates.
      It does not systematically cover EEG, fMRI, calcium imaging, or purely behavioral
      studies.
    </p>
    <p class="desc">
      Please feel free to copy, update, redistribute, etc.
    </p>
  </section>

</main>

<footer>
  Generated on {updated} from <code>04_cleaned_db.xlsx</code> &nbsp;·&nbsp;
  <a href="https://github.com/okazawagouki/NHPLit" style="color:{subtext};">Source on GitHub</a>
</footer>

</body>
</html>
"""


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displayModeBar": False})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build interactive HTML report from 04_cleaned_db.xlsx."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=repo_root / "config.yaml")
    parser.add_argument("--top", type=int, default=25,
                        help="Top N terms to show in bar charts and heatmaps (default: 25)")
    parser.add_argument("--since", type=int, default=1980,
                        help="Earliest year to include (default: 1980)")
    args = parser.parse_args()

    cfg      = yaml.safe_load(open(args.config))
    out_path = repo_root / cfg["paths"]["report"]

    # Prefer cleaned DB if available, fall back to annotated DB
    db_path = repo_root / cfg["paths"].get("cleaned_db", cfg["paths"]["annotated_db"])
    if not db_path.exists():
        db_path = repo_root / cfg["paths"]["annotated_db"]
    if not db_path.exists():
        raise SystemExit(f"ERROR: database not found — run annotate_papers.py first.")
    print(f"Using: {db_path.name}")

    print(f"Loading {db_path.name}...")
    df = pd.read_excel(db_path, dtype=str)
    print(f"  {len(df)} papers loaded")

    # Filter by year
    df["year_int"] = df["year"].apply(safe_int)
    df = df[df["year_int"].notna()]

    # Summary stats
    n_papers   = len(df)
    years      = sorted(df["year_int"].dropna().astype(int).unique())
    year_range = f"{years[0]}–{years[-1]}" if years else "N/A"
    updated    = date.today().strftime("%Y-%m-%d")

    print("Building charts...")
    fig_years          = chart_papers_per_year(df, args.since)
    fig_regions        = chart_top_regions(df, args.top)
    fig_functions      = chart_top_functions(df, args.top)
    fig_journals       = chart_top_journals(df, 20)
    fig_journal_trends = chart_journal_trends(df, args.since)  # uses SELECTED_JOURNALS list
    fig_heatmap_rf     = chart_region_function_heatmap(df, args.top)
    fig_heatmap_ry     = chart_region_year_heatmap(df, args.since, args.top)
    fig_heatmap_fy     = chart_function_year_heatmap(df, args.since, args.top)
    print("  All charts done")

    html = HTML_TEMPLATE.format(
        primary      = PRIMARY,
        text         = TEXT,
        subtext      = SUBTEXT,
        grid         = GRID,
        n_papers     = f"{n_papers:,}",
        year_range   = year_range,
        updated      = updated,
        fig_years          = fig_to_div(fig_years),
        fig_regions        = fig_to_div(fig_regions),
        fig_functions      = fig_to_div(fig_functions),
        fig_journals       = fig_to_div(fig_journals),
        fig_journal_trends = fig_to_div(fig_journal_trends),
        fig_heatmap_rf     = fig_to_div(fig_heatmap_rf),
        fig_heatmap_ry     = fig_to_div(fig_heatmap_ry),
        fig_heatmap_fy     = fig_to_div(fig_heatmap_fy),
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}")
    print(f"  Open in browser or host at GitHub Pages.")


if __name__ == "__main__":
    main()

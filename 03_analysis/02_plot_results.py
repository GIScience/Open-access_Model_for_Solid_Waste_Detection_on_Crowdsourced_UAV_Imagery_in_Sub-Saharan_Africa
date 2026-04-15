#!/usr/bin/env python3
"""
03_analysis/01_plot_results.py
=====================================
Produce the core paper figures from the final merged GPKG or CSV.

Input data
----------
The input file must contain the following columns:

  - waste_pct  or  rswci             -- Relative Solid Waste Contamination Index (%)
  - shdi       or  SHDI              -- Subnational Human Development Index
  - k_complexity_weighted            -- Street block complexity (weighted)
  - worldpop_population_un_density_hectare_weighted  -- Population density [people/ha]
  - green_pct                        -- Vegetation coverage (%) -- for bar overlay
  - country                          -- for bar chart colour (optional)

Figures generated
-----------------
1. waste_green_bar.png
   RSWCI bars ordered by decreasing value, with greenery fraction overlaid
   as a line on a secondary axis.  Bars are coloured by AEZ zone (DN column)
   if present, otherwise by country.

2. three_panel_analysis.png
   Three-panel scatter: left=SHDI (log x), middle=Block complexity (log x),
   right=Population density (log x).  Each panel has an OLS regression line,
   95% confidence interval band, and Spearman r annotation.

Usage
-----
    python 03_analysis/01_plot_results.py --input data/AOI.gpkg --outdir results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress, t

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Optional manual overrides: add entries here to pin a specific hex colour for
# a particular country name.  Any country not listed is auto-assigned from the
# qualitative palette below, so no changes are needed when new regions are added.
COUNTRY_COLOUR_OVERRIDES: dict[str, str] = {}


def _country_colour_map(countries: list[str]) -> dict[str, str]:
    """
    Return a {country: hex} mapping for all unique values in *countries*.

    Colours are assigned deterministically (sorted order) from a 20-colour
    qualitative palette so that the same country always gets the same colour
    across runs, regardless of which other countries are present.
    Manual overrides in COUNTRY_COLOUR_OVERRIDES take priority.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    unique = sorted(set(str(c) for c in countries if pd.notna(c)))
    cmap   = cm.get_cmap("tab20", max(len(unique), 1))
    auto   = {
        name: mcolors.to_hex(cmap(i))
        for i, name in enumerate(unique)
    }
    auto.update(COUNTRY_COLOUR_OVERRIDES)   # manual pins override auto
    return auto


# AEZ zone hex colours (DN -> hex) matching plots.ipynb
AEZ_HEX_MAP: dict[int, str] = {
    101: "#f5bfff", 102: "#e072ff", 103: "#a749ff", 104: "#7200ff",
    211: "#e3ffa8", 212: "#c5fe47", 213: "#88cc00", 214: "#2e8800",
    221: "#a8ffb6", 222: "#00ff70", 223: "#00cc76", 224: "#00593a",
    311: "#fedebe", 312: "#cbb885", 313: "#be9886", 314: "#b17c7e",
    321: "#fef4be", 322: "#fede4b", 323: "#fea700", 324: "#f45100",
    400: "#d6fdfd",
}
_AEZ_FALLBACK = "#b3b3b3"


def _bar_colours(df: pd.DataFrame) -> list[str]:
    """Return one colour per row, preferring AEZ DN if available."""
    if "DN" in df.columns:
        return [
            AEZ_HEX_MAP.get(int(dn), _AEZ_FALLBACK) if pd.notna(dn) else _AEZ_FALLBACK
            for dn in df["DN"]
        ]
    country_col = pick_col(df, ["country", "new_country"])
    if country_col:
        colour_map = _country_colour_map(list(df[country_col]))
        return [colour_map.get(str(c), _AEZ_FALLBACK) for c in df[country_col]]
    return [_AEZ_FALLBACK] * len(df)


# ---------------------------------------------------------------------------
# Regression line + 95% confidence interval
# ---------------------------------------------------------------------------

def _fit_and_ci(ax, x: np.ndarray, y: np.ndarray, xscale_log: bool = True) -> None:
    """OLS regression line + 95% CI band on ax (always log10(x) fit)."""
    x_fit = np.log10(x) if xscale_log else x
    fit = linregress(x_fit, y)
    x_line_fit = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_line = fit.intercept + fit.slope * x_line_fit

    n = len(x_fit)
    x_mean = x_fit.mean()
    ssx = np.sum((x_fit - x_mean) ** 2)
    if n > 2 and ssx > 0:
        residuals = y - (fit.intercept + fit.slope * x_fit)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        t_val = t.ppf(0.975, df=n - 2)
        se_mean = s_err * np.sqrt(1 / n + (x_line_fit - x_mean) ** 2 / ssx)
        ci_band = t_val * se_mean
        y_lower, y_upper = y_line - ci_band, y_line + ci_band
    else:
        y_lower = y_upper = y_line

    x_line = 10 ** x_line_fit if xscale_log else x_line_fit
    ax.plot(x_line, y_line, color="black", linewidth=2.5, zorder=3)
    ax.fill_between(x_line, y_lower, y_upper, color="grey", alpha=0.2, zorder=2)


# ---------------------------------------------------------------------------
# Figure 1: RSWCI bar chart with greenery overlay
# ---------------------------------------------------------------------------

def plot_waste_with_green_overlay(df: pd.DataFrame, y_col: str, outdir: Path) -> None:
    """
    Bar chart of waste_pct (descending) with green_pct overlaid as a line on
    a secondary y-axis.  Matches plots.ipynb plot_waste_with_green_overlay().
    """
    green_col = pick_col(df, ["green_pct"])

    df_s = df.sort_values(y_col, ascending=False).copy()
    n = len(df_s)

    # Tick labels from name / oam_id
    id_col = pick_col(df_s, ["name", "oam_id", "file_id"])
    x_labels = list(df_s[id_col]) if id_col else [str(i) for i in range(n)]

    colours = _bar_colours(df_s)

    fig, ax1 = plt.subplots(figsize=(14, 8), facecolor="white")
    ax1.set_facecolor("white")

    ax1.bar(range(n), df_s[y_col], color=colours,
            edgecolor="black", linewidth=0.6, width=0.65, zorder=2)
    ax1.set_ylabel("Relative solid waste contamination index [%]", fontsize=12, labelpad=15)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(axis="both", labelsize=11)

    legend_elements: list = []

    # Country legend if no DN
    if "DN" not in df_s.columns:
        country_col = pick_col(df_s, ["country", "new_country"])
        if country_col:
            seen: dict[str, str] = {}
            for c, col in zip(df_s[country_col], colours):
                seen.setdefault(str(c), col)
            legend_elements = [
                mpatches.Patch(facecolor=col, edgecolor="black", label=name)
                for name, col in seen.items()
            ]

    if green_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(range(n), df_s[green_col], color="#006400",
                 linewidth=2.5, marker="o", markersize=4, zorder=1)
        ax2.set_ylabel("Vegetation coverage [%]", fontsize=12, labelpad=15)
        ax2.tick_params(axis="y", labelsize=11)
        ax2.spines["top"].set_visible(False)
        max_green = df_s[green_col].max()
        ax2.set_ylim(0, max(max_green * 1.1 if pd.notna(max_green) else 100, 100))
        legend_elements.append(
            Line2D([0], [0], color="#006400", lw=2, marker="o", label="Vegetation coverage")
        )

    if legend_elements:
        ax1.legend(handles=legend_elements, loc="upper right",
                   frameon=False, fontsize=10, title_fontsize=11)

    fig.tight_layout()
    out = outdir / "waste_green_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Three-panel scatter
# ---------------------------------------------------------------------------

def plot_scatter_panels(df: pd.DataFrame, y_col: str, outdir: Path) -> None:
    """
    Three panels (SHDI | block complexity | pop density) vs RSWCI.
    All x-axes are log scale.  Matches plots.ipynb cell 74.
    """
    shdi_col = pick_col(df, ["shdi", "SHDI"])
    k_col    = pick_col(df, ["k_complexity_weighted", "k_complexity"])
    pop_col  = pick_col(df, [
        "worldpop_population_un_density_hectare_weighted",
        "worldpop_population_un_density_hectare",
        "pop_density",
    ])

    missing = [name for name, col in [
        ("shdi", shdi_col), ("k_complexity_weighted", k_col), ("pop_density", pop_col)
    ] if col is None]
    if missing:
        print(f"  Missing columns for scatter panels: {missing}  skipping.")
        print(f"   Available: {list(df.columns)}")
        return

    plot_df = df[[shdi_col, k_col, pop_col, y_col]].copy()
    plot_df = plot_df[
        (plot_df[shdi_col] > 0) & (plot_df[k_col] > 0) & (plot_df[pop_col] > 0)
    ].dropna()

    if len(plot_df) < 3:
        print(f"  Only {len(plot_df)} valid rows after filtering  cannot plot.")
        return

    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor="white")

    y_vals = plot_df[y_col].to_numpy(float)
    y_lo   = y_vals.min() - 1
    y_hi   = y_vals.max() + 1

    panels = [
        (shdi_col, "Subnational Human Development Index", 0),
        (k_col,    "Block complexity",                    1),
        (pop_col,  "Population density [individuals/ha]", 2),
    ]

    for col, label, idx in panels:
        ax     = axes[idx]
        x_vals = plot_df[col].to_numpy(float)

        ax.scatter(x_vals, y_vals, c="black", s=100, alpha=0.7,
                   edgecolors="white", linewidth=0.5, zorder=4)
        _fit_and_ci(ax, x_vals, y_vals, xscale_log=True)

        rho, p = spearmanr(x_vals, y_vals)
        p_str  = "< 0.001" if p < 0.001 else f"{p:.3f}"
        ax.text(
            0.95, 0.99,
            f"Spearman \u03c1 = {rho:+.3f}\n$p$ = {p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=15, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                      boxstyle="round,pad=0.5"),
        )

        ax.set_xscale("log")
        ax.set_xlim(x_vals.min() / 1.3, x_vals.max() * 1.3)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(label, fontsize=20, labelpad=12)

        if idx == 0:
            ax.set_ylabel("Relative solid waste contamination index",
                          fontsize=20, labelpad=12)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

        ax.tick_params(axis="both", which="major", labelsize=13, length=8, width=2)
        ax.tick_params(axis="both", which="minor", length=4, width=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    out = outdir / "three_panel_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {out}")
    print(f"  Points plotted: {len(plot_df)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate waste detection figures from AOI.gpkg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required columns in input file:\n"
            "  waste_pct / rswci\n"
            "  shdi\n"
            "  k_complexity_weighted\n"
            "  worldpop_population_un_density_hectare_weighted\n"
            "  green_pct  (for bar overlay)\n"
        ),
    )
    parser.add_argument(
        "--input",
        default="data/AOI.gpkg",
        help="GeoPackage or CSV with per-AOI metrics (default: data/AOI.gpkg)",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Output directory for figures (default: results/figures)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if in_path.suffix.lower() == ".gpkg":
        if not HAS_GEO:
            raise ImportError("geopandas is required for GPKG input  pip install geopandas")
        gdf = gpd.read_file(in_path)
        df  = pd.DataFrame(gdf.drop(columns="geometry"))
    else:
        df = pd.read_csv(in_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y_col = pick_col(df, ["waste_pct", "rswci", "areal_fraction"])
    if y_col is None:
        raise ValueError(
            "No RSWCI column found. Expected: waste_pct, rswci, or areal_fraction. "
            f"Available: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} AOIs from {in_path}  (y={y_col})")

    plot_waste_with_green_overlay(df, y_col, outdir)
    plot_scatter_panels(df, y_col, outdir)

    print(f"\nFigures saved to: {outdir}")


if __name__ == "__main__":
    main()

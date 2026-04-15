#!/usr/bin/env python3
"""
03_analysis/01_calculate_aoi_metrics.py
=======================================
Compute per-AOI RSWCI, green coverage, and weighted urban metrics,
then write the final ``data/AOI.gpkg`` used in all downstream analysis.

This is the single script that brings all inference streams together:
  - Tile GeoPackages from ``02_predict.py`` (pred_class, pred_class_green columns)
  - MillionNeighborhoods Africa parquet (urban morphology)
  - OAM catalog (AOI footprint geometries)
  - Global Data Lab Subnational HDI (auto-downloaded from Zenodo)

RSWCI
-----
    waste_pct = N(pred_class == "waste") / N(tiles) * 100

    Waste classification is the raw YOLO model output (pred_class column set
    by ``02_predict.py``).

Green coverage
--------------
    green_pct = N(pred_class_green == "green") / N(tiles) * 100

    Tiles are marked green by ``02_predict.py --sam greenery`` when the SAM
    polygon covers >= 25 % of the tile area (pred_class_green column in the
    tile GeoPackage).

Urban morphology weighting
---------------------------
    w_i            = intersection_area(block_i, AOI_extent) / block_area_i
    metric_weighted = sum(metric_i * w_i) / sum(w_i)

Blocks with w_i < 0.01 (edge slivers) are excluded.

Output columns in data/AOI.gpkg
--------------------------------
  oam_id
  geometry                                         AOI footprint (WGS-84)
  waste_pct                                        RSWCI (%)
  green_pct                                        vegetation coverage (%)
  k_complexity_weighted                            weighted street-network complexity
  worldpop_population_un_density_hectare_weighted  weighted pop density [persons/ha]
  shdi                                             Subnational HDI

Usage
-----
    python 03_analysis/01_calculate_aoi_metrics.py

    python 03_analysis/01_calculate_aoi_metrics.py --shdi data/auxiliary/shdi_national.csv

    python 03_analysis/01_calculate_aoi_metrics.py --shdi no
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import difflib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script's repo root
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

# Parquet columns to aggregate as coverage-weighted means.
# Only columns that are actually used in downstream analysis are listed here.
# Add further column names from africa_geodata.parquet if needed.
WEIGHTED_COLS = [
    "k_complexity",
    "worldpop_population_un_density_hectare",
]

# Zenodo record hosting the Global Data Lab SHDI dataset.
SHDI_ZENODO_RECORD_ID = 17467221
SHDI_FILE_KEY         = "Subnational HDI Data v8.3.csv"
SHDI_DEFAULT_PATH     = DATA / "auxiliary" / "shdi_national.csv"


def _fuzzy_country(name: str, candidates: set[str]) -> str:
    """Return the closest GDL country name for *name* using fuzzy matching.

    Falls back to the original string if nothing scores above 0.6 so that
    the caller can detect the miss rather than silently producing wrong data.
    """
    if name in candidates:
        return name
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return name



# ---------------------------------------------------------------------------
# SHDI helpers
# ---------------------------------------------------------------------------

def _download_shdi_from_zenodo(save_path: Path | None = None) -> pd.DataFrame:
    """
    Download the Global Data Lab Subnational HDI dataset from Zenodo and
    return a DataFrame with columns ``country`` (AOI-side name) and ``shdi``
    using the **latest available year** at national level per country.

    The result is optionally cached to *save_path* so subsequent runs do
    not need network access.

    Source: https://zenodo.org/records/17467221
    File  : Subnational HDI Data v8.3.csv
    """
    print(f"  Downloading SHDI from Zenodo record {SHDI_ZENODO_RECORD_ID} ...")
    record_url = f"https://zenodo.org/api/records/{SHDI_ZENODO_RECORD_ID}"
    with urllib.request.urlopen(record_url, timeout=60) as resp:
        record = json.loads(resp.read().decode())

    csv_url = None
    for f in record.get("files", []):
        if f.get("key") == SHDI_FILE_KEY:
            csv_url = f["links"]["self"].replace(" ", "%20")
            break
    if csv_url is None:
        raise RuntimeError(
            f"Could not locate '{SHDI_FILE_KEY}' in Zenodo record "
            f"{SHDI_ZENODO_RECORD_ID}.  Check the record URL manually."
        )

    with urllib.request.urlopen(csv_url, timeout=180) as resp:
        raw_bytes = resp.read()

    shdi_raw = pd.read_csv(io.BytesIO(raw_bytes))
    print(f"  Downloaded {len(shdi_raw):,} rows.")

    # Keep only national-level rows and take the latest year per country
    shdi_nat = shdi_raw[shdi_raw["level"] == "National"].copy()
    shdi_nat = (
        shdi_nat.sort_values("year", ascending=False)
        .drop_duplicates(subset=["country"], keep="first")
        [["country", "shdi", "year"]]
        .rename(columns={"country": "country_gdl", "year": "shdi_year"})
    )

    # Keep GDL names as-is; fuzzy matching happens at join time.
    shdi_nat["country"] = shdi_nat["country_gdl"]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shdi_nat.to_csv(save_path, index=False)
        print(f"  Cached SHDI to: {save_path}")

    return shdi_nat[["country", "shdi", "shdi_year"]]


def _load_shdi(shdi_arg: str) -> pd.DataFrame | None:
    """
    Resolve the --shdi argument and return a DataFrame with at minimum
    ``country`` and ``shdi`` columns, or None if SHDI is disabled.

    shdi_arg values:
        "auto"   — download from Zenodo (cache to data/auxiliary/shdi_national.csv)
        "no"     — skip SHDI entirely
        <path>   — load from a pre-downloaded CSV
    """
    if shdi_arg.lower() == "no":
        return None

    if shdi_arg.lower() == "auto":
        # Use cached file if it already exists
        if SHDI_DEFAULT_PATH.exists():
            print(f"  [SHDI] Using cached file: {SHDI_DEFAULT_PATH}")
            df = pd.read_csv(SHDI_DEFAULT_PATH)
        else:
            df = _download_shdi_from_zenodo(save_path=SHDI_DEFAULT_PATH)
        return df

    p = Path(shdi_arg)
    if not p.exists():
        raise FileNotFoundError(f"SHDI CSV not found: {p}")
    df = pd.read_csv(p)
    # Accept either country-keyed or oam_id-keyed CSVs
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coverage_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean: sum(v_i * w_i) / sum(w_i).  Returns NaN if sum(w)=0."""
    w_total = weights.sum()
    if w_total == 0:
        return float("nan")
    return float(np.dot(values, weights) / w_total)


# ---------------------------------------------------------------------------
# Tile-based RSWCI + green coverage
# ---------------------------------------------------------------------------

def _compute_rswci_from_tiles(
    tiles_dir: Path,
) -> pd.DataFrame:
    """
    Scan *tiles_dir* for ``*_tiles.gpkg`` files produced by ``02_predict.py``,
    compute per-AOI RSWCI from the YOLO waste predictions, and return a
    DataFrame with columns::

        oam_id, waste_pct, crs, minx, miny, maxx, maxy
    """
    gpkg_files = sorted(tiles_dir.glob("*_tiles.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError(
            f"No *_tiles.gpkg files found in: {tiles_dir}\n"
            "  Run 02_predict.py first, or pass --tiles-dir."
        )

    records: list[dict] = []
    for p in tqdm(gpkg_files, desc="Scanning tile GPKGs"):
        oam_id = p.stem.replace("_tiles", "")
        try:
            gdf = gpd.read_file(p)
        except Exception as exc:
            tqdm.write(f"  WARNING: could not read {p.name}: {exc}")
            continue
        if gdf.empty:
            tqdm.write(f"  WARNING: empty GPKG: {p.name}")
            continue

        total   = len(gdf)
        n_waste = int((gdf["pred_class"] == "waste").sum()) if "pred_class" in gdf.columns else 0
        bounds  = gdf.total_bounds  # [minx, miny, maxx, maxy]

        # green_pct from pred_class_green written by 02_predict.py --sam greenery
        if "pred_class_green" in gdf.columns:
            n_green = int((gdf["pred_class_green"] == "green").sum())
            green_pct = round(n_green / total * 100, 4) if total else 0.0
        else:
            green_pct = float("nan")

        rec: dict = dict(
            oam_id    = oam_id,
            waste_pct = round(n_waste / total * 100, 4) if total else 0.0,
            green_pct = green_pct,
            crs       = str(gdf.crs),
            minx=bounds[0], miny=bounds[1],
            maxx=bounds[2], maxy=bounds[3],
        )

        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(
        f"  {len(df)} AOIs  "
        f"(mean RSWCI={df['waste_pct'].mean():.2f}%  "
        f"range={df['waste_pct'].min():.2f}\u2013{df['waste_pct'].max():.2f}%)"
    )
    return df


def compute_weighted_metrics(
    oam_id: str,
    aoi_bounds: tuple[float, float, float, float],
    aoi_crs: str,
    urban_gdf: gpd.GeoDataFrame,
) -> dict[str, float]:
    """
    Compute coverage-weighted mean for each column in WEIGHTED_COLS.

    Weighting:
        w_i = intersection_area(block_i, AOI_extent) / block_area_i

    A block that is 100 % inside the AOI gets w_i = 1.0.
    A block that straddles the boundary with 30 % inside gets w_i = 0.3.
    Blocks with w_i < 0.01 are excluded.
    """
    result = {f"{c}_weighted": float("nan") for c in WEIGHTED_COLS}

    aoi_extent = box(*aoi_bounds)

    # Reproject parquet to match the AOI's CRS
    if urban_gdf.crs and str(urban_gdf.crs) != aoi_crs:
        try:
            urban_local = urban_gdf.to_crs(aoi_crs)
        except Exception:
            urban_local = urban_gdf
    else:
        urban_local = urban_gdf

    # Candidate blocks via spatial index
    candidate_idx = list(urban_local.sindex.intersection(aoi_bounds))
    if not candidate_idx:
        return result

    cands = urban_local.iloc[candidate_idx].copy()
    cands["block_area"] = cands.geometry.area

    # Weight = fraction of block that falls inside the AOI extent
    cands["intersect_area"] = cands.geometry.apply(
        lambda g: g.intersection(aoi_extent).area
    )
    cands["w"] = (
        cands["intersect_area"]
        / cands["block_area"].replace(0, float("nan"))
    ).fillna(0.0)

    # Drop slivers
    cands = cands[cands["w"] > 0.01]
    if cands.empty:
        return result

    weights = cands["w"].values
    for col in WEIGHTED_COLS:
        if col not in cands.columns:
            continue
        vals  = pd.to_numeric(cands[col], errors="coerce").values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue
        result[f"{col}_weighted"] = _coverage_weighted_mean(
            vals[valid], weights[valid]
        )

    return result


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def calculate_aoi_metrics(
    tiles_dir:  Path,
    parquet:    Path,
    catalog:    Path,
    shdi_arg:   str,
    output:     Path,
) -> gpd.GeoDataFrame:
    """
    Compute RSWCI and green coverage from tile GPKGs (pred_class and
    pred_class_green columns written by 02_predict.py), join all auxiliary
    metrics, and write data/AOI.gpkg.  Returns the GeoDataFrame.
    """

    # 1 ── RSWCI + bounding boxes from tile GPKGs ───────────────────────────
    print(f"\n[1/5] Computing RSWCI from tiles: {tiles_dir}")
    rswci_df = _compute_rswci_from_tiles(tiles_dir)
    if rswci_df.empty:
        raise RuntimeError(f"No valid tile GPKGs found in: {tiles_dir}")
    print(f"  {len(rswci_df)} valid AOIs")

    # 2 ── OAM catalog (footprint geometry) ─────────────────────────────────
    print(f"\n[2/5] Loading catalog: {catalog}")
    cat = gpd.read_file(catalog)
    id_col = "oam_id" if "oam_id" in cat.columns else "uuid"
    cat = cat.rename(columns={id_col: "oam_id"})
    cat_crs = str(cat.crs)
    print(f"  {len(cat)} rows, CRS={cat_crs}")

    # SHDI ── resolve immediately after catalog so country col is available
    print(f"\n  [SHDI] Resolving (--shdi={shdi_arg!r}) ...")
    shdi_df = _load_shdi(shdi_arg)

    # 3 ── Urban morphology parquet ─────────────────────────────────────────
    print(f"\n[3/5] Loading parquet: {parquet}")
    if not parquet.exists():
        print(f"  NOT FOUND — run 01_data_acquisition_preprocessing/"
              f"04_download_auxiliary_data.py first.")
        urban_gdf = None
    else:
        urban_gdf = gpd.read_parquet(parquet)
        print(f"  {len(urban_gdf):,} blocks, CRS={urban_gdf.crs}")

    # 4 ── Per-AOI stats ────────────────────────────────────────────────────
    print(f"\n[4/5] Computing per-AOI metrics ({len(rswci_df)} AOIs) ...")
    rows: list[dict] = []

    for _, r in tqdm(rswci_df.iterrows(), total=len(rswci_df)):
        oam_id = str(r["oam_id"])
        row: dict = {
            "oam_id":    oam_id,
            "waste_pct": r["waste_pct"],
            "green_pct": r["green_pct"],   # from pred_class_green in tile GPKG
        }

        bounds  = (r["minx"], r["miny"], r["maxx"], r["maxy"])
        aoi_crs = r.get("crs") or cat_crs

        # Coverage-weighted urban metrics
        if urban_gdf is not None:
            row.update(
                compute_weighted_metrics(oam_id, bounds, aoi_crs, urban_gdf)
            )
        else:
            for col in WEIGHTED_COLS:
                row[f"{col}_weighted"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)

    # 5 ── Join geometry + SHDI, write GPKG ─────────────────────────────────
    print(f"\n[5/5] Writing AOI.gpkg ...")

    merged = cat[["oam_id", "geometry"]].merge(df, on="oam_id", how="inner")
    if merged.empty:
        print("  WARNING: no AOIs matched between RSWCI CSV and catalog.")
        return gpd.GeoDataFrame()

    # SHDI join — by country column from catalog
    if shdi_df is not None:
        country_col = next(
            (c for c in cat.columns if c.lower() in ("country", "country_name")),
            None,
        )
        if country_col and country_col in merged.columns:
            # Fuzzy-match catalog country names to GDL names at join time.
            _gdl_names: set[str] = set(shdi_df["country"].astype(str))
            merged["_country_norm"] = merged[country_col].astype(str).map(
                lambda c: _fuzzy_country(c, _gdl_names)
            )
            # Warn for any names that could not be matched.
            unmatched = (
                merged.loc[
                    ~merged["_country_norm"].isin(_gdl_names),
                    country_col,
                ].dropna().unique()
            )
            if len(unmatched):
                print(
                    f"  ⚠ Country names not fuzzy-matched to GDL "
                    f"(SHDI will be NaN): {list(unmatched)}"
                )
            # The shdi_df already has an AOI-side "country" column if downloaded
            # via _load_shdi(); fall back to merging on the normalised name.
            shdi_join = shdi_df.copy()
            if "country" not in shdi_join.columns:
                # Pre-downloaded CSV may have country_gdl or a different key —
                # try to detect it
                ckey = next(
                    (c for c in shdi_join.columns if "country" in c.lower()),
                    shdi_join.columns[0],
                )
                shdi_join = shdi_join.rename(columns={ckey: "country"})
            merged = merged.merge(
                shdi_join[["country", "shdi"]].rename(columns={"country": "_country_norm"}),
                on="_country_norm",
                how="left",
            ).drop(columns=["_country_norm"])
            print(f"  SHDI joined: {merged['shdi'].notna().sum()}/{len(merged)} AOIs")
        else:
            print(
                "  ⚠ No 'country' column found in catalog — "
                "SHDI will be NaN for all AOIs."
            )
            merged["shdi"] = float("nan")

    # Column order — geometry always last so GPKG writes cleanly
    ordered = [
        "oam_id", "waste_pct", "green_pct",
    ]
    ordered += [f"{c}_weighted" for c in WEIGHTED_COLS if f"{c}_weighted" in merged.columns]
    if "shdi" in merged.columns:
        ordered += ["shdi"]
    ordered += ["geometry"]

    # Ensure any extra columns (future additions) are not silently dropped
    extra = [c for c in merged.columns if c not in ordered]
    if extra:
        ordered = ordered[:-1] + extra + ["geometry"]

    gdf_out = gpd.GeoDataFrame(
        merged[[c for c in ordered if c in merged.columns]],
        crs=cat.crs,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(output, driver="GPKG", layer="AOI_metrics")
    print(f"\nAOI.gpkg saved: {output}  ({len(gdf_out)} AOIs)")

    # Summary
    print(f"\n{'=' * 60}")
    summary_cols = [c for c in [
        "waste_pct", "green_pct",
        "k_complexity_weighted",
        "worldpop_population_un_density_hectare_weighted",
    ] if c in gdf_out.columns]
    for col in summary_cols:
        vals = gdf_out[col].dropna()
        if len(vals):
            print(
                f"  {col:<55s}  "
                f"mean={vals.mean():.3f}  "
                f"range={vals.min():.3f}–{vals.max():.3f}  "
                f"n={len(vals)}"
            )
    print(f"{'=' * 60}")

    return gdf_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-AOI RSWCI and all auxiliary metrics, then write "
            "data/AOI.gpkg from the MillionNeighborhoods urban morphology parquet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run order:\n"
            "  02_predict.py                        -> data/tiles/*_tiles.gpkg\n"
            "  03_analysis/01_calculate_aoi_metrics.py -> data/AOI.gpkg\n"
        ),
    )
    parser.add_argument(
        "--tiles-dir",
        default=str(DATA / "tiles"),
        help="Folder containing *_tiles.gpkg files from 02_predict.py "
             "(default: data/tiles/)",
    )
    parser.add_argument(
        "--parquet",
        default=str(DATA / "auxiliary" / "africa_geodata.parquet"),
        help="MillionNeighborhoods Africa parquet "
             "(default: data/auxiliary/africa_geodata.parquet)",
    )
    parser.add_argument(
        "--catalog",
        default=str(DATA / "oam_catalog.gpkg"),
        help="OAM catalog GeoPackage for AOI footprint geometry "
             "(default: data/oam_catalog.gpkg)",
    )
    parser.add_argument(
        "--shdi",
        default="auto",
        metavar="auto|no|PATH",
        help="How to obtain Subnational HDI values. "
             "'auto' (default) downloads from Zenodo record 17467221 and caches "
             "to data/auxiliary/shdi_national.csv. "
             "A file path loads a pre-downloaded CSV with columns: country, shdi.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA / "AOI.gpkg"),
        help="Output GeoPackage path (default: data/AOI.gpkg)",
    )
    args = parser.parse_args()

    calculate_aoi_metrics(
        tiles_dir  = Path(args.tiles_dir),
        parquet    = Path(args.parquet),
        catalog    = Path(args.catalog),
        shdi_arg   = args.shdi,
        output     = Path(args.output),
    )


if __name__ == "__main__":
    main()

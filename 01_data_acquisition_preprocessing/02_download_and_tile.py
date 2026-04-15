#!/usr/bin/env python3
"""
02_download_and_tile.py
=======================
Download UAV GeoTIFFs from the selected AOI GeoPackage, then generate a
5 × 5 m vector tile grid for each scene for use in manual labeling and
dataset creation.

Pipeline
--------
1. Read ``data/oam_AOI.gpkg`` (built by the QGIS manual selection step).
2. Download each GeoTIFF to ``data/imagery/`` as ``<oam_id>.tif``.
3. For every downloaded TIF, build a 5 × 5 m regular grid clipped to the
   raster extent and save it as ``data/tiles/<oam_id>_tiles.gpkg``.

   Once labeling is complete, open each GPKG in QGIS, select tiles, and
   set the ``label`` column (0 / 1) or ``class`` column ("waste" /
   "background"). Save, then run ``03_create_yolo_dataset.py``.

Overlapping scenes
------------------
If two downloaded scenes overlap spatially they can be merged in QGIS
using **Raster > Miscellaneous > Merge** (gdal_merge) before tiling.
The merged output should be renamed ``<oam_id>_merged.tif`` and placed
in ``data/imagery/``; ``03_create_yolo_dataset.py`` will find it
automatically.

Usage
-----
    # Download + tile (default):
    python 01_data_acquisition_preprocessing/02_download_and_tile.py

    # Use a different AOI GPKG or output directories:
    python 01_data_acquisition_preprocessing/02_download_and_tile.py \\
        --gpkg  data/oam_AOI.gpkg \\
        --outdir data/imagery/ \\
        --tiles-dir data/tiles/

Requirements: geopandas, rasterio, shapely, pyproj, requests, tqdm
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from shapely.geometry import box
from tqdm import tqdm

try:
    import rasterio
    from rasterio.crs import CRS
except ImportError:
    raise SystemExit("rasterio is required: pip install rasterio")

try:
    from pyproj import Transformer
except ImportError:
    raise SystemExit("pyproj is required: pip install pyproj")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def is_url(value: str) -> bool:
    try:
        p = urlparse(str(value))
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def create_session(timeout: int = 300, retries: int = 5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "waste-detection-ssa/1.0 (+https://openaerialmap.org)",
        "Accept": "*/*",
    })
    return session


def filename_from_url(url: str, fallback: str = "download.tif") -> str:
    try:
        name = Path(urlparse(url).path).name
        return name if name else fallback
    except Exception:
        return fallback


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    timeout: int,
) -> tuple[bool, Optional[str]]:
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            total = int(r.headers.get("Content-Length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total or None, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as pbar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True, None
    except Exception as e:
        return False, str(e)


def download_all(
    gdf: gpd.GeoDataFrame,
    url_col: str,
    outdir: Path,
    timeout: int,
    overwrite: bool,
    log_path: Path,
    id_col: Optional[str] = None,
) -> list[dict]:
    """Download all URLs; save as ``<oam_id>.tif``.  Return list of log dicts."""
    session = create_session(timeout=timeout)
    logs: list[dict] = []

    print(f"\nDownloading {len(gdf)} GeoTIFFs → {outdir}/")
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading", unit="file"):
        url_val = row.get(url_col)
        if pd.isna(url_val) or not str(url_val).strip():
            logs.append({"idx": idx, "url": None, "file": None,
                         "status": "skip-empty", "error": "empty URL"})
            continue

        url = str(url_val).strip()
        if not is_url(url):
            logs.append({"idx": idx, "url": url, "file": None,
                         "status": "skip-not-url", "error": "not a URL"})
            continue

        # Prefer <oam_id>.tif  over the file UUID embedded in the download URL
        if id_col and id_col in gdf.columns:
            oid = str(row.get(id_col, "")).strip()
            fname = f"{oid}.tif" if oid else filename_from_url(url)
        else:
            fname = filename_from_url(url)
        dest = outdir / fname

        if dest.exists() and not overwrite:
            logs.append({"idx": idx, "url": url, "file": str(dest),
                         "status": "skip-exists", "error": None})
            continue

        ok, err = download_file(session, url, dest, timeout)
        logs.append({
            "idx": idx, "url": url,
            "file": str(dest) if ok else None,
            "status": "ok" if ok else "error",
            "error": err,
        })
        if not ok:
            print(f"  ✗ {fname}: {err}", file=sys.stderr)
            if dest.exists():
                dest.unlink()

    # Write log
    if logs:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "url", "file", "status", "error"])
            writer.writeheader()
            writer.writerows(logs)
        print(f"  Download log: {log_path}")

    n_ok   = sum(1 for l in logs if l["status"] == "ok")
    n_skip = sum(1 for l in logs if l["status"].startswith("skip"))
    n_err  = sum(1 for l in logs if l["status"] == "error")
    print(f"  Done — {n_ok} downloaded, {n_skip} skipped, {n_err} failed")
    return logs


# ---------------------------------------------------------------------------
# Tile-grid creation
# ---------------------------------------------------------------------------

def _best_utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the best UTM zone for a given WGS-84 point."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def create_tile_grid(
    tif_path: Path,
    out_gpkg: Path,
    tile_m: float = 5.0,
    overwrite: bool = False,
) -> bool:
    """
    Build a 5 × 5 m regular grid over a GeoTIFF and save it as a GeoPackage.

    The grid is computed in the native projected CRS of the raster when it is
    already metric, or in the best-fit UTM zone when the raster is in
    geographic coordinates (degrees).

    Parameters
    ----------
    tif_path : path to the source GeoTIFF
    out_gpkg : output path for the tile-grid GeoPackage
    tile_m   : tile side-length in metres (default 5)
    overwrite: overwrite existing output file
    """
    if out_gpkg.exists() and not overwrite:
        print(f"  Tile grid already exists: {out_gpkg.name} — skipping")
        return True

    oam_id = tif_path.stem
    try:
        with rasterio.open(tif_path) as src:
            native_crs = src.crs
            bounds_native = src.bounds          # left, bottom, right, top
            width_px  = src.width
            height_px = src.height

        # ------------------------------------------------------------------
        # Choose a metric CRS for tile placement
        # ------------------------------------------------------------------
        native_is_geographic = native_crs.is_geographic

        if native_is_geographic:
            # Reproject the centre to WGS-84 and pick best UTM
            cx = (bounds_native.left + bounds_native.right)  / 2
            cy = (bounds_native.bottom + bounds_native.top)  / 2
            utm_epsg = _best_utm_epsg(cx, cy)
            metric_crs = CRS.from_epsg(utm_epsg)
        else:
            metric_crs = native_crs
            utm_epsg = None                    # already metric

        # ------------------------------------------------------------------
        # Convert raster bounds to metric CRS
        # ------------------------------------------------------------------
        if native_is_geographic:
            transformer = Transformer.from_crs(native_crs, metric_crs,
                                               always_xy=True)
            xs = [bounds_native.left,  bounds_native.right,
                  bounds_native.right, bounds_native.left]
            ys = [bounds_native.bottom, bounds_native.bottom,
                  bounds_native.top,   bounds_native.top]
            txs, tys = transformer.transform(xs, ys)
            min_x, max_x = min(txs), max(txs)
            min_y, max_y = min(tys), max(tys)
        else:
            min_x, min_y = bounds_native.left,  bounds_native.bottom
            max_x, max_y = bounds_native.right, bounds_native.top

        # ------------------------------------------------------------------
        # Build grid
        # ------------------------------------------------------------------
        cols_arr = np.arange(min_x, max_x, tile_m)
        rows_arr = np.arange(min_y, max_y, tile_m)

        n_total = len(cols_arr) * len(rows_arr)
        if n_total == 0:
            print(f"  ⚠  Zero tiles generated for {tif_path.name} — extent too small?")
            return False

        print(f"  Building {len(rows_arr)} × {len(cols_arr)} = {n_total:,} tiles "
              f"({tile_m} m grid,  CRS: {metric_crs.to_epsg() or metric_crs})")

        geoms: list     = []
        tile_ids: list  = []
        rows_idx: list  = []
        cols_idx: list  = []
        filenames: list = []
        tid = 0
        for ci, x0 in enumerate(cols_arr):
            for ri, y0 in enumerate(rows_arr):
                geoms.append(box(x0, y0, x0 + tile_m, y0 + tile_m))
                tile_ids.append(tid)
                rows_idx.append(ri)
                cols_idx.append(ci)
                filenames.append(f"{oam_id}_{tid}.png")
                tid += 1

        gdf = gpd.GeoDataFrame(
            {
                "tile_id":  tile_ids,
                "oam_id":   oam_id,
                "row":      rows_idx,
                "col":      cols_idx,
                "filename": filenames,
                # label is set in QGIS before training: 1 = waste, 0 = background
                "label":    None,
                # added by 02_predict.py after inference:
                # "pred_class":  None,
                # "confidence":  None,
                # "green_pct":   None,
                # "water_pct":   None,
            },
            geometry=geoms,
            crs=metric_crs,
        )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_gpkg, driver="GPKG", layer="tiles")
        print(f"  ✓ Saved → {out_gpkg.name}  ({len(gdf):,} tiles)")
        return True

    except Exception as e:
        print(f"  ✗ Tile grid failed for {tif_path.name}: {e}", file=sys.stderr)
        return False


def tile_all(
    imagery_dir: Path,
    tiles_dir: Path,
    tile_m: float,
    overwrite: bool,
) -> None:
    """Create tile grids for all TIFs in imagery_dir."""
    tif_paths = sorted(imagery_dir.glob("*.tif"))
    if not tif_paths:
        print(f"\n⚠  No TIF files found in {imagery_dir}")
        return

    print(f"\nCreating {tile_m} m tile grids for {len(tif_paths)} GeoTIFF(s) → {tiles_dir}/")
    ok = fail = skip = 0
    for tif in tqdm(tif_paths, desc="Tiling", unit="scene"):
        out = tiles_dir / f"{tif.stem}_tiles.gpkg"
        result = create_tile_grid(tif, out, tile_m=tile_m, overwrite=overwrite)
        if result:
            if out.exists():
                ok += 1
            else:
                skip += 1
        else:
            fail += 1

    print(f"\n  Tiling summary: {ok} created, {skip} skipped (already exist), {fail} failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download OAM GeoTIFFs and build 5 × 5 m vector tile grids for "
            "manual labeling and YOLO dataset creation."
        )
    )

    # ── AOI selection (mutually exclusive approaches) ──────────────────────
    aoi_group = parser.add_mutually_exclusive_group()
    aoi_group.add_argument(
        "--oam-ids",
        help=(
            "Comma-separated OAM scene IDs to download.  The script looks "
            "these up in the full catalog (--catalog) so no QGIS step is "
            "needed.  Example: --oam-ids 5be9bf8ac6c3bf0005896106,59e62b8a..."
        ),
    )
    aoi_group.add_argument(
        "--gpkg", default=None,
        help=(
            "Pre-filtered AOI GeoPackage.  If omitted and --oam-ids is also "
            "not set the script falls back to data/oam_AOI.gpkg (legacy)."
        ),
    )
    parser.add_argument(
        "--catalog", default="data/oam_catalog.gpkg",
        help=(
            "Full OAM catalog GeoPackage produced by 01_query_oam_catalog.py "
            "(used when --oam-ids is supplied; default: data/oam_catalog.gpkg)"
        ),
    )
    parser.add_argument(
        "--id-column", default="oam_id",
        help="Column with OAM scene ID; downloaded files saved as <id>.tif "
             "(default: oam_id)"
    )
    parser.add_argument(
        "--url-column", default="download",
        help="Column containing download URLs (default: download)"
    )
    parser.add_argument(
        "--outdir", default="data/imagery",
        help="Directory for downloaded GeoTIFFs (default: data/imagery)"
    )
    parser.add_argument(
        "--tiles-dir", default="data/tiles",
        help="Directory for tile-grid GeoPackages (default: data/tiles)"
    )
    parser.add_argument(
        "--tile-size", type=float, default=5.0,
        help="Tile side-length in metres (default: 5)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download / re-tile files that already exist"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-request download timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--log", default=None,
        help="CSV path for download log (default: <outdir>/download_log.csv)"
    )
    args = parser.parse_args()

    gpkg_path  = Path(args.gpkg) if args.gpkg else Path("data/oam_AOI.gpkg")
    outdir     = Path(args.outdir)
    tiles_dir  = Path(args.tiles_dir)
    log_path   = Path(args.log) if args.log else outdir / "download_log.csv"

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    # ── Resolve AOI GeoDataFrame ──────────────────────────────────
    if args.oam_ids:
            # User passed --oam-ids: filter the full catalog, no QGIS needed.
        requested = [i.strip() for i in args.oam_ids.split(",") if i.strip()]
        catalog_path = Path(args.catalog)
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {catalog_path}\n"
                "Run 01_query_oam_catalog.py first."
            )
        cat = gpd.read_file(catalog_path)
        id_col = args.id_column if args.id_column in cat.columns else "oam_id"
        gdf = cat[cat[id_col].isin(requested)].copy()
        missing = set(requested) - set(gdf[id_col])
        if missing:
            print(f"  ⚠ IDs not found in catalog: {sorted(missing)}")
        print(f"Loaded {len(gdf)} AOIs from catalog (filtered by --oam-ids)")
    else:
        # Fall back to a pre-built AOI GeoPackage (legacy / paper reproduction).
        if not gpkg_path.exists():
            raise FileNotFoundError(
                f"GeoPackage not found: {gpkg_path}\n"
                "Either pass --oam-ids <id1,id2,...>  (recommended) or\n"
                "create the GeoPackage manually in QGIS from oam_catalog.gpkg."
            )
        gdf = gpd.read_file(gpkg_path)
        print(f"Loaded {len(gdf)} AOIs from {gpkg_path}")

    # Auto-detect URL column
    url_col = args.url_column
    if url_col not in gdf.columns:
        for alt in ["download", "uuid", "url", "tms_url"]:
            if alt in gdf.columns:
                url_col = alt
                print(f"  URL column '{args.url_column}' not found; using '{url_col}'")
                break
        else:
            raise ValueError(
                f"URL column '{url_col}' not found. "
                f"Available columns: {list(gdf.columns)}"
            )

    outdir.mkdir(parents=True, exist_ok=True)
    download_all(gdf, url_col, outdir, args.timeout, args.overwrite,
                 log_path, id_col=args.id_column)

    # ------------------------------------------------------------------
    # 2. Tile
    # ------------------------------------------------------------------
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_all(outdir, tiles_dir, tile_m=args.tile_size, overwrite=args.overwrite)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Next steps:")
    print()
    print("  1. (Optional) If scenes overlap, merge them in QGIS:")
    print("       Raster > Miscellaneous > Merge, save as <oam_id>_merged.tif")
    print("       in data/imagery/")
    print()
    print(f"  2. Open tile GeoPackages in QGIS from {tiles_dir}/")
    print("       For each <oam_id>_tiles.gpkg, select ~100 waste tiles")
    print("       and ~100 background tiles and set the 'label' column:")
    print("           1 = waste,  0 = background")
    print("       Save the GPKG from QGIS — do NOT rename it.")
    print()
    print("  3. Build the YOLO dataset:")
    print("       python 01_data_acquisition_preprocessing/03_create_yolo_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

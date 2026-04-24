#!/usr/bin/env python3
"""
02_predict.py
==============
Waste detection prediction script.

For each scene this script:
  1. Loads the 5 x 5 m tile grid GeoPackage (``<oam_id>_tiles.gpkg``).
  2. Runs the trained YOLO waste-classification model on every tile by
     cropping directly from the source GeoTIFF -- no pre-extract step needed.
     Each tile gets a ``pred_class`` (waste / background) and ``confidence``.
  3. Saves the enriched tile GeoDataFrame to:
         data/predictions/waste/<oam_id>_predictions.gpkg

All operations preserve the native CRS of the tile grid (projected UTM).

Usage
-----
    # Batch — waste classification only:
    python 02_model_training/02_predict.py \
        --imagery-dir data/imagery/ \
        --tiles-dir   data/tiles/ \
        --model       02_model_training/checkpoints/best.pt

    # Single scene:
    python 02_model_training/02_predict.py \
        --tif   data/imagery/59e62b8a3d6412ef72209d69.tif \
        --tiles data/tiles/59e62b8a3d6412ef72209d69_tiles.gpkg \
        --model 02_model_training/checkpoints/best.pt

Requirements
------------
    pip install ultralytics geopandas rasterio shapely numpy tqdm torch
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics + torch required: pip install ultralytics torch")

try:
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.windows import Window, from_bounds as window_from_bounds
    import rasterio.transform
    import rasterio.windows
except ImportError:
    raise SystemExit("rasterio is required: pip install rasterio")

try:
    from shapely.geometry import shape as shapely_shape
except ImportError:
    raise SystemExit("shapely is required: pip install shapely")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_NAMES      = ["background", "waste"]
YOLO_TILE_PX     = 128    # resize each 5 m crop to this for YOLO


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def _normalise_rgb(data: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) raster array to 3-band uint8 for YOLO."""
    def _scale(band: np.ndarray) -> np.ndarray:
        bmin, bmax = float(np.nanmin(band)), float(np.nanmax(band))
        if bmax > bmin:
            return ((band - bmin) / (bmax - bmin) * 255).astype(np.uint8)
        return np.zeros_like(band, dtype=np.uint8)

    n = data.shape[0]
    if n >= 3:
        return np.stack([_scale(data[i]) for i in range(3)], axis=0)
    elif n == 1:
        scaled = _scale(data[0])
        return np.stack([scaled, scaled, scaled], axis=0)
    else:
        scaled_bands = [_scale(data[i]) for i in range(n)]
        while len(scaled_bands) < 3:
            scaled_bands.append(np.zeros_like(scaled_bands[0]))
        return np.stack(scaled_bands[:3], axis=0)


def predict_tiles_yolo(
    tif_path: Path,
    tiles_gdf: gpd.GeoDataFrame,
    model: YOLO,
    tile_px: int,
) -> tuple:
    """
    Classify every tile in *tiles_gdf* by cropping from *tif_path*.
    Returns (pred_classes list, confidences list).
    """
    pred_classes = [None] * len(tiles_gdf)
    confidences  = [None] * len(tiles_gdf)

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        if tiles_gdf.crs != raster_crs:
            tiles_reproj = tiles_gdf.to_crs(raster_crs)
        else:
            tiles_reproj = tiles_gdf

        for i, (_, row) in enumerate(tqdm(
            tiles_reproj.iterrows(),
            total=len(tiles_reproj),
            desc="  YOLO", leave=False,
        )):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            window = window_from_bounds(minx, miny, maxx, maxy,
                                        transform=src.transform)
            if window.width < 1 or window.height < 1:
                continue
            try:
                data = src.read(window=window)
                if np.all(data == 0) or np.all(np.isnan(data)):
                    continue
                rgb = _normalise_rgb(data)
                img = Image.fromarray(np.moveaxis(rgb, 0, -1))
                img = img.resize((tile_px, tile_px), Image.LANCZOS)
                arr = np.array(img)
                result = model.predict(arr, verbose=False)[0]
                pred_classes[i] = CLASS_NAMES[result.probs.top1]
                confidences[i]  = float(result.probs.top1conf)
            except Exception:
                continue

    return pred_classes, confidences


# ---------------------------------------------------------------------------
# Per-scene orchestration
# ---------------------------------------------------------------------------

def process_scene(
    tif_path, tiles_gpkg, model, outdir, overwrite,
):
    oam_id   = tif_path.stem
    out_gpkg = outdir / f"{oam_id}_predictions.gpkg"

    if out_gpkg.exists() and not overwrite:
        print(f"  skip (exists): {out_gpkg.name}")
        return True

    if not tiles_gpkg.exists():
        print(f"  WARNING: tile GPKG not found: {tiles_gpkg.name} -- skipping")
        return False

    tiles_gdf = gpd.read_file(tiles_gpkg)
    print(f"  Tiles  : {len(tiles_gdf):,}  CRS: {tiles_gdf.crs}")

    # YOLO prediction
    print("  Running YOLO waste classification...")
    pred_classes, confidences = predict_tiles_yolo(
        tif_path, tiles_gdf, model, YOLO_TILE_PX
    )
    tiles_gdf["pred_class"] = pred_classes
    tiles_gdf["confidence"] = confidences
    waste_n = sum(1 for c in pred_classes if c == "waste")
    bg_n    = sum(1 for c in pred_classes if c == "background")
    none_n  = sum(1 for c in pred_classes if c is None)
    print(f"  YOLO   : {waste_n} waste, {bg_n} background, {none_n} skipped")

    # Save combined output
    outdir.mkdir(parents=True, exist_ok=True)
    tiles_gdf.to_file(out_gpkg, driver="GPKG", layer="predictions")
    print(f"  Saved: {out_gpkg.name}  ({len(tiles_gdf):,} tiles)")

    # Also write pred_class / confidence back to the source tile GPKG so that
    # 01_calculate_aoi_metrics.py can read them by scanning data/tiles/.
    try:
        tiles_gdf.to_file(tiles_gpkg, driver="GPKG", layer="tiles")
        print(f"  Updated source tile GPKG: {tiles_gpkg.name}")
    except Exception as e:
        print(f"  WARNING: could not update source tile GPKG ({e})")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO waste prediction. "
            "Adds pred_class, confidence to each 5 m tile and saves to data/predictions/waste/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    single = parser.add_mutually_exclusive_group(required=True)
    single.add_argument("--tif", type=Path, metavar="PATH",
                        help="Single GeoTIFF to process.")
    single.add_argument("--imagery-dir", type=Path, metavar="DIR",
                        help="Directory of GeoTIFFs (batch mode).")

    parser.add_argument("--tiles", type=Path, default=None, metavar="GPKG",
                        help="Tile GPKG for single scene (inferred from --tif stem).")
    parser.add_argument("--tiles-dir", type=Path, default=Path("data/tiles"),
                        help="Directory of *_tiles.gpkg files (batch). "
                             "(default: %(default)s)")
    parser.add_argument("--model", type=Path,
                        default=Path("02_model_training/checkpoints/best.pt"),
                        help="Trained YOLO best.pt checkpoint.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--outdir", type=Path,
                        default=Path("data/predictions/waste"),
                        help="Output directory. (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # Load YOLO model
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\n"
                 "Train first with 01_train_waste_classification.py.")
    print(f"Loading YOLO model: {args.model}")
    yolo_model = YOLO(str(args.model))
    print(f"  CUDA: {torch.cuda.is_available()}")

    # Collect TIF paths
    if args.tif:
        if not args.tif.exists():
            sys.exit(f"File not found: {args.tif}")
        tif_paths = [args.tif]
    else:
        if not args.imagery_dir.exists():
            sys.exit(f"Directory not found: {args.imagery_dir}")
        tif_paths = [p for p in sorted(args.imagery_dir.glob("*.tif"))
                     if not p.stem.startswith("_")]
        if not tif_paths:
            sys.exit(f"No TIF files found in {args.imagery_dir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"WASTE PREDICTION  ({len(tif_paths)} scene(s))")
    print(f"  YOLO model : {args.model}")
    print(f"  Device     : {args.device}")
    print(f"  Output     : {args.outdir}")
    print(f"{'='*60}")

    completed = failed = 0
    for k, tif in enumerate(tif_paths, 1):
        tiles_gpkg = (args.tiles if args.tiles and len(tif_paths) == 1
                      else args.tiles_dir / f"{tif.stem}_tiles.gpkg")
        print(f"\n[{k}/{len(tif_paths)}] {tif.name}")
        try:
            ok = process_scene(
                tif_path      = tif,
                tiles_gpkg    = tiles_gpkg,
                model         = yolo_model,
                outdir        = args.outdir,
                overwrite     = args.overwrite,
            )
            completed += 1 if ok else 0
            failed    += 0 if ok else 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done -- {completed} completed, {failed} failed")
    print(f"Outputs: {args.outdir.resolve()}")
    print(f"\nNext step:")
    print(f"  python 03_analysis/01_calculate_aoi_metrics.py")

if __name__ == "__main__":
    main()

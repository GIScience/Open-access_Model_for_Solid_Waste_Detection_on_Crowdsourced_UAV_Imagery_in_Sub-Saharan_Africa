#!/usr/bin/env python3
"""
02_predict.py
==============
Combined waste detection + SAM segmentation coverage script.

For each scene this script:
  1. Loads the 5 x 5 m tile grid GeoPackage (``<oam_id>_tiles.gpkg``).
  2. Runs the trained YOLO waste-classification model on every tile by
     cropping directly from the source GeoTIFF -- no pre-extract step needed.
     Each tile gets a ``pred_class`` (waste / background) and ``confidence``.
  3. Optionally runs Meta SAM 3 (via samgeo SamGeo3) to produce segmentation
     polygons for greenery across the full raster, using the text prompt
     "trees, bushes, vegetation" applied in 2048 × 2048 px windows.
  4. Intersects each 5 m tile with the SAM polygons and computes
     ``sam_greenery_pct`` -- the percentage of the tile area covered by the
     SAM greenery prediction polygons.
  5. Saves the enriched tile GeoDataFrame to:
         data/predictions/waste/<oam_id>_predictions.gpkg

All operations preserve the native CRS of the tile grid (projected UTM).

Usage
-----
    # Batch — waste classification only (no SAM):
    python 02_model_training/02_predict.py \
        --imagery-dir data/imagery/ \
        --tiles-dir   data/tiles/ \
        --model       02_model_training/checkpoints/best.pt

    # Batch — waste classification + greenery SAM:
    python 02_model_training/02_predict.py \
        --imagery-dir data/imagery/ \
        --tiles-dir   data/tiles/ \
        --model       02_model_training/checkpoints/best.pt \
        --sam         greenery

    # Single scene:
    python 02_model_training/02_predict.py \
        --tif   data/imagery/59e62b8a3d6412ef72209d69.tif \
        --tiles data/tiles/59e62b8a3d6412ef72209d69_tiles.gpkg \
        --model 02_model_training/checkpoints/best.pt

Requirements
------------
    pip install ultralytics geopandas rasterio shapely numpy tqdm torch
    pip install samgeo          # only needed when --sam is greenery
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
    from shapely.ops import unary_union
except ImportError:
    raise SystemExit("shapely is required: pip install shapely")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAM_TILE_SIZE    = 2048
SAM_MIN_SIZE     = 100
GREENERY_PROMPT  = "trees, bushes, vegetation"
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
# SAM helpers
# ---------------------------------------------------------------------------

def _window_to_transform(src, window: Window):
    return rasterio.transform.from_bounds(
        *rasterio.windows.bounds(window, src.transform),
        window.width,
        window.height,
    )


def _normalise_sam(data: np.ndarray):
    data_norm = np.zeros_like(data, dtype=np.uint8)
    for b in range(data.shape[0]):
        band = data[b]
        if np.all(band == 0) or np.all(np.isnan(band)):
            continue
        bmin, bmax = float(np.nanmin(band)), float(np.nanmax(band))
        if bmax > bmin:
            data_norm[b] = ((band - bmin) / (bmax - bmin) * 255).astype(np.uint8)
    n = data_norm.shape[0]
    if n == 1:
        return np.repeat(data_norm, 3, axis=0)
    elif n == 2:
        return np.concatenate([data_norm, data_norm[0:1]], axis=0)
    elif n >= 3:
        return data_norm[:3]
    return None


def run_sam_on_tif(tif_path, sam3, prompt, tile_size, min_size, clear_memory):
    """Run SamGeo3 tiled over *tif_path*. Returns GeoDataFrame of polygons or None."""
    all_geoms = []
    with rasterio.open(tif_path) as src:
        native_crs = src.crs
        windows = []
        for row in range(0, src.height, tile_size):
            for col in range(0, src.width, tile_size):
                w = min(tile_size, src.width  - col)
                h = min(tile_size, src.height - row)
                if w < tile_size // 2 or h < tile_size // 2:
                    continue
                windows.append(Window(col, row, w, h))

        print(f"  SAM: {len(windows)} tile(s)  prompt='{prompt}'")

        for i, window in enumerate(windows):
            data = src.read(window=window)
            if np.all(data == 0) or np.all(np.isnan(data)):
                continue
            rgb = _normalise_sam(data)
            if rgb is None:
                continue
            tile_transform = _window_to_transform(src, window)

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with rasterio.open(
                    tmp_path, "w", driver="GTiff",
                    height=window.height, width=window.width,
                    count=3, dtype=np.uint8,
                    crs=native_crs, transform=tile_transform,
                ) as dst:
                    dst.write(rgb)

                sam3.set_image_batch([str(tmp_path)])
                sam3.generate_masks_batch(prompt, min_size=min_size)

                if not sam3.batch_results:
                    continue
                result = sam3.batch_results[0]
                masks  = result.get("masks")
                if masks is None or len(masks) == 0:
                    continue
                if hasattr(masks, "numpy"):
                    masks = masks.numpy()

                first   = masks[0]
                h_px    = first.shape[-2] if first.ndim == 3 else first.shape[0]
                w_px    = first.shape[-1] if first.ndim == 3 else first.shape[1]
                labeled = np.zeros((h_px, w_px), dtype="uint16")
                for obj_id, m in enumerate(masks, start=1):
                    m2 = m[0] if m.ndim == 3 else m
                    labeled[m2 > 0] = obj_id

                for geom_json, val in rasterio_shapes(
                    labeled, mask=labeled > 0, transform=tile_transform
                ):
                    if int(val) == 0:
                        continue
                    geom = shapely_shape(geom_json)
                    if not geom.is_empty:
                        all_geoms.append({"geometry": geom})

            except Exception as e:
                print(f"    tile {i}: SAM error -> {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

            if clear_memory and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    if not all_geoms:
        return None
    return gpd.GeoDataFrame(all_geoms, crs=native_crs)


# ---------------------------------------------------------------------------
# Intersection coverage
# ---------------------------------------------------------------------------

def compute_sam_coverage(tiles_gdf, sam_gdf, col_name):
    """
    Add *col_name* = percentage [0-100] of each tile covered by SAM polygons.
    Both inputs must be in the same projected CRS (for area in m2).
    """
    tiles_gdf = tiles_gdf.copy()
    if sam_gdf is None or sam_gdf.empty:
        tiles_gdf[col_name] = 0.0
        return tiles_gdf

    if sam_gdf.crs != tiles_gdf.crs:
        sam_gdf = sam_gdf.to_crs(tiles_gdf.crs)

    sam_sindex  = sam_gdf.sindex
    pct_values  = []

    for tile_geom in tiles_gdf.geometry:
        if tile_geom is None or tile_geom.is_empty:
            pct_values.append(0.0)
            continue
        tile_area = tile_geom.area
        if tile_area == 0:
            pct_values.append(0.0)
            continue
        candidate_ids = list(sam_sindex.intersection(tile_geom.bounds))
        if not candidate_ids:
            pct_values.append(0.0)
            continue
        try:
            sam_union    = unary_union(sam_gdf.iloc[candidate_ids].geometry.values)
            intersection = tile_geom.intersection(sam_union)
            pct          = min(100.0, intersection.area / tile_area * 100.0)
        except Exception:
            pct = 0.0
        pct_values.append(round(pct, 2))

    tiles_gdf[col_name] = pct_values
    return tiles_gdf


# ---------------------------------------------------------------------------
# Per-scene orchestration
# ---------------------------------------------------------------------------

def process_scene(
    tif_path, tiles_gpkg, model, sam3, sam_modes, outdir,
    min_size, sam_tile_size, clear_memory, overwrite,
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

    # SAM segmentation + coverage per tile
    for mode in sam_modes:
        if mode == "greenery":
            prompt  = GREENERY_PROMPT
            col     = "sam_greenery_pct"
            sam_dir = outdir.parent / "green"
            layer   = "greenery"
            suffix  = "_green.gpkg"

        sam_out = sam_dir / f"{oam_id}{suffix}"

        if sam_out.exists() and not overwrite:
            print(f"  SAM ({mode}): loading cached {sam_out.name}")
            sam_gdf = gpd.read_file(sam_out)
        else:
            print(f"  SAM ({mode}): running...")
            sam_dir.mkdir(parents=True, exist_ok=True)
            sam_gdf = run_sam_on_tif(
                tif_path, sam3, prompt,
                tile_size=sam_tile_size,
                min_size=min_size,
                clear_memory=clear_memory,
            )
            if sam_gdf is not None:
                sam_gdf["oam_id"] = oam_id
                sam_ea = sam_gdf.to_crs({"proj": "cea"})
                sam_gdf["area_m2"] = sam_ea.geometry.area.round(1)
                sam_gdf["area_ha"] = (sam_gdf["area_m2"] / 10_000).round(3)
                sam_gdf.to_file(sam_out, driver="GPKG", layer=layer)
                print(f"  SAM ({mode}): {len(sam_gdf)} polygons -> {sam_out.name}")
            else:
                print(f"  SAM ({mode}): no polygons found")

        print(f"  Computing {col} per tile...")
        tiles_gdf = compute_sam_coverage(tiles_gdf, sam_gdf, col_name=col)
        covered = (tiles_gdf[col] > 0).sum()
        print(f"  {col}: {covered:,} tiles > 0%  (mean {tiles_gdf[col].mean():.1f}%)")

        # Mark tiles where SAM coverage >= 25% of tile area as green.
        # Adds pred_class_green column to the tile GPKG, which
        # 01_calculate_aoi_metrics.py reads to compute green_pct.
        if col == "sam_greenery_pct":
            tiles_gdf["pred_class_green"] = tiles_gdf[col].apply(
                lambda x: "green" if x >= 25.0 else "background"
            )
            green_n = (tiles_gdf["pred_class_green"] == "green").sum()
            print(f"  pred_class_green: {green_n:,} green tiles (>= 25% coverage)")

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
            "Combined YOLO waste prediction + SAM greenery segmentation coverage. "
            "Adds pred_class, confidence, sam_greenery_pct, pred_class_green "
            "to each 5 m tile and saves to data/predictions/waste/."
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
    parser.add_argument("--sam", choices=["greenery", "none"],
                        default="none",
                        help="SAM mode: 'greenery' runs SamGeo3 and writes pred_class_green "
                             "to each tile GPKG; 'none' skips SAM (default).")
    parser.add_argument("--sam-tile-size", type=int, default=SAM_TILE_SIZE)
    parser.add_argument("--min-size", type=int, default=SAM_MIN_SIZE,
                        help="Min SAM object size in pixels.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--clear-memory", action="store_true",
                        help="Empty GPU cache after each SAM tile.")
    parser.add_argument("--outdir", type=Path,
                        default=Path("data/predictions/waste"),
                        help="Output directory. (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    sam_modes = {
        "greenery": ["greenery"],
        "none":     [],
    }[args.sam]

    # Load YOLO model
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\n"
                 "Train first with 01_train_waste_classification.py.")
    print(f"Loading YOLO model: {args.model}")
    yolo_model = YOLO(str(args.model))
    print(f"  CUDA: {torch.cuda.is_available()}")

    # Load SAM model (once)
    sam3 = None
    if sam_modes:
        try:
            from samgeo import SamGeo3
        except ImportError:
            sys.exit("samgeo not installed: pip install samgeo")
        print("Initialising SamGeo3 (downloads from HuggingFace on first run)...")
        sam3 = SamGeo3(backend="meta", device=args.device,
                       checkpoint_path=None, load_from_HF=True)
        print("  SamGeo3 ready")

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
    print(f"WASTE + SAM PREDICTION  ({len(tif_paths)} scene(s))")
    print(f"  YOLO model : {args.model}")
    print(f"  SAM modes  : {sam_modes or ['none']}")
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
                sam3          = sam3,
                sam_modes     = sam_modes,
                outdir        = args.outdir,
                min_size      = args.min_size,
                sam_tile_size = args.sam_tile_size,
                clear_memory  = args.clear_memory,
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

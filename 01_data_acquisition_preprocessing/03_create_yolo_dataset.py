#!/usr/bin/env python3
"""
03_create_yolo_dataset.py
==========================
Build a YOLO image-classification dataset from manually labeled tile GeoPackages
produced by ``02_download_and_tile.py``.

Workflow
--------
1. ``02_download_and_tile.py`` creates ``data/tiles/<oam_id>_tiles.gpkg`` for
   each downloaded scene — a 5 × 5 m regular grid over the raster extent.
2. Open each ``_tiles.gpkg`` in QGIS and set the ``label`` column:
       1 = waste,  0 = background
   (or set the ``class`` column to ``"waste"`` / ``"background"``)
   Aim for ~100 waste + ~100 background tiles per scene.
3. Run this script to build the YOLO dataset.

What this script does
---------------------
  1. Discovers all ``*_tiles.gpkg`` files in ``--tiles-dir``.
  2. For each GPKG, samples up to ``--max-per-class`` labeled tiles per class
     (default 100 waste + 100 background) to keep the per-AOI balance.
  3. Locates the corresponding GeoTIFF in ``--imagery-dir`` by stem matching.
  4. Crops each tile from the raster with rasterio, resizes to
     ``--tile-size × --tile-size`` pixels (default 128 px).
  5. Splits the sampled tiles **per AOI** into train / val / test
     (default 70 / 15 / 15) and saves PNGs under:

       <outdir>/
         train/waste/
         train/background/
         val/waste/
         val/background/
         test/waste/
         test/background/

  6. Prints a class-balance summary.

Split (per AOI, per class — e.g. 100 waste per scene)
------------------------------------------------------
  train : 70 tiles
  val   : 15 tiles
  test  : 15 tiles
The split is applied per AOI per class with a fixed random seed for
full reproducibility.

GPKG label conventions
-----------------------
Label column (checked in order):
  ``label``  — integer: 1 = waste, 0 = background
  ``class``  — string:  'waste' | 'background'

Raster matching (checked in order for each ``<oam_id>``):
  1. Exact match:  ``<imagery-dir>/<oam_id>.tif``
  2. Merged:       ``<imagery-dir>/<oam_id>_merged.tif``
  3. Any TIF whose stem starts with ``<oam_id>``
  4. Spatial fallback: first TIF whose bounding box contains the GPKG centroid

Usage
-----
    python 01_data_acquisition_preprocessing/03_create_yolo_dataset.py

    # Adjust split, sampling cap, or tile size:
    python 01_data_acquisition_preprocessing/03_create_yolo_dataset.py \\
        --tiles-dir  data/tiles/ \\
        --imagery-dir data/imagery/ \\
        --outdir     data/dataset/ \\
        --tile-size  128 \\
        --max-per-class 100 \\
        --train 0.70 --val 0.15 --test 0.15 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError:
    print("rasterio is required.  pip install rasterio", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow is required.  pip install Pillow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "background", 1: "waste"}
CLASS_IDS   = {"background": 0, "waste": 1}


def read_labels(gpkg_path: Path) -> pd.DataFrame | None:
    """
    Load a labeled GeoPackage and return a DataFrame with columns
    ['label', 'geometry'].  Returns None if no usable label column is found.
    """
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return None

    if "label" in gdf.columns:
        gdf["label"] = gdf["label"].astype(int)
        valid = gdf[gdf["label"].isin([0, 1])].copy()

    elif "class" in gdf.columns:
        gdf["label"] = gdf["class"].str.strip().str.lower().map(CLASS_IDS)
        valid = gdf[gdf["label"].notna()].copy()
        valid["label"] = valid["label"].astype(int)

    else:
        return None

    if valid.empty:
        return None

    return valid[["label", "geometry"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raster matching
# ---------------------------------------------------------------------------

def find_raster(gpkg_path: Path, imagery_dir: Path) -> Path | None:
    """
    Locate the GeoTIFF that corresponds to ``gpkg_path`` using the
    name-matching + spatial-fallback strategy described in the module docstring.
    """
    stem = gpkg_path.stem
    # Strip common label suffixes so 'accra_labels' → 'accra'
    for suffix in ("_labels", "_label", "_waste_yolo", "_annotated",
                   "_tiles", "_tile"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidates = sorted(imagery_dir.glob("*.tif"))

    # 1. Exact
    exact = imagery_dir / f"{stem}.tif"
    if exact.exists():
        return exact

    # 2. Merged variant
    merged = imagery_dir / f"{stem}_merged.tif"
    if merged.exists():
        return merged

    # 3. Any TIF whose stem starts with our stem
    prefix_matches = [p for p in candidates if p.stem.startswith(stem)]
    if prefix_matches:
        return prefix_matches[0]

    # 4. Spatial fallback: load GPKG centroid, check raster bounds
    try:
        gdf_tmp = gpd.read_file(gpkg_path).to_crs("EPSG:4326")
        cx, cy = (
            gdf_tmp.geometry.union_all().centroid.x,
            gdf_tmp.geometry.union_all().centroid.y,
        )
        for tif in candidates:
            with rasterio.open(tif) as src:
                bounds = src.bounds
                if (bounds.left <= cx <= bounds.right and
                        bounds.bottom <= cy <= bounds.top):
                    return tif
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Tile cropping
# ---------------------------------------------------------------------------

def crop_tile(
    src: rasterio.DatasetReader,
    geom,
    tile_size: int,
) -> Image.Image | None:
    """
    Crop a tile from an open rasterio dataset using the tile geometry bounds.
    Returns a PIL Image (RGB) resized to tile_size × tile_size, or None on error.
    """
    try:
        minx, miny, maxx, maxy = geom.bounds
        window = window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        if window.width < 1 or window.height < 1:
            return None

        data = src.read(window=window)

        # Use first 3 bands as RGB; pad with zeros if fewer than 3
        if data.shape[0] >= 3:
            rgb = data[:3]
        elif data.shape[0] == 1:
            rgb = np.repeat(data, 3, axis=0)
        else:
            rgb = np.concatenate([data, np.zeros((3 - data.shape[0], *data.shape[1:]),
                                                  dtype=data.dtype)], axis=0)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.moveaxis(rgb, 0, -1))
        img = img.resize((tile_size, tile_size), Image.LANCZOS)
        return img

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def sample_labels(
    labels_gdf: gpd.GeoDataFrame,
    max_per_class: int,
    rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """
    Randomly sample up to *max_per_class* tiles from each of the two classes
    (waste=1, background=0) so that each AOI contributes at most
    2 x max_per_class labeled tiles to the dataset.
    Preserves CRS of the input GeoDataFrame.
    """
    parts = []
    for cls_id in (0, 1):
        subset = labels_gdf[labels_gdf["label"] == cls_id]
        n = min(len(subset), max_per_class)
        if n > 0:
            sampled = subset.sample(n=n, random_state=int(rng.integers(2**31)))
            parts.append(sampled)
    if not parts:
        return labels_gdf.iloc[0:0]   # empty
    result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                              crs=labels_gdf.crs)
    return result


def build_dataset(
    tiles_dir: Path,
    imagery_dir: Path,
    outdir: Path,
    tile_size: int,
    train_frac: float,
    val_frac: float,
    seed: int,
    max_per_class: int,
) -> None:
    # Create output directories
    splits = ("train", "val", "test")
    for split in splits:
        for cls in CLASS_NAMES.values():
            (outdir / split / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Accept both plain *.gpkg and *_tiles.gpkg
    gpkg_files = sorted(tiles_dir.glob("*_tiles.gpkg"))
    if not gpkg_files:
        # Fall back to any gpkg in case naming differs
        gpkg_files = sorted(tiles_dir.glob("*.gpkg"))
    if not gpkg_files:
        print(f"⚠  No .gpkg files found in {tiles_dir}", file=sys.stderr)
        print("  Make sure you have labeled tiles in QGIS and saved them.",
              file=sys.stderr)
        return

    total_counts: dict[str, dict[str, int]] = {
        s: {"waste": 0, "background": 0} for s in splits
    }
    skipped_gpkg = 0

    for gpkg_path in gpkg_files:
        print(f"\n{'─'*60}")
        print(f"Scene: {gpkg_path.name}")

        # --- Read labels ---
        labels_df = read_labels(gpkg_path)
        if labels_df is None:
            print(f"  ⚠  No usable label/class column — skipping.")
            print(f"     (Open in QGIS, add 'label' column: 1=waste, 0=background)")
            skipped_gpkg += 1
            continue

        n_waste = int((labels_df["label"] == 1).sum())
        n_bg    = int((labels_df["label"] == 0).sum())
        print(f"  Labeled : {len(labels_df)} tiles total  "
              f"({n_waste} waste, {n_bg} background)")

        if n_waste == 0 or n_bg == 0:
            print(f"  ⚠  Need both waste and background labels — skipping.")
            skipped_gpkg += 1
            continue

        # --- Sample 100 waste + 100 background per AOI ---
        labels_df = sample_labels(labels_df, max_per_class, rng)
        s_waste = int((labels_df["label"] == 1).sum())
        s_bg    = int((labels_df["label"] == 0).sum())
        print(f"  Sampled : {s_waste} waste + {s_bg} background "
              f"(cap {max_per_class} per class)")

        # --- Find raster ---
        raster_path = find_raster(gpkg_path, imagery_dir)
        if raster_path is None:
            print(f"  ⚠  No matching raster found in {imagery_dir} — skipping.")
            skipped_gpkg += 1
            continue

        print(f"  Raster  : {raster_path.name}")

        # --- Reproject labels to raster CRS ---
        # read_labels() returns a GeoDataFrame with CRS from the GPKG
        # (tile grids are in a projected UTM CRS; reproject to raster CRS)
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        labels_gdf = labels_df   # already a GeoDataFrame from read_labels()
        if labels_gdf.crs is None:
            labels_gdf = labels_gdf.set_crs("EPSG:4326")
        if labels_gdf.crs != raster_crs:
            labels_gdf = labels_gdf.to_crs(raster_crs)

        # --- Assign train/val/test split per class (per AOI) ---
        for cls_id, cls_name in CLASS_NAMES.items():
            subset = labels_gdf[labels_gdf["label"] == cls_id].reset_index(drop=True)
            if subset.empty:
                continue

            n = len(subset)
            n_train = max(1, int(round(n * train_frac)))
            n_val   = max(1, int(round(n * val_frac)))
            n_test  = max(0, n - n_train - n_val)

            idx = rng.permutation(n)
            split_idx = {
                "train": idx[:n_train],
                "val":   idx[n_train: n_train + n_val],
                "test":  idx[n_train + n_val:],
            }

            stem   = gpkg_path.stem.replace(" ", "_")
            n_crop = 0
            n_fail = 0

            with rasterio.open(raster_path) as src:
                for split_name, idxs in split_idx.items():
                    out_split_dir = outdir / split_name / cls_name
                    for i, tile_idx in enumerate(tqdm(
                        idxs,
                        desc=f"  {split_name}/{cls_name}",
                        leave=False,
                    )):
                        geom = subset.geometry.iloc[tile_idx]
                        img  = crop_tile(src, geom, tile_size)
                        if img is None:
                            n_fail += 1
                            continue
                        fname = f"{stem}_{cls_name}_{tile_idx:05d}.png"
                        img.save(out_split_dir / fname)
                        n_crop += 1
                        total_counts[split_name][cls_name] += 1

            print(f"  {cls_name:10s}: {n_crop} tiles saved "
                  f"(train {split_idx['train'].shape[0]}, "
                  f"val {split_idx['val'].shape[0]}, "
                  f"test {split_idx['test'].shape[0]}), "
                  f"{n_fail} failed")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Dataset summary")
    print(f"{'='*60}")
    print(f"{'Split':<8}  {'waste':>8}  {'background':>12}  {'total':>8}")
    print(f"{'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}")
    for split in splits:
        w = total_counts[split]["waste"]
        b = total_counts[split]["background"]
        print(f"{split:<8}  {w:>8}  {b:>12}  {w+b:>8}")
    grand_w = sum(total_counts[s]["waste"]       for s in splits)
    grand_b = sum(total_counts[s]["background"]  for s in splits)
    print(f"{'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}")
    print(f"{'TOTAL':<8}  {grand_w:>8}  {grand_b:>12}  {grand_w+grand_b:>8}")

    if skipped_gpkg:
        print(f"\n⚠  {skipped_gpkg} GeoPackage(s) skipped (see warnings above).")

    print(f"\n✓ Dataset written to: {outdir}")
    print(f"\nNext step:")
    print(f"  python 02_model_training/01_train_waste_classification.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build YOLO classification dataset from labeled tile GeoPackages "
            "(*_tiles.gpkg, produced by 02_download_and_tile.py and labeled in QGIS)."
        )
    )
    parser.add_argument(
        "--tiles-dir", default="data/tiles",
        help="Directory containing labeled *_tiles.gpkg files "
             "(default: data/tiles)"
    )
    parser.add_argument(
        "--imagery-dir", default="data/imagery",
        help="Directory containing GeoTIFFs (default: data/imagery)"
    )
    parser.add_argument(
        "--outdir", default="data/dataset",
        help="Output directory for YOLO PNG tiles (default: data/dataset)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=128,
        help="Output tile size in pixels (default: 128)"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=100,
        help="Max tiles sampled per class per AOI (default: 100)"
    )
    parser.add_argument(
        "--train", type=float, default=0.70, dest="train_frac",
        help="Train fraction per AOI per class (default: 0.70 → 70 tiles)"
    )
    parser.add_argument(
        "--val", type=float, default=0.15, dest="val_frac",
        help="Validation fraction per AOI per class (default: 0.15 → 15 tiles)"
    )
    parser.add_argument(
        "--test", type=float, default=0.15, dest="test_frac",
        help="Test fraction per AOI per class (default: 0.15 → 15 tiles)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible split (default: 0)"
    )
    args = parser.parse_args()

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 0.01:
        parser.error(f"--train + --val + --test must sum to 1.0 (got {frac_sum:.3f})")

    tiles_dir   = Path(args.tiles_dir)
    imagery_dir = Path(args.imagery_dir)
    outdir      = Path(args.outdir)

    if not tiles_dir.exists():
        raise FileNotFoundError(
            f"Tiles directory not found: {tiles_dir}\n"
            "Run 02_download_and_tile.py first, then label the "
            "*_tiles.gpkg files in QGIS."
        )
    if not imagery_dir.exists():
        raise FileNotFoundError(
            f"Imagery directory not found: {imagery_dir}\n"
            "Run 02_download_and_tile.py first."
        )

    build_dataset(
        tiles_dir     = tiles_dir,
        imagery_dir   = imagery_dir,
        outdir        = outdir,
        tile_size     = args.tile_size,
        train_frac    = args.train_frac,
        val_frac      = args.val_frac,
        seed          = args.seed,
        max_per_class = args.max_per_class,
    )


if __name__ == "__main__":
    main()

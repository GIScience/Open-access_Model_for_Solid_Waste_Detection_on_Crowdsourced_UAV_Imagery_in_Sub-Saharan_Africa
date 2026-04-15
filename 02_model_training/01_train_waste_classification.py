#!/usr/bin/env python3
"""
02_model_training/01_train_waste_classification.py
===================================================
Fine-tune YOLO11x-cls on the labeled solid waste / background tile dataset.

Usage
-----
    # run from repo root:
    python 02_model_training/01_train_waste_classification.py
    python 02_model_training/01_train_waste_classification.py --name my_run

Outputs
-------
  data/models/<name>/                      YOLO weights + training artefacts
  data/results/test_metrics_overall.json   overall TP/FP/FN/TN + metrics
  data/results/test_file_metrics_29AOI.csv per-AOI metrics

Requirements: ultralytics, rasterio, geopandas, Pillow, numpy, tqdm, torch (GPU)
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# Directories (relative to repo root; resolved to absolute where needed)
DATASET_DIR  = Path("data/dataset")    # output of 03_create_yolo_dataset.py
MODELS_DIR   = Path("data/models")     # YOLO training output: data/models/<name>/
RESULTS_DIR  = Path("data/results")    # per-AOI CSV + overall JSON

# YOLO hyper-parameters
EPOCHS     = 150
IMG_SIZE   = 128
BATCH_SIZE = 64
MODEL_NAME = "yolo11x-cls.pt"


# =============================================================================
# Pipeline steps
# =============================================================================

def create_dataset_yaml() -> None:
    print("\n" + "=" * 70)
    print("STEP 3 · DATASET YAML")
    print("=" * 70)
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        "train: train\nval: val\ntest: test\n\n"
        "names:\n  0: background\n  1: waste\nnc: 2\n"
    )
    print(f"✓ {yaml_path}")


def train_model(run_name: str):
    # Resolve to absolute path so YOLO never nests inside its own default
    # runs/classify/ prefix regardless of cwd.
    models_abs = MODELS_DIR.resolve()
    print("\n" + "=" * 70)
    print("STEP 4 · TRAIN YOLO11x-cls")
    print(f"  model={MODEL_NAME}  epochs={EPOCHS}  imgsz={IMG_SIZE}  batch={BATCH_SIZE}")
    print(f"  output → {models_abs / run_name}")
    print("=" * 70)

    models_abs.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(DATASET_DIR.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(models_abs),
        name=run_name,
        lr0=0.0005,
        lrf=1e-05,
        cos_lr=True,
        patience=15,
        auto_augment="randaugment",
        fliplr=0.7,
        flipud=0.7,
        degrees=15.0,   # ±15° random rotation (paper §2)
        translate=0.2,
        scale=0.5,
        hsv_h=0.05,
        hsv_s=0.8,
        hsv_v=0.5,
        mosaic=1.0,
        mixup=0.2,
        erasing=0.3,
        dropout=0.0,
        optimizer="AdamW",
        weight_decay=0.001,
        momentum=0.937,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        close_mosaic=20,
        amp=True,
        val=True,
        plots=True,
        save_period=10,
        workers=8,
        seed=0,
        deterministic=True,
        device=0,
    )
    print(f"\n✓ Training complete — weights in: {Path(results.save_dir) / 'weights'}")
    return results


def evaluate_test_split(save_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("STEP 5 · TEST-SET EVALUATION (overall)")
    print("=" * 70)
    best_weights = save_dir / "weights" / "best.pt"
    model = YOLO(best_weights)
    metrics = model.val(
        data=str(DATASET_DIR), split="test",
        imgsz=IMG_SIZE, batch=BATCH_SIZE, device=0, verbose=False,
    )
    cm = metrics.confusion_matrix.matrix
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    npv         = tn / (tn + fn + 1e-9)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  NPV         : {npv:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(
        split="test",
        TP=tp, FP=fp, FN=fn, TN=tn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        accuracy=round(accuracy, 6),
        specificity=round(specificity, 6),
        npv=round(npv, 6),
        top1_acc=round(float(metrics.top1), 6) if hasattr(metrics, "top1") else None,
        top5_acc=round(float(metrics.top5), 6) if hasattr(metrics, "top5") else None,
        weights=str(best_weights),
    )
    json_path = RESULTS_DIR / "test_metrics_overall.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved → {json_path}")
    return best_weights


def aoi_stem_from_filename(stem: str) -> str:
    """Extract AOI name from a tile filename.
    e.g. 'kenya_r10_c20'  -> 'kenya'
         'Dar_Msimbazi_r3_c7' -> 'Dar_Msimbazi'
    """
    m = re.match(r'^(.+?)_r\d+_c\d+', stem)
    if m:
        return m.group(1)
    # fallback: everything before last two underscore-separated tokens
    parts = stem.split('_')
    return '_'.join(parts[:-2]) if len(parts) > 2 else stem


def evaluate_per_aoi(best_weights: Path) -> None:
    print("\n" + "=" * 70)
    print("STEP 6 · PER-AOI TEST METRICS  →  test_file_metrics_29AOI.csv")
    print("=" * 70)

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    # Collect test images with true labels
    records = []
    for cls_name, true_label in (("waste", 1), ("background", 0)):
        cls_dir = DATASET_DIR / "test" / cls_name
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                records.append({"path": f, "true": true_label,
                                 "aoi": aoi_stem_from_filename(f.stem)})

    if not records:
        print("  No test images found — skipping per-AOI evaluation.")
        return

    print(f"  {len(records)} test images across "
          f"{len(set(r['aoi'] for r in records))} AOIs")

    model = YOLO(best_weights)

    # Predict in batches of 512 to avoid OOM
    all_paths = [r["path"] for r in records]
    preds = []
    BS = 512
    for i in tqdm(range(0, len(all_paths), BS), desc="  predicting", unit="batch"):
        batch = [str(p) for p in all_paths[i:i + BS]]
        res = model.predict(batch, imgsz=IMG_SIZE, device=0,
                            verbose=False, stream=False)
        for r in res:
            preds.append(int(r.probs.top1))   # predicted class index

    # class index mapping: YOLO sorts classes alphabetically
    # background=0, waste=1  (alphabetical order matches label)
    df = pd.DataFrame(records)
    df["pred"] = preds

    # Per-AOI aggregation
    rows = []
    for aoi, grp in df.groupby("aoi"):
        tp = int(((grp["true"] == 1) & (grp["pred"] == 1)).sum())
        tn = int(((grp["true"] == 0) & (grp["pred"] == 0)).sum())
        fp = int(((grp["true"] == 0) & (grp["pred"] == 1)).sum())
        fn = int(((grp["true"] == 1) & (grp["pred"] == 0)).sum())
        n_waste = int((grp["true"] == 1).sum())
        n_bg    = int((grp["true"] == 0).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        rows.append(dict(
            aoi=aoi, n_waste=n_waste, n_background=n_bg,
            TP=tp, FP=fp, FN=fn, TN=tn,
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(f1, 4), accuracy=round(acc, 4),
        ))

    out_df = pd.DataFrame(rows).sort_values("aoi").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "test_file_metrics_29AOI.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(out_df.to_string(index=False))


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLO waste classifier and evaluate per AOI."
    )
    p.add_argument(
        "--name", default="train",
        help="Run name; weights saved to data/models/<name>/  (default: train)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.name

    print("\n" + "=" * 70)
    print("WASTE CLASSIFICATION — TRAINING PIPELINE")
    print(f"  run name : {run_name}")
    print("=" * 70)

    # Check the dataset exists (built by 03_create_yolo_dataset.py)
    required_dirs = [
        DATASET_DIR / split / cls
        for split in ("train", "val", "test")
        for cls in ("waste", "background")
    ]
    missing = [d for d in required_dirs if not d.exists()]
    if missing:
        print(
            "\u2717 Dataset not found. Build it first:\n"
            "  python 01_data_acquisition_preprocessing/03_create_yolo_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    n_train_waste = len(list((DATASET_DIR / "train" / "waste").glob("*.png")))
    n_train_bg    = len(list((DATASET_DIR / "train" / "background").glob("*.png")))
    print(f"\u2713 Dataset found: {DATASET_DIR}")
    print(f"  train/waste={n_train_waste}  train/background={n_train_bg}")

    create_dataset_yaml()
    results    = train_model(run_name)
    save_dir   = Path(results.save_dir)
    best_w     = evaluate_test_split(save_dir)
    evaluate_per_aoi(best_w)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print(f"  weights  : {save_dir / 'weights' / 'best.pt'}")
    print(f"  overall  : {RESULTS_DIR / 'test_metrics_overall.json'}")
    print(f"  per-AOI  : {RESULTS_DIR / 'test_file_metrics_29AOI.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

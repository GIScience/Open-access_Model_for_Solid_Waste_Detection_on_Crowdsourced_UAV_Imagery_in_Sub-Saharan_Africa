#!/usr/bin/env python3
"""
02_model_training/03_evaluate_robustness.py
===========================================
Comprehensive robustness evaluation of the trained waste-classification model.

Outputs (all saved to data/results/):
  - roc_curve.png                  ROC curve with AUC on the held-out test set
  - metric_vs_sample_size.png      F1 / AUC vs test-sample size (bootstrap stability)
  - bootstrap_ci.png               Bootstrap confidence intervals for all core metrics
  - per_aoi_metrics.png            Heatmap + violin plots of per-AOI performance
  - robustness_report.md           summary

Usage
-----
  python 02_model_training/03_evaluate_robustness.py
  python 02_model_training/03_evaluate_robustness.py --weights data/models/run2/weights/best.pt
  python 02_model_training/03_evaluate_robustness.py --bootstrap-n 2000

Dependencies: ultralytics, matplotlib, seaborn, scikit-learn, numpy, pandas, tqdm
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
DATASET_DIR = ROOT / "data" / "dataset"
RESULTS_DIR = ROOT / "data" / "results"
MODELS_DIR  = ROOT / "data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["background", "waste"]   # YOLO alphabetical order by default
WASTE_IDX = 1                        # index of 'waste' class in softmax output

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"     : 150,
    "font.family"    : "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {"waste": "#E63946", "background": "#457B9D"}

# ───────────────────────────────────────────────────────────────────────────
# 1. Data collection
# ───────────────────────────────────────────────────────────────────────────

def collect_test_images(split: str = "test", test_dir: Path | None = None) -> dict[str, list[Path]]:
    """Return {class_name: [image_paths]} for the given split.

    If *test_dir* is provided it is used directly (ignores DATASET_DIR / split).
    Expected layout: <test_dir>/{waste,background}/*.png
    """
    split_dir = Path(test_dir) if test_dir else DATASET_DIR / split
    result: dict[str, list[Path]] = {}
    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            result[cls_dir.name] = sorted(imgs)
    return result


def run_inference(model: YOLO, images_by_class: dict[str, list[Path]]) -> pd.DataFrame:
    """
    Run model.predict on all test images.
    Returns a DataFrame with columns:
        path, true_class, true_label, waste_conf, pred_label, pred_class
    """
    records = []
    for cls_name, paths in images_by_class.items():
        true_label = CLASSES.index(cls_name) if cls_name in CLASSES else 0
        for img_path in tqdm(paths, desc=f"  Predicting [{cls_name}]", leave=False):
            results = model.predict(str(img_path), verbose=False)
            probs = results[0].probs
            # probs.data is a tensor of shape [n_classes]
            conf_array = probs.data.cpu().numpy()
            # conf_array is sorted alphabetically: [background, waste]
            waste_conf = float(conf_array[WASTE_IDX])
            pred_label = int(np.argmax(conf_array))
            records.append(dict(
                path       = str(img_path),
                true_class = cls_name,
                true_label = true_label,
                waste_conf = waste_conf,
                pred_label = pred_label,
                pred_class = CLASSES[pred_label],
            ))
    return pd.DataFrame(records)


# ───────────────────────────────────────────────────────────────────────────
# 2. ROC curve
# ───────────────────────────────────────────────────────────────────────────

def plot_roc(df: pd.DataFrame, out_dir: Path) -> float:
    """Plot ROC curve and return AUC."""
    fpr, tpr, thresholds = roc_curve(df["true_label"], df["waste_conf"])
    roc_auc = auc(fpr, tpr)

    # Find Youden's J optimal threshold
    j_scores = tpr - fpr
    opt_idx  = np.argmax(j_scores)
    opt_thr  = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E63946", lw=2.5,
            label=f"YOLO11x-cls  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC = 0.50)")
    ax.scatter(fpr[opt_idx], tpr[opt_idx],
               s=100, zorder=5, color="#1D3557",
               label=f"Optimal threshold = {opt_thr:.3f}\n"
                     f"(TPR={tpr[opt_idx]:.3f}, FPR={fpr[opt_idx]:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Waste Classification (Test Set)", fontsize=13, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out = out_dir / "roc_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ ROC curve saved  (AUC = {roc_auc:.4f}) → {out.name}")
    return roc_auc


# ───────────────────────────────────────────────────────────────────────────
# 3. Metric vs sample size  (bootstrap stability curve)
# ───────────────────────────────────────────────────────────────────────────
# Per-class sample sizes for bootstrap stability curve.
# Kept to ≤ min(n_waste, n_background) so both classes can always contribute equally.
# The full-dataset point is appended dynamically in plot_metric_vs_sample_size().
SAMPLE_SIZES_PER_CLASS = [10, 20, 40, 70, 100, 150, 200, 300, 400]

def metric_at_sample(df: pd.DataFrame, n_per_class: int, n_reps: int = 200) -> dict:
    """
    Stratified bootstrap: draw exactly `n_per_class` waste tiles AND
    `n_per_class` background tiles per repetition (without replacement).
    This keeps class balance constant across all sample sizes so that
    metric changes reflect true stability, not random class imbalance.
    """
    f1s, aucs, accs = [], [], []
    waste_idx = df.index[df["true_label"] == 1].tolist()
    bg_idx    = df.index[df["true_label"] == 0].tolist()
    # guard: can't sample more than available
    n_per_class = min(n_per_class, len(waste_idx), len(bg_idx))
    for _ in range(n_reps):
        chosen = (
            random.sample(waste_idx, n_per_class) +
            random.sample(bg_idx,   n_per_class)
        )
        sub = df.loc[chosen]
        yt = sub["true_label"].values
        yp = sub["pred_label"].values
        yc = sub["waste_conf"].values
        f1s.append(f1_score(yt, yp, zero_division=0))
        aucs.append(roc_auc_score(yt, yc))
        accs.append(accuracy_score(yt, yp))
    total_n = n_per_class * 2
    return dict(
        n_per_class=n_per_class,
        n_total=total_n,
        f1_mean=np.mean(f1s),   f1_std=np.std(f1s),
        auc_mean=np.mean(aucs), auc_std=np.std(aucs),
        acc_mean=np.mean(accs), acc_std=np.std(accs),
    )


def plot_metric_vs_sample_size(df: pd.DataFrame, out_dir: Path) -> None:
    # Maximum balanced per-class size = min(n_waste, n_background)
    n_waste = int((df["true_label"] == 1).sum())
    n_bg    = int((df["true_label"] == 0).sum())
    max_per_class = min(n_waste, n_bg)

    sizes = sorted(set(
        [s for s in SAMPLE_SIZES_PER_CLASS if s < max_per_class] + [max_per_class]
    ))
    print(f"  Stratified subsampling: {sizes} samples/class  "
          f"(max balanced = {max_per_class}/class, waste={n_waste}, bg={n_bg})")

    rows = [metric_at_sample(df, n) for n in tqdm(sizes, leave=False)]
    df_s = pd.DataFrame(rows)

    # Compute shared y range across all three metrics
    configs = [
        ("f1_mean",  "f1_std",  "F1 score",  "#E63946"),
        ("auc_mean", "auc_std", "ROC AUC",   "#1D3557"),
        ("acc_mean", "acc_std", "Accuracy",  "#457B9D"),
    ]
    all_y   = np.concatenate([df_s[c].values for c, *_ in configs])
    all_ye  = np.concatenate([df_s[e].values for _, e, *_ in configs])
    y_min   = max(0.0, (all_y - all_ye).min() - 0.02)
    y_max   = 1.00

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    configs = [
        ("f1_mean",  "f1_std",  "F1 score",  "#E63946"),
        ("auc_mean", "auc_std", "ROC AUC",   "#1D3557"),
        ("acc_mean", "acc_std", "Accuracy",  "#457B9D"),
    ]
    for idx, (ax, (mean_col, std_col, label, color)) in enumerate(zip(axes, configs)):
        y  = df_s[mean_col].values
        ye = df_s[std_col].values
        legend_str = f"{label} = {y[-1]:.4f}"
        ax.plot(df_s["n_total"], y, "o-", color=color, lw=2.2, ms=6)
        ax.fill_between(df_s["n_total"], y - ye, y + ye, color=color, alpha=0.18)
        ax.axhline(y[-1], ls="--", lw=1.2, color=color, alpha=0.5, label=legend_str)
        ax.set_ylim(y_min, y_max)

        ax.legend(fontsize=8, loc="upper right")

        # y-axis label on leftmost panel only; hide numbers (not ticks) on others
        if idx == 0:
            ax.set_ylabel("Score", fontsize=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", labelleft=False)

        # Top twin axis (per-class count) on all panels — label only on middle panel
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(df_s["n_total"].values)
        ax2.set_xticklabels(
            [str(r) for r in df_s["n_per_class"].values], fontsize=7, rotation=45
        )
        if idx == 1:
            ax2.set_xlabel("Per-class count", fontsize=8)

    # Single centred x-label spanning all panels
    fig.text(0.5, -0.01,
             "Amount of test sample\n(solid waste + background)",
             ha="center", va="top", fontsize=10)

    fig.tight_layout()
    out = out_dir / "metric_vs_sample_size.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Sample-size curve saved → {out.name}")


# ───────────────────────────────────────────────────────────────────────────
# 4. Bootstrap confidence intervals
# ───────────────────────────────────────────────────────────────────────────

def bootstrap_ci(df: pd.DataFrame, n_boot: int = 2000, ci: float = 0.95) -> pd.DataFrame:
    """Full bootstrap (with replacement) CI for precision, recall, F1, accuracy, AUC."""
    metrics_names = ["precision", "recall", "f1", "accuracy", "auc"]
    boot_vals = {m: [] for m in metrics_names}
    yt_all = df["true_label"].values
    yp_all = df["pred_label"].values
    yc_all = df["waste_conf"].values
    n = len(df)

    for _ in tqdm(range(n_boot), desc="  Bootstrapping", leave=False):
        idx = np.random.choice(n, size=n, replace=True)
        yt = yt_all[idx]; yp = yp_all[idx]; yc = yc_all[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_vals["precision"].append(precision_score(yt, yp, zero_division=0))
        boot_vals["recall"].append(recall_score(yt, yp, zero_division=0))
        boot_vals["f1"].append(f1_score(yt, yp, zero_division=0))
        boot_vals["accuracy"].append(accuracy_score(yt, yp))
        boot_vals["auc"].append(roc_auc_score(yt, yc))

    alpha = (1 - ci) / 2
    obs = {
        "precision": precision_score(yt_all, yp_all, zero_division=0),
        "recall"   : recall_score(yt_all, yp_all, zero_division=0),
        "f1"       : f1_score(yt_all, yp_all, zero_division=0),
        "accuracy" : accuracy_score(yt_all, yp_all),
        "auc"      : roc_auc_score(yt_all, yc_all),
    }
    rows = []
    for m in metrics_names:
        v = np.array(boot_vals[m])
        rows.append(dict(
            metric=m,
            observed=float(obs[m]),
            mean=float(v.mean()),
            std=float(v.std()),
            ci_lo=float(np.percentile(v, alpha * 100)),
            ci_hi=float(np.percentile(v, (1 - alpha) * 100)),
        ))
    return pd.DataFrame(rows)


def plot_bootstrap_ci(ci_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
    y_pos = range(len(ci_df))

    for i, (_, row) in enumerate(ci_df.iterrows()):
        ax.barh(i, row["observed"], left=0, height=0.55, color=colors[i % len(colors)],
                alpha=0.75, label=row["metric"].capitalize())
        ax.errorbar(row["observed"], i,
                    xerr=[[row["observed"] - row["ci_lo"]],
                          [row["ci_hi"] - row["observed"]]],
                    fmt="none", color="black", capsize=6, lw=2, capthick=2)
        ax.text(row["ci_hi"] + 0.003, i,
                f"{row['observed']:.4f}  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]",
                va="center", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([r["metric"].capitalize() for _, r in ci_df.iterrows()], fontsize=11)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.18)
    ax.set_title("Bootstrap 95 % Confidence Intervals — Test Set", fontsize=13, pad=10)
    ax.axvline(1.0, ls="--", lw=1, color="grey", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "bootstrap_ci.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Bootstrap CI plot saved → {out.name}")


# ───────────────────────────────────────────────────────────────────────────
# 5. Per-AOI metrics
# ───────────────────────────────────────────────────────────────────────────

def build_per_aoi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add aoi column by stripping tile index suffix from filename.

    Filenames are like  5be9bb18080ac000051474fe_6127.png
    AOI id = everything before the LAST underscore.
    Falls back to the full stem if no underscore found.
    Later this can be enriched via data/AOI.gpkg using the uuid / oam_id column.
    """
    def _aoi(p: str) -> str:
        stem = Path(p).stem          # 5be9bb18080ac000051474fe_6127
        if "_" in stem:
            return stem.rsplit("_", 1)[0]   # 5be9bb18080ac000051474fe
        return stem

    df = df.copy()
    df["aoi"] = df["path"].apply(_aoi)

    records = []
    for aoi, sub in df.groupby("aoi"):
        yt = sub["true_label"].values
        yp = sub["pred_label"].values
        waste_count = int((yt == 1).sum())
        bg_count    = int((yt == 0).sum())
        if len(yt) < 2 or len(np.unique(yt)) < 2:
            # AOIs with only one class — record accuracy only
            records.append(dict(aoi=aoi, n=len(yt),
                                n_waste=waste_count, n_bg=bg_count,
                                precision=float("nan"), recall=float("nan"),
                                f1=float("nan"),
                                accuracy=accuracy_score(yt, yp)))
            continue
        records.append(dict(
            aoi=aoi, n=len(yt),
            n_waste=waste_count, n_bg=bg_count,
            precision=precision_score(yt, yp, zero_division=0),
            recall=recall_score(yt, yp, zero_division=0),
            f1=f1_score(yt, yp, zero_division=0),
            accuracy=accuracy_score(yt, yp),
        ))
    return pd.DataFrame(records).sort_values("f1", ascending=False, na_position="last")


def plot_per_aoi(aoi_df: pd.DataFrame, out_dir: Path) -> None:
    valid = aoi_df.dropna(subset=["f1"]).copy()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # A) Horizontal bar chart — F1 per AOI
    ax_bar = fig.add_subplot(gs[0, :])
    colors = ["#2A9D8F" if f >= 0.85 else "#E9C46A" if f >= 0.6 else "#E63946"
              for f in valid["f1"]]
    yp = range(len(valid))
    ax_bar.barh(list(yp), valid["f1"].values, color=colors, height=0.7)
    ax_bar.set_yticks(list(yp))
    ax_bar.set_yticklabels(
        [f"{r['aoi'][:40]}  (n={r['n']})" for _, r in valid.iterrows()],
        fontsize=7
    )
    ax_bar.set_xlabel("F1 Score", fontsize=10)
    ax_bar.set_title("Per-AOI F1 Score (sorted, test tiles only)", fontsize=11)
    ax_bar.axvline(valid["f1"].mean(), ls="--", color="#1D3557", lw=1.5,
                   label=f"Mean F1 = {valid['f1'].mean():.4f}")
    ax_bar.legend(fontsize=9)
    ax_bar.set_xlim(0, 1.05)

    # B) Violin — F1 distribution
    ax_v = fig.add_subplot(gs[1, 0])
    sns.violinplot(y=valid["f1"], inner="box", color="#457B9D", ax=ax_v, linewidth=1.2)
    ax_v.set_ylabel("F1 Score", fontsize=10)
    ax_v.set_title("F1 Distribution across AOIs", fontsize=10)

    # C) n_waste scatter vs F1
    ax_s = fig.add_subplot(gs[1, 1])
    sc = ax_s.scatter(valid["n_waste"], valid["f1"],
                      c=valid["accuracy"], cmap="viridis",
                      s=60, edgecolors="grey", lw=0.5)
    cb = plt.colorbar(sc, ax=ax_s)
    cb.set_label("Accuracy", fontsize=9)
    ax_s.set_xlabel("Waste tiles per AOI", fontsize=10)
    ax_s.set_ylabel("F1 Score", fontsize=10)
    ax_s.set_title("F1 vs Waste Tile Count", fontsize=10)

    fig.suptitle("Per-AOI Robustness Analysis", fontsize=14, y=1.01)
    out = out_dir / "per_aoi_metrics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Per-AOI metrics plot saved → {out.name}")


# ───────────────────────────────────────────────────────────────────────────
# 6. Markdown robustness report
# ───────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, ci_df: pd.DataFrame,
                 aoi_df: pd.DataFrame, roc_auc: float,
                 out_dir: Path) -> None:
    valid_aoi = aoi_df.dropna(subset=["f1"])
    low_aois  = valid_aoi[valid_aoi["f1"] < 0.7].sort_values("f1")

    report = f"""# Waste Classification — Robustness Report

> Generated by `03_evaluate_robustness.py`  
> Model: `data/models/train/weights/best.pt`  
> Evaluation split: **test** ({len(df)} tiles)

---

## 1. Overall Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy_score(df['true_label'], df['pred_label']):.4f} |
| F1 Score | {f1_score(df['true_label'], df['pred_label'], zero_division=0):.4f} |
| Precision | {precision_score(df['true_label'], df['pred_label'], zero_division=0):.4f} |
| Recall | {recall_score(df['true_label'], df['pred_label'], zero_division=0):.4f} |
| ROC AUC | {roc_auc:.4f} |

---

## 2. Bootstrap 95 % Confidence Intervals ({ci_df['mean'].count()} iterations)

| Metric | Observed | 95 % CI Low | 95 % CI High | Std |
|--------|----------|-------------|--------------|-----|
"""
    for _, row in ci_df.iterrows():
        report += (f"| {row['metric'].capitalize()} | {row['observed']:.4f} | "
                   f"{row['ci_lo']:.4f} | {row['ci_hi']:.4f} | {row['std']:.4f} |\n")

    report += f"""
---

## 3. Per-AOI Summary

| Stat | F1 | Precision | Recall | Accuracy |
|------|----|-----------|--------|----------|
| Mean | {valid_aoi['f1'].mean():.4f} | {valid_aoi['precision'].mean():.4f} | {valid_aoi['recall'].mean():.4f} | {valid_aoi['accuracy'].mean():.4f} |
| Std  | {valid_aoi['f1'].std():.4f}  | {valid_aoi['precision'].std():.4f}  | {valid_aoi['recall'].std():.4f}  | {valid_aoi['accuracy'].std():.4f}  |
| Min  | {valid_aoi['f1'].min():.4f}  | {valid_aoi['precision'].min():.4f}  | {valid_aoi['recall'].min():.4f}  | {valid_aoi['accuracy'].min():.4f}  |
| Max  | {valid_aoi['f1'].max():.4f}  | {valid_aoi['precision'].max():.4f}  | {valid_aoi['recall'].max():.4f}  | {valid_aoi['accuracy'].max():.4f}  |

AOIs evaluated (with both classes): **{len(valid_aoi)}** / {len(aoi_df)}

"""
    if len(low_aois) > 0:
        report += "### Low-Performing AOIs (F1 < 0.70)\n\n"
        report += "| AOI | n_waste | n_bg | F1 | Recall | Precision |\n"
        report += "|-----|---------|------|-----|--------|----------|\n"
        for _, row in low_aois.iterrows():
            report += (f"| {row['aoi']} | {int(row['n_waste'])} | {int(row['n_bg'])} | "
                       f"{row['f1']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} |\n")
    else:
        report += "### All AOIs achieve F1 ≥ 0.70 ✓\n\n"

    report += """
---

## 4. Robustness Methods

### 4.1 Bootstrap Confidence Intervals
All reported metrics (F1, AUC, Precision, Recall, Accuracy) are accompanied  
by 95 % bootstrap CIs computed over the full test set (sampling with  
replacement, 2 000 iterations). A narrow CI indicates the metric is  
reliable on this test set size.

### 4.2 Metric Stability vs Sample Size
Metrics were evaluated on random subsets of the test set of increasing size  
(bootstrap stability curve). This shows whether the test-set estimate  
converges and whether a smaller evaluation set would give comparable  
conclusions.

### 4.3 Per-AOI Variance
F1, Precision, Recall, and Accuracy are computed independently for each of  
the 29 AOIs. High per-AOI variance indicates the model may not generalise  
uniformly across urban contexts (e.g., different city morphologies, spatial  
waste distribution). AOIs with F1 < 0.70 are flagged for diagnostic review.

### 4.4 Class Balance at AOI Level
The scatter plot of F1 vs waste-tile count per AOI reveals whether the model  
degrades on sparsely labelled sites, indicating potential class-imbalance  
sensitivity in low-density waste areas.

### 4.5 Augmentation Robustness (recommended next step)
To further assess robustness, evaluate the best.pt checkpoint on  
geometrically and photometrically augmented test tiles (rotation ×6,  
brightness ±20 %, Gaussian blur σ=1.5). A robust model should maintain  
F1 drop < 0.03 under these perturbations.

---

## 5. Figures

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curve with AUC and optimal Youden threshold |
| `metric_vs_sample_size.png` | F1, AUC, Accuracy stability vs test-set size |
| `bootstrap_ci.png` | 95 % bootstrap CIs for all core metrics |
| `per_aoi_metrics.png` | Per-AOI F1 bar chart, violin, and waste-count scatter |
"""

    out = out_dir / "robustness_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"  ✓ Robustness report saved → {out.name}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness: ROC, sample-size curve, bootstrap CIs."
    )
    parser.add_argument(
        "--weights",
        default=str(MODELS_DIR / "train" / "weights" / "best.pt"),
        help="Path to YOLO .pt weights file (default: data/models/train/weights/best.pt)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split to evaluate (default: test). Ignored if --test-dir is set.",
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="Path to a test folder with waste/ and background/ subfolders. "
             "Overrides --split / DATASET_DIR when provided.",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=2000,
        help="Bootstrap iterations for CI (default: 2000)",
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR),
        help="Output directory for plots and report",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip inference; reload predictions from <out>/test_predictions.csv "
             "and regenerate all plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("WASTE MODEL — ROBUSTNESS EVALUATION")
    print("=" * 60)
    test_dir = Path(args.test_dir) if args.test_dir else None

    # ── Plot-only mode: skip inference entirely ──────────────────────────
    if args.plot_only:
        pred_csv = out_dir / "test_predictions.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(
                f"--plot-only requires {pred_csv}\n"
                "Run without --plot-only first to generate it."
            )
        print(f"  [plot-only] Loading predictions from {pred_csv}")
        df = pd.read_csv(pred_csv)
        print(f"  Total predictions: {len(df)}")
        overall_f1  = f1_score(df["true_label"], df["pred_label"], zero_division=0)
        overall_auc = roc_auc_score(df["true_label"], df["waste_conf"])
        print(f"  Overall F1  = {overall_f1:.4f}")
        print(f"  Overall AUC = {overall_auc:.4f}")
        print(f"  Out dir : {out_dir}\n")

        print("► Plotting ROC curve …")
        roc_auc = plot_roc(df, out_dir)
        print("\n► Computing metric stability vs sample size …")
        plot_metric_vs_sample_size(df, out_dir)
        print(f"\n► Bootstrapping CIs ({args.bootstrap_n} iterations) …")
        ci_df = bootstrap_ci(df, n_boot=args.bootstrap_n)
        plot_bootstrap_ci(ci_df, out_dir)
        print("\n► Computing per-AOI metrics …")
        aoi_df = build_per_aoi_df(df)
        plot_per_aoi(aoi_df, out_dir)
        aoi_csv = out_dir / "test_file_metrics_29AOI_robustness.csv"
        aoi_df.to_csv(aoi_csv, index=False)
        print(f"  ✓ Per-AOI CSV saved → {aoi_csv.name}")
        print("\n► Writing robustness report …")
        write_report(df, ci_df, aoi_df, roc_auc, out_dir)
        print("\n" + "=" * 60)
        print("DONE — all outputs in:  " + str(out_dir))
        print("=" * 60)
        return
    # ─────────────────────────────────────────────────────────────────────

    print(f"  Weights : {args.weights}")
    print(f"  Test dir: {test_dir if test_dir else DATASET_DIR / args.split}")
    print(f"  Out dir : {out_dir}")
    print()
    # 1. Load model
    model = YOLO(args.weights)

    # 2. Collect images
    print("► Collecting test images …")
    images_by_class = collect_test_images(args.split, test_dir=test_dir)
    for cls, imgs in images_by_class.items():
        print(f"    {cls}: {len(imgs)} images")

    # 3. Run inference
    print("\n► Running inference …")
    df = run_inference(model, images_by_class)
    print(f"  Total predictions: {len(df)}")

    # Quick sanity print
    overall_f1  = f1_score(df["true_label"], df["pred_label"], zero_division=0)
    overall_auc = roc_auc_score(df["true_label"], df["waste_conf"])
    print(f"  Overall F1  = {overall_f1:.4f}")
    print(f"  Overall AUC = {overall_auc:.4f}")

    # 4. ROC curve
    print("\n► Plotting ROC curve …")
    roc_auc = plot_roc(df, out_dir)

    # 5. Metric vs sample size
    print("\n► Computing metric stability vs sample size …")
    plot_metric_vs_sample_size(df, out_dir)

    # 6. Bootstrap CI
    print(f"\n► Bootstrapping CIs ({args.bootstrap_n} iterations) …")
    ci_df = bootstrap_ci(df, n_boot=args.bootstrap_n)
    plot_bootstrap_ci(ci_df, out_dir)

    # 7. Per-AOI
    print("\n► Computing per-AOI metrics …")
    aoi_df = build_per_aoi_df(df)
    plot_per_aoi(aoi_df, out_dir)

    # Save per-AOI CSV (enriched with AUC-capable fields)
    aoi_csv = out_dir / "test_file_metrics_29AOI_robustness.csv"
    aoi_df.to_csv(aoi_csv, index=False)
    print(f"  ✓ Per-AOI CSV saved → {aoi_csv.name}")

    # 8. Markdown report
    print("\n► Writing robustness report …")
    write_report(df, ci_df, aoi_df, roc_auc, out_dir)

    # 9. Save full predictions
    pred_csv = out_dir / "test_predictions.csv"
    df.to_csv(pred_csv, index=False)
    print(f"  ✓ Full predictions saved → {pred_csv.name}")

    print("\n" + "=" * 60)
    print("DONE — all outputs in:  " + str(out_dir))
    print("=" * 60)


if __name__ == "__main__":
    main()

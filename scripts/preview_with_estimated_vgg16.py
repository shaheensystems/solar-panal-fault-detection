"""
*** PREVIEW ONLY — NOT FOR CITATION IN THE THESIS ***

Generates a preview of what the 5-fold cross-validation summary/charts would
look like once VGG16's remaining folds (currently 2/5 real) are complete, by
filling folds 3-5 with SYNTHETIC estimated values (jittered around the mean
of VGG16's 2 real folds) — purely so the chart layout/shape can be previewed
before the real training finishes.

This deliberately does NOT touch output_cv/cache/ (the real, resumable fold
cache) — writing fake fold3/4/5 entries there would make
scripts/run_cross_validation.py think those folds are already done and skip
real training forever. All synthetic data lives only in this script's own
in-memory DataFrame and its clearly-prefixed PREVIEW_* output files, fully
separate from the real cv_summary.csv / cv_per_fold_results.csv / CV_RESULTS.md
/ fig_cv_*.png produced by scripts/summarize_cross_validation.py from
genuine cached fold results.

Every output file, chart, and table is watermarked "ESTIMATED" so this can
never be mistaken for real cross-validation results.

Run: python scripts/preview_with_estimated_vgg16.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path("output_cv")
CACHE_DIR = OUT_DIR / "cache"
N_SPLITS_TARGET = 5
RNG = np.random.default_rng(100)

MODEL_ORDER = ["MobileNetV2", "EfficientNet-Lite", "ShuffleNetV2", "VGG16"]
COLORS = {"MobileNetV2": "steelblue", "EfficientNet-Lite": "coral",
          "ShuffleNetV2": "mediumseagreen", "VGG16": "mediumpurple"}
METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# ---------------------------------------------------------------------------
# Load real cached folds
# ---------------------------------------------------------------------------
entries = [json.loads(f.read_text()) for f in sorted(CACHE_DIR.glob("*_fold*.json"))]
df = pd.DataFrame(entries)
df["estimated"] = False

# ---------------------------------------------------------------------------
# Synthesize VGG16's missing folds: jitter around the mean of its real folds,
# using a jitter magnitude derived from the *other* models' observed fold-to-
# fold spread (VGG16 only has 2 real folds, too few to estimate its own std
# meaningfully) — this keeps the synthetic points from looking suspiciously
# identical without pretending to know VGG16's true variance.
# ---------------------------------------------------------------------------
vgg_real = df[(df["model"] == "VGG16") & (~df["estimated"])].sort_values("fold")
n_real = len(vgg_real)
n_missing = N_SPLITS_TARGET - n_real

if n_missing > 0:
    other_models_std = df[df["model"] != "VGG16"].groupby("model")[METRIC_COLS].std().mean()
    synthetic_rows = []
    for i in range(n_missing):
        fold_idx = n_real + i
        row = {"model": "VGG16", "fold": fold_idx, "estimated": True,
               "epochs_run": int(vgg_real["epochs_run"].mean()),
               "train_time_s": float(vgg_real["train_time_s"].mean()),
               "n_train": int(vgg_real["n_train"].mean()),
               "n_val": int(vgg_real["n_val"].mean()),
               "n_test": int(vgg_real["n_test"].mean())}
        for col in METRIC_COLS:
            mean_val = vgg_real[col].mean()
            jitter = RNG.normal(0, other_models_std[col] * 0.6)
            row[col] = float(np.clip(mean_val + jitter, 0, 1))
        synthetic_rows.append(row)
    df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

df = df.sort_values(["model", "fold"]).reset_index(drop=True)
df.to_csv(OUT_DIR / "PREVIEW_cv_per_fold_results.csv", index=False)

models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

summary = df.groupby("model").agg(
    n_folds=("fold", "count"),
    n_estimated=("estimated", "sum"),
    mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
    mean_f1=("f1_macro", "mean"), std_f1=("f1_macro", "std"),
    mean_precision=("precision_macro", "mean"), std_precision=("precision_macro", "std"),
    mean_recall=("recall_macro", "mean"), std_recall=("recall_macro", "std"),
).reindex(models_present).reset_index()
summary.to_csv(OUT_DIR / "PREVIEW_cv_summary.csv", index=False)

print("!" * 100)
print("PREVIEW ONLY — VGG16 folds below marked 'estimated=True' are SYNTHETIC, not real training runs.")
print("Do not cite these numbers. Re-run scripts/run_cross_validation.py to get real ones.")
print("!" * 100)
print(summary.to_string(index=False))

WATERMARK = "PREVIEW ONLY — VGG16 folds 3-5 are SYNTHETIC (not real training runs)"


def add_watermark(fig):
    fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
              fontsize=10, color="red", fontweight="bold")


# ---------------------------------------------------------------------------
# Figure 1: accuracy bar + per-fold line, with estimated VGG16 points marked
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("[PREVIEW] 5-Fold Cross-Validation — VGG16 folds 3-5 estimated",
             fontsize=13, fontweight="bold", color="darkred")

bar_colors = [COLORS[m] for m in models_present]
means = summary.set_index("model").loc[models_present, "mean_accuracy"]
stds = summary.set_index("model").loc[models_present, "std_accuracy"].fillna(0)
n_est = summary.set_index("model").loc[models_present, "n_estimated"]

bars = axes[0].bar(models_present, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85)
for bar, m, s, ne in zip(bars, means, stds, n_est):
    label = f"{m:.4f}\n±{s:.4f}" + (f"\n({int(ne)} estimated)" if ne > 0 else "")
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Test Accuracy", fontsize=12)
axes[0].set_title("Mean Accuracy ± Std Dev (5 folds, VGG16 partly estimated)", fontsize=11)
axes[0].tick_params(axis="x", rotation=15)

for m in models_present:
    sub = df[df["model"] == m].sort_values("fold")
    axes[1].plot(sub["fold"] + 1, sub["accuracy"], marker="o", label=m, color=COLORS[m])
    est_sub = sub[sub["estimated"]]
    if len(est_sub):
        axes[1].scatter(est_sub["fold"] + 1, est_sub["accuracy"], marker="x", s=140,
                         color="red", zorder=5, label=f"{m} (estimated)" if m == "VGG16" else None)
axes[1].set_xlabel("Fold", fontsize=12)
axes[1].set_ylabel("Test Accuracy", fontsize=12)
axes[1].set_title("Per-Fold Accuracy (red X = estimated, not real)", fontsize=11)
axes[1].set_xticks(range(1, N_SPLITS_TARGET + 1))
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle="--", alpha=0.4)

add_watermark(fig)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_DIR / "PREVIEW_fig_cv_accuracy_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: PREVIEW_fig_cv_accuracy_summary.png")

# ---------------------------------------------------------------------------
# Figure 2: macro-F1 bar + accuracy box plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("[PREVIEW] VGG16 folds 3-5 estimated", fontsize=12, fontweight="bold", color="darkred")

f1_means = summary.set_index("model").loc[models_present, "mean_f1"]
f1_stds = summary.set_index("model").loc[models_present, "std_f1"].fillna(0)
bars = axes[0].bar(models_present, f1_means, yerr=f1_stds, capsize=6, color=bar_colors, alpha=0.85)
for bar, v, s, ne in zip(bars, f1_means, f1_stds, n_est):
    label = f"{v:.4f}\n±{s:.4f}" + (f"\n({int(ne)} estimated)" if ne > 0 else "")
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Macro F1-Score", fontsize=12)
axes[0].set_title("Mean Macro-F1 ± Std Dev", fontsize=11)
axes[0].tick_params(axis="x", rotation=15)

box_data = [df[df["model"] == m]["accuracy"].values for m in models_present]
bp = axes[1].boxplot(box_data, tick_labels=models_present, patch_artist=True)
for patch, m in zip(bp["boxes"], models_present):
    patch.set_facecolor(COLORS[m])
    patch.set_alpha(0.6)
axes[1].set_ylabel("Test Accuracy", fontsize=12)
axes[1].set_title("Accuracy Distribution Across Folds", fontsize=11)
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

add_watermark(fig)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_DIR / "PREVIEW_fig_cv_f1_and_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: PREVIEW_fig_cv_f1_and_distribution.png")

# ---------------------------------------------------------------------------
# Written report
# ---------------------------------------------------------------------------
lines = []
lines.append("# [PREVIEW] 5-Fold Cross-Validation — VGG16 folds 3-5 ESTIMATED\n")
lines.append("> **DO NOT CITE.** This file exists only to preview chart layout/shape "
              "before VGG16's real remaining folds finish training. VGG16 folds 3-5 "
              "below are synthetic (jittered around the mean of its 2 real folds, "
              "using the other models' observed fold-to-fold spread) — not measurements "
              "from actual training runs. The honest, real-data-only version of this "
              "report is `CV_RESULTS.md` (VGG16 correctly shown there as partial, n=2).\n")
lines.append("\n## Summary (mean ± std across folds — VGG16 partly synthetic)\n")
lines.append("| Model | Folds | Estimated folds | Accuracy | Macro F1 |")
lines.append("|---|---|---|---|---|")
for _, row in summary.iterrows():
    lines.append(f"| {row['model']} | {int(row['n_folds'])}/{N_SPLITS_TARGET} | "
                  f"{int(row['n_estimated'])} | "
                  f"{row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f} | "
                  f"{row['mean_f1']:.4f} ± {row['std_f1']:.4f} |")

lines.append("\n## Per-fold results (⚠ = estimated, not a real training run)\n")
lines.append("| Model | Fold | Accuracy | Macro F1 | Real / Estimated |")
lines.append("|---|---|---|---|---|")
for _, row in df.iterrows():
    flag = "⚠ ESTIMATED" if row["estimated"] else "Real"
    lines.append(f"| {row['model']} | {int(row['fold']) + 1} | {row['accuracy']:.4f} | "
                  f"{row['f1_macro']:.4f} | {flag} |")

(OUT_DIR / "PREVIEW_CV_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved: PREVIEW_CV_RESULTS.md")
print("\nDone. Remember: PREVIEW_* files are not real data.")

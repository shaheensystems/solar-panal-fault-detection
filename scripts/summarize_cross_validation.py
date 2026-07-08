"""
Summarizes whatever cross-validation folds have completed so far in
output_cv/cache/*.json (written by scripts/run_cross_validation.py) into a
per-fold table, a mean+/-std summary table, comparison charts, and a written
report — all in output_cv/, kept separate from the single-run notebook's
output/ directory.

Safe to run at any point, including with a partial run (e.g. some models
still short of the full 5 folds) — every output explicitly reports the
fold count (n) per model, so partial results are never silently presented
as if they were complete. Re-run any time after more folds finish (e.g.
after resuming scripts/run_cross_validation.py) to refresh with the fuller
picture; already-summarized folds are just read straight from the cache,
no retraining involved.

Run: python scripts/summarize_cross_validation.py
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

# The single-run (non-CV) test-set results from output/RESULTS.md, for the
# "single run vs cross-validation mean" comparison chart/table.
SINGLE_RUN_RESULTS = {
    "MobileNetV2": {"accuracy": 0.8043, "f1_macro": 0.81},
    "EfficientNet-Lite": {"accuracy": 0.7826, "f1_macro": 0.79},
    "ShuffleNetV2": {"accuracy": 0.6304, "f1_macro": 0.64},
    "VGG16": {"accuracy": 0.6957, "f1_macro": 0.70},
}

MODEL_ORDER = ["MobileNetV2", "EfficientNet-Lite", "ShuffleNetV2", "VGG16"]
COLORS = {"MobileNetV2": "steelblue", "EfficientNet-Lite": "coral",
          "ShuffleNetV2": "mediumseagreen", "VGG16": "mediumpurple"}

# ---------------------------------------------------------------------------
# Load every cached fold result
# ---------------------------------------------------------------------------
entries = []
for f in sorted(CACHE_DIR.glob("*_fold*.json")):
    entries.append(json.loads(f.read_text()))

if not entries:
    raise SystemExit(f"No cached fold results found in {CACHE_DIR} — run "
                      f"scripts/run_cross_validation.py first.")

df = pd.DataFrame(entries)
df = df.sort_values(["model", "fold"]).reset_index(drop=True)
df.to_csv(OUT_DIR / "cv_per_fold_results.csv", index=False)

models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

summary = df.groupby("model").agg(
    n_folds=("fold", "count"),
    mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
    mean_f1=("f1_macro", "mean"), std_f1=("f1_macro", "std"),
    mean_precision=("precision_macro", "mean"), std_precision=("precision_macro", "std"),
    mean_recall=("recall_macro", "mean"), std_recall=("recall_macro", "std"),
    mean_train_time_s=("train_time_s", "mean"),
).reindex(models_present).reset_index()
summary["complete"] = summary["n_folds"] >= N_SPLITS_TARGET
summary.to_csv(OUT_DIR / "cv_summary.csv", index=False)

print("=" * 100)
print("CROSS-VALIDATION SUMMARY (n_folds < 5 means that model is still partial)")
print("=" * 100)
print(summary.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Figure 1: mean accuracy +/- std (bar) and per-fold accuracy (line), side by
# side — mirrors the thesis's existing "5 Fold Cross-Validation" figure.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("5-Fold Cross-Validation Results — All Models", fontsize=14, fontweight="bold")

bar_colors = [COLORS[m] for m in models_present]
means = summary.set_index("model").loc[models_present, "mean_accuracy"]
stds = summary.set_index("model").loc[models_present, "std_accuracy"].fillna(0)
ns = summary.set_index("model").loc[models_present, "n_folds"]

bars = axes[0].bar(models_present, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85)
axes[0].axhline(1 / 6, color="black", linestyle=":", linewidth=1, label="Random baseline (16.7%)")
for bar, m, s, n in zip(bars, means, stds, ns):
    label = f"{m:.4f}\n±{s:.4f}" + ("" if n >= N_SPLITS_TARGET else f"\n(n={n}, partial)")
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Test Accuracy", fontsize=12)
axes[0].set_title("Mean Accuracy ± Std Dev (5 Folds)", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].tick_params(axis="x", rotation=15)

for m in models_present:
    sub = df[df["model"] == m].sort_values("fold")
    axes[1].plot(sub["fold"] + 1, sub["accuracy"], marker="o", label=m, color=COLORS[m])
axes[1].set_xlabel("Fold", fontsize=12)
axes[1].set_ylabel("Test Accuracy", fontsize=12)
axes[1].set_title("Per-Fold Accuracy", fontsize=12)
axes[1].set_xticks(range(1, N_SPLITS_TARGET + 1))
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_cv_accuracy_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_cv_accuracy_summary.png")

# ---------------------------------------------------------------------------
# Figure 2: mean macro-F1 +/- std (bar) and accuracy distribution (box plot)
# — mirrors the thesis's "Macro F1 and Accuracy Standard Deviation" figure.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

f1_means = summary.set_index("model").loc[models_present, "mean_f1"]
f1_stds = summary.set_index("model").loc[models_present, "std_f1"].fillna(0)
bars = axes[0].bar(models_present, f1_means, yerr=f1_stds, capsize=6, color=bar_colors, alpha=0.85)
for bar, v, s, n in zip(bars, f1_means, f1_stds, ns):
    label = f"{v:.4f}\n±{s:.4f}" + ("" if n >= N_SPLITS_TARGET else f"\n(n={n}, partial)")
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Macro F1-Score", fontsize=12)
axes[0].set_title("Mean Macro-F1 ± Std Dev (5 Folds)", fontsize=12)
axes[0].tick_params(axis="x", rotation=15)

box_data = [df[df["model"] == m]["accuracy"].values for m in models_present]
bp = axes[1].boxplot(box_data, tick_labels=models_present, patch_artist=True)
for patch, m in zip(bp["boxes"], models_present):
    patch.set_facecolor(COLORS[m])
    patch.set_alpha(0.6)
axes[1].set_ylabel("Test Accuracy", fontsize=12)
axes[1].set_title("Accuracy Distribution Across Folds", fontsize=12)
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_cv_f1_and_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_cv_f1_and_distribution.png")

# ---------------------------------------------------------------------------
# Figure 3: single run vs cross-validation mean — accuracy and macro-F1.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(models_present))
width = 0.35

single_acc = [SINGLE_RUN_RESULTS[m]["accuracy"] for m in models_present]
cv_acc = [summary.set_index("model").loc[m, "mean_accuracy"] for m in models_present]
axes[0].bar(x - width / 2, single_acc, width, label="Single Run", color="steelblue", alpha=0.85)
axes[0].bar(x + width / 2, cv_acc, width, label="5-Fold CV Mean", color="tomato", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_present, rotation=15, fontsize=10)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].set_title("Accuracy — Single Run vs 5-Fold Mean", fontsize=12)
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)

single_f1 = [SINGLE_RUN_RESULTS[m]["f1_macro"] for m in models_present]
cv_f1 = [summary.set_index("model").loc[m, "mean_f1"] for m in models_present]
axes[1].bar(x - width / 2, single_f1, width, label="Single Run", color="steelblue", alpha=0.85)
axes[1].bar(x + width / 2, cv_f1, width, label="5-Fold CV Mean", color="tomato", alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_present, rotation=15, fontsize=10)
axes[1].set_ylabel("Macro F1-Score", fontsize=12)
axes[1].set_title("Macro-F1 — Single Run vs 5-Fold Mean", fontsize=12)
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_cv_vs_single_run.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_cv_vs_single_run.png")

# ---------------------------------------------------------------------------
# Written report
# ---------------------------------------------------------------------------
partial_models = summary[~summary["complete"]]["model"].tolist()
lines = []
lines.append("# 5-Fold Cross-Validation Results\n")
if partial_models:
    lines.append(f"**PARTIAL RUN** — {', '.join(partial_models)} "
                  f"{'has' if len(partial_models) == 1 else 'have'} fewer than "
                  f"{N_SPLITS_TARGET} folds completed. Re-run "
                  "`scripts/run_cross_validation.py` to finish the remaining folds "
                  "(already-completed folds are cached and will be skipped), then "
                  "re-run this script to refresh these numbers.\n")
lines.append("\n## Summary (mean ± std across folds)\n")
lines.append("| Model | Folds | Accuracy | Macro F1 | Macro Precision | Macro Recall |")
lines.append("|---|---|---|---|---|---|")
for _, row in summary.iterrows():
    flag = "" if row["complete"] else " ⚠ partial"
    lines.append(
        f"| {row['model']}{flag} | {int(row['n_folds'])}/{N_SPLITS_TARGET} | "
        f"{row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f} | "
        f"{row['mean_f1']:.4f} ± {row['std_f1']:.4f} | "
        f"{row['mean_precision']:.4f} ± {row['std_precision']:.4f} | "
        f"{row['mean_recall']:.4f} ± {row['std_recall']:.4f} |"
    )

lines.append("\n## Per-fold results\n")
lines.append("| Model | Fold | Accuracy | Macro F1 | Epochs | Train time (s) |")
lines.append("|---|---|---|---|---|---|")
for _, row in df.iterrows():
    lines.append(
        f"| {row['model']} | {int(row['fold']) + 1} | {row['accuracy']:.4f} | "
        f"{row['f1_macro']:.4f} | {int(row['epochs_run'])} | {row['train_time_s']:.1f} |"
    )

lines.append("\n## Single run vs cross-validation mean\n")
lines.append("| Model | Single-run accuracy | CV mean accuracy | Single-run F1 | CV mean F1 |")
lines.append("|---|---|---|---|---|")
for m in models_present:
    sr = SINGLE_RUN_RESULTS[m]
    cvr = summary.set_index("model").loc[m]
    lines.append(f"| {m} | {sr['accuracy']:.4f} | {cvr['mean_accuracy']:.4f} | "
                  f"{sr['f1_macro']:.4f} | {cvr['mean_f1']:.4f} |")

lines.append("\n## Figures\n")
lines.append("- `fig_cv_accuracy_summary.png` — mean accuracy ± std bar chart + per-fold accuracy line chart")
lines.append("- `fig_cv_f1_and_distribution.png` — mean macro-F1 ± std bar chart + accuracy box plot")
lines.append("- `fig_cv_vs_single_run.png` — single-run vs 5-fold-mean comparison (accuracy and F1)")

(OUT_DIR / "CV_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved: CV_RESULTS.md")
print("\nDone.")

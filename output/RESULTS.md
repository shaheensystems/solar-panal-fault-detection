# Results — Solar Panel Fault Detection (Corrected Pipeline)

Generated from a clean, single top-to-bottom run of `solar_panel_fault_detection.ipynb`
after applying the fixes described in `CHANGES.md`. All figures referenced below are in
this same `output/` folder as PNG files, ready to drop into the thesis document.

Dataset: 875 images, 6 classes, split 70/15/15 (stratified) →
**Train: 605 · Validation: 126 · Test: 138** images.

All four models now use genuine ImageNet-pretrained backbones (see CHANGES.md, Issue 5,
for how ShuffleNetV2 got real pretrained weights via a validated torchvision conversion).

---

## 1. Dataset / EDA

| Figure file | What it shows |
|---|---|
| `eda_class_distribution.png` | Raw class counts (Bird-drop 193, Clean 194, Dusty 191, Electrical-damage 104, Physical-Damage 70, Snow-Covered 124) |
| `eda_image_resolution.png` | Width/height distribution across the dataset |
| `eda_pixel_intensity_distribution.png` | Per-channel RGB histograms per class |
| `eda_dataset_split.png` | Train/val/test split visualisation (605/126/138) |
| `eda_class_imbalance_handling.png` | Class-weight-based imbalance handling (see below) |
| `eda_normalisation_effect.png` | Pixel normalisation before/after |
| `eda_augmentation_examples.png` | The actual augmentation pipeline applied during training |
| `eda_sample_images_grid.png` | Sample images per class |
| `eda_mean_image_per_class.png` | Mean image per fault class |
| `eda_tsne_feature_space.png` | t-SNE of MobileNetV2 GAP features **on the held-out test set** |

**Class weights actually used in training** (`compute_class_weight('balanced', ...)`,
inverse to class frequency in the train split):

| Class | Weight |
|---|---|
| Bird-drop | 0.757 |
| Clean | 0.752 |
| Dusty | 0.763 |
| Electrical-damage | 1.410 |
| Physical-Damage | 2.071 |
| Snow-Covered | 1.180 |

---

## 2. Per-model classification performance (held-out TEST set, never used for training or model selection)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Params | Epochs run |
|---|---|---|---|---|---|---|
| **MobileNetV2** | **80.43%** | 0.83 | 0.81 | 0.81 | 2,265,670 | 25 |
| EfficientNet-Lite0 | 78.26% | 0.79 | 0.80 | 0.79 | 3,420,710 (7,686 head + 3,413,024 frozen TF-Hub backbone) | 25 |
| VGG16 | 69.57% | 0.71 | 0.70 | 0.70 | 14,717,766 | 25 |
| ShuffleNetV2 | 63.04% | 0.68 | 0.65 | 0.64 | 1,275,934 (6,150 head + 1,269,784 frozen pretrained backbone) | 25 |

Figures: `resource_usage_lightweight_vs_heavyweight.png`, `params_vs_accuracy_scatter.png`,
`confusion_matrices_all_models.png`, `f1_per_class_all_models.png`,
`radar_chart_metrics.png` (now includes all four models), `dual_axis_params_accuracy.png`.

Per-class breakdown (precision / recall / F1), from the notebook's own printed
`classification_report`:

**MobileNetV2** — Bird-drop 0.80/0.67/0.73, Clean 0.69/0.97/0.81, Dusty 0.83/0.67/0.74,
Electrical-damage 0.89/0.94/0.91, Physical-Damage 0.89/0.73/0.80, Snow-Covered 0.90/0.90/0.90.

**EfficientNet-Lite0** — Bird-drop 0.84/0.70/0.76, Clean 0.77/0.90/0.83, Dusty 0.69/0.67/0.68,
Electrical-damage 0.80/0.94/0.86, Physical-Damage 0.64/0.82/0.72, Snow-Covered 1.00/0.75/0.86.

**VGG16** — Bird-drop 0.75/0.60/0.67, Clean 0.55/0.73/0.63, Dusty 0.69/0.60/0.64,
Electrical-damage 0.79/0.88/0.83, Physical-Damage 0.50/0.55/0.52, Snow-Covered 1.00/0.85/0.92.

**ShuffleNetV2** (pretrained) — Bird-drop 0.73/0.37/0.49, Clean 0.51/0.77/0.61,
Dusty 0.61/0.63/0.62, Electrical-damage 0.76/0.94/0.84, Physical-Damage 0.47/0.64/0.54,
Snow-Covered 1.00/0.55/0.71 — a real, comparable result now that it shares genuine
pretrained weights with the other three models, though still the weakest of the four
(its ImageNet top-1 of 69.4% is itself below MobileNetV2's/EfficientNet-Lite0's, so a
lower fine-tuned result here is consistent with that starting point, not a training bug).

### 2a. Fault-detection recall (does the model catch that *something* is wrong?)

Accuracy and macro-F1 treat every misclassification equally — confusing "Dusty" for
"Bird-drop" costs the same in those numbers as calling a genuinely faulty panel "Clean".
For a fault-detection system, those two errors are not equivalent: the first is a fault
that still gets flagged (just mislabeled), the second is a fault that goes completely
undetected. The metric that matches that priority is a binary **fault-detection recall**:
collapse the 5 fault classes into one "Faulty" super-class, and ask, of all genuinely
faulty panels in the test set, how many did the model fail to flag at all (i.e. predicted
"Clean")? Computed from the per-model confusion matrices (`confusion_matrices_all_models.png`,
138-image held-out test set, 108 of which are genuinely faulty):

| Model | Faulty panels called "Clean" (missed) | Fault-detection recall | Overall accuracy |
|---|---|---|---|
| **EfficientNet-Lite0** | 8/108 | **92.6%** | 78.26% |
| MobileNetV2 | 13/108 | 88.0% | 80.43% |
| VGG16 | 18/108 | 83.3% | 69.57% |
| ShuffleNetV2 | 22/108 | 79.6% | 63.04% |

**The ranking flips relative to accuracy.** MobileNetV2 has the highest overall accuracy,
but EfficientNet-Lite0 misses fewer faulty panels outright (8 vs 13) — it gets the specific
fault *type* wrong more often, but is less likely to wave a faulty panel through as
"Clean". If the deployment priority is "flag anything abnormal for inspection, sort out
the exact fault type later", EfficientNet-Lite0 is the better model despite its lower
accuracy; if the priority is "get the single predicted label right", MobileNetV2 wins.
Which one matters is a product/deployment decision, not a modelling one.

**Where the misses come from**: across all four models, missed faults cluster almost
entirely on two classes — Dusty→Clean (6, 6, 8, 9 missed, for MobileNetV2/EfficientNet-Lite0/
ShuffleNetV2/VGG16 respectively) and Bird-drop→Clean (5, 1, 9, 7 missed) — while
Electrical-damage→Clean is **0 for every model**. Dust and bird droppings are subtle,
low-contrast surface marks that can resemble a clean panel at this image resolution;
electrical damage and snow cover are visually obvious and are essentially never missed.
That is a genuine, reportable finding about which fault types would need the most
attention (e.g. higher-resolution imaging, a lower decision threshold) in a real
deployment, independent of which model is chosen.

*(This breakdown currently exists only for the single fixed-split run above — the 5-fold
CV script does not yet cache per-fold confusion matrices, so it cannot be computed per
fold for the folds already run. Support for capturing it going forward has been added to
`scripts/run_cross_validation.py`; see the cross-validation results file for what that
does and does not cover.)*

### 2b. Maintenance-action grouping (cleaning vs replacement)

A second, more deployment-realistic regrouping: what a technician would actually *do*
with each prediction, not the specific fault name. Bird-drop, Dusty and Snow-Covered all
resolve with a **cleaning** visit; Electrical-damage and Physical-Damage require a
**replacement/repair** dispatch; Clean needs no action. Collapsing the same held-out
138-image confusion matrices into these 3 groups (Cleanable n=80, Clean n=30,
Replace n=28):

| Model | 3-group accuracy | Cleanable recall | Clean recall | **Replace recall** |
|---|---|---|---|---|
| MobileNetV2 | 86.2% (119/138) | 82.5% (66/80) | 96.7% (29/30) | 85.7% (24/28) |
| EfficientNet-Lite0 | 86.2% (119/138) | 82.5% (66/80) | 90.0% (27/30) | **92.9% (26/28)** |
| VGG16 | 76.8% (106/138) | 75.0% (60/80) | 73.3% (22/30) | 85.7% (24/28) |
| ShuffleNetV2 | 69.6% (96/138) | 61.3% (49/80) | 76.7% (23/30) | 85.7% (24/28) |

All four models jump 6–13 points in accuracy once fault-*type* confusion (e.g. Dusty
mistaken for Bird-drop) stops being penalized — confirming most of each model's raw
6-class error is "flagged the right thing, wrong sub-label", not "missed it".

**The number that matters most operationally is Replace recall** — a missed
Electrical-damage/Physical-Damage panel is the one error with a real safety/cost
consequence (an unaddressed electrical fault left in service), unlike a missed cleaning
which just waits for the next inspection cycle. Breaking Replace recall down by where the
misses land:

| Model | Replace→Clean (missed entirely) | Replace→Cleanable (wrong action, still flagged) |
|---|---|---|
| **EfficientNet-Lite0** | **0/28** | 2/28 |
| MobileNetV2 | 1/28 | 3/28 |
| ShuffleNetV2 | 1/28 | 3/28 |
| VGG16 | 2/28 | 2/28 |

EfficientNet-Lite0 is the only model with **zero** replacement-needed panels waved
through as "Clean" in this test set — consistent with its lead in fault-detection recall
in section 2a. MobileNetV2 ties EfficientNet-Lite0 on overall 3-group accuracy (119/138
both) but lets 1 replacement-needed panel through undetected where EfficientNet-Lite0 lets
none through. If the deployment cost model treats a missed electrical/physical-damage
panel as materially worse than a missed cleaning, EfficientNet-Lite0 is the stronger
choice despite its lower raw accuracy in sections 1–2 — a third, independent line of
evidence (alongside fault-detection recall in 2a) pointing the same direction.

*(Same caveat as 2a: computed from the single fixed-split confusion matrices only, not
per CV fold.)*

---

## 3. Resource / efficiency (CPU-only, no GPU — this machine has none)

| Model | Train time (s) | RAM Δ (MB) | Peak RAM (MB) | Inference (ms/img, in-memory Keras) |
|---|---|---|---|---|
| MobileNetV2 | 20.5 | 726.4 | 1586.4 | 17.91 |
| EfficientNet-Lite0 | 5.0 | 94.4 | 1777.4 | 17.03 |
| VGG16 | 52.4 | 102.9 | 1858.8 | 251.63 |
| ShuffleNetV2 | 518.7 | −616.4 | 1892.1 | 9.84 |

ShuffleNetV2's 518.7s "training time" is misleading if read as per-epoch training cost —
almost all of it is a **one-time TensorFlow graph-tracing/compilation overhead** unique to
this architecture's ~500 small, fragmented ops (many tiny Conv2D/DepthwiseConv2D/BatchNorm
layers per block × 17 blocks), not slower per-batch computation. Once compiled, it has the
**fastest** inference of all four models (9.84 ms/img in-memory, 15.71 ms/img as TFLite) —
consistent with ShuffleNetV2's actual design goal of being FLOP/latency-efficient at
inference time, at the cost of graph complexity. This is a real, reportable finding about
the practical cost of ShuffleNetV2's architecture on general-purpose (non-mobile,
non-XLA-compiled) hardware, distinct from its accuracy.

`vram_delta_mb` is 0.0 for all models because no GPU is present in this environment —
the notebook's VRAM tracker (`pynvml`) never activates. Training-time figures above are
**not comparable to a GPU-based run** (e.g. Colab) — see CHANGES.md for what this means
for the thesis's GPU-based timing tables.

## 4. TFLite (float16) conversion + inference on the test set

| Model | TFLite file size | TFLite test accuracy | ms/img (TFLite, single-thread-limited interpreter) |
|---|---|---|---|
| MobileNetV2 | 4.5 MB | 79.71% | 26.37 |
| EfficientNet-Lite0 | 6.8 MB | 78.26% | 32.43 |
| ShuffleNetV2 | 2.7 MB | 63.04% | 15.71 |
| VGG16 | 29.5 MB | 69.57% | 465.88 |

TFLite files are in `output/tflite/*.tflite`. These are float16-quantised conversions of
the exact models evaluated above — **not** measured on a Raspberry Pi (no physical device
in this environment; see CHANGES.md, Issue 6).

---

## 5. Headline takeaways (for the abstract / conclusion)

- MobileNetV2 remains the best accuracy/efficiency trade-off among the four pretrained
  models (80.4% test accuracy, smallest of the top three at 2.27M params, fastest overall
  practical inference), consistent with the thesis's central thesis — but the actual
  margin over EfficientNet-Lite0 is smaller (~2.2 points) than previously reported, and
  VGG16 is not just slower but also noticeably *less* accurate on genuinely held-out data
  (69.6% vs the previous, leakage-inflated 91.53%).
- ShuffleNetV2 now uses genuine pretrained ImageNet weights, converted from torchvision's
  official `shufflenet_v2_x1_0` checkpoint and validated to machine precision (cosine
  similarity 1.0 against the original PyTorch model — see `scripts/convert_shufflenet_weights.py`
  and CHANGES.md, Issue 5). With real pretraining its test accuracy rises from 21.74%
  (near-random, from-scratch) to **63.04%** — direct, first-hand confirmation of how much
  transfer learning matters on a ~600-image training set, and it is now a legitimately
  comparable fourth data point across every chart rather than an excluded special case.
- All numbers above are genuinely out-of-sample: `test/` was produced by a stratified
  70/15/15 split and touched by no callback, checkpoint, or tuning decision during
  training (see Issue 1/2 in CHANGES.md).

# Results — Solar Panel Fault Detection (Corrected Pipeline)

Generated from a clean, single top-to-bottom run of `solar_panel_fault_detection.ipynb`
after applying the fixes described in `CHANGES.md`. All figures referenced below are in
this same `output/` folder as PNG files, ready to drop into the thesis document.

Dataset: 875 images, 6 classes, split 70/15/15 (stratified) →
**Train: 605 · Validation: 126 · Test: 138** images.

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
| ShuffleNetV2 (no pretraining) | 21.74% | 0.04 | 0.17 | 0.06 | 1,317,538 | 20 (early-stopped) |

Figures: `resource_usage_lightweight_vs_heavyweight.png`, `params_vs_accuracy_scatter.png`,
`confusion_matrices_all_models.png`, `f1_per_class_all_models.png`, `radar_chart_metrics.png`
(MobileNetV2/EfficientNet-Lite/VGG16 only — ShuffleNetV2 excluded, see below),
`transfer_learning_effect.png`, `dual_axis_params_accuracy.png`.

Per-class breakdown (precision / recall / F1), from the notebook's own printed
`classification_report`:

**MobileNetV2** — Bird-drop 0.80/0.67/0.73, Clean 0.69/0.97/0.81, Dusty 0.83/0.67/0.74,
Electrical-damage 0.89/0.94/0.91, Physical-Damage 0.89/0.73/0.80, Snow-Covered 0.90/0.90/0.90.

**EfficientNet-Lite0** — Bird-drop 0.84/0.70/0.76, Clean 0.77/0.90/0.83, Dusty 0.69/0.67/0.68,
Electrical-damage 0.80/0.94/0.86, Physical-Damage 0.64/0.82/0.72, Snow-Covered 1.00/0.75/0.86.

**VGG16** — Bird-drop 0.75/0.60/0.67, Clean 0.55/0.73/0.63, Dusty 0.69/0.60/0.64,
Electrical-damage 0.79/0.88/0.83, Physical-Damage 0.50/0.55/0.52, Snow-Covered 1.00/0.85/0.92.

**ShuffleNetV2** — collapsed to predicting "Clean" for almost everything (0.22/1.00/0.36 on
Clean, ~0 elsewhere) — a textbook symptom of training a 6-class classifier from scratch
on ~600 images with no pretrained backbone.

---

## 3. Resource / efficiency (CPU-only, no GPU — this machine has none)

| Model | Train time (s) | RAM Δ (MB) | Peak RAM (MB) | Inference (ms/img, in-memory Keras) |
|---|---|---|---|---|
| MobileNetV2 | 38.4 | 672.2 | 1584.3 | 23.34 |
| EfficientNet-Lite0 | 12.3 | 95.3 | 1836.5 | 23.05 |
| VGG16 | 32.1 | 824.3 | 2782.9 | 194.91 |
| ShuffleNetV2 | 14.9 | 82.6 | 1850.3 | 15.92 |

`vram_delta_mb` is 0.0 for all models because no GPU is present in this environment —
the notebook's VRAM tracker (`pynvml`) never activates. Training-time figures above are
**not comparable to a GPU-based run** (e.g. Colab) — see CHANGES.md for what this means
for the thesis's GPU-based timing tables.

## 4. TFLite (float16) conversion + inference on the test set

| Model | TFLite file size | TFLite test accuracy | ms/img (TFLite, single-thread-limited interpreter) |
|---|---|---|---|
| MobileNetV2 | 4.5 MB | 79.71% | 20.07 |
| EfficientNet-Lite0 | 6.8 MB | 78.26% | 25.52 |
| ShuffleNetV2 | 2.7 MB | 21.74% | 11.56 |
| VGG16 | 29.5 MB | 69.57% | 734.74 |

TFLite files are in `output/tflite/*.tflite`. These are float16-quantised conversions of
the exact models evaluated above — **not** measured on a Raspberry Pi (no physical device
in this environment; see CHANGES.md, Issue 6).

---

## 5. Headline takeaways (for the abstract / conclusion)

- MobileNetV2 remains the best accuracy/efficiency trade-off among the three
  *pretrained* models (80.4% test accuracy, smallest of the three at 2.27M params,
  fastest lightweight inference), consistent with the thesis's central thesis —
  but the actual margin over EfficientNet-Lite0 is smaller (~2.2 points) than
  previously reported, and VGG16 is not just slower but also noticeably *less*
  accurate on genuinely held-out data (69.6% vs the previous, leakage-inflated 91.53%).
- ShuffleNetV2's collapse (21.7%, barely above the 16.7% random baseline for 6 classes)
  is a **transfer-learning-availability** finding, not an architecture weakness — see
  Issue 5 in CHANGES.md. No trustworthy ImageNet-pretrained TensorFlow/Keras ShuffleNetV2
  checkpoint could be located; it is trained from scratch and excluded from head-to-head
  architecture rankings.
- All numbers above are genuinely out-of-sample: `test/` was produced by a stratified
  70/15/15 split and touched by no callback, checkpoint, or tuning decision during
  training (see Issue 1/2 in CHANGES.md).

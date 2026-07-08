# [PREVIEW] 5-Fold Cross-Validation — VGG16 folds 3-5 ESTIMATED

> **DO NOT CITE.** This file exists only to preview chart layout/shape before VGG16's real remaining folds finish training. VGG16 folds 3-5 below are synthetic (jittered around the mean of its 2 real folds, using the other models' observed fold-to-fold spread) — not measurements from actual training runs. The honest, real-data-only version of this report is `CV_RESULTS.md` (VGG16 correctly shown there as partial, n=2).


## Summary (mean ± std across folds — VGG16 partly synthetic)

| Model | Folds | Estimated folds | Accuracy | Macro F1 |
|---|---|---|---|---|
| MobileNetV2 | 5/5 | 0 | 0.7790 ± 0.0130 | 0.7835 ± 0.0210 |
| EfficientNet-Lite | 5/5 | 0 | 0.7595 ± 0.0500 | 0.7653 ± 0.0585 |
| ShuffleNetV2 | 5/5 | 0 | 0.6732 ± 0.0274 | 0.6815 ± 0.0298 |
| VGG16 | 5/5 | 3 | 0.6991 ± 0.0143 | 0.7138 ± 0.0121 |

## Per-fold results (⚠ = estimated, not a real training run)

| Model | Fold | Accuracy | Macro F1 | Real / Estimated |
|---|---|---|---|---|
| EfficientNet-Lite | 1 | 0.7816 | 0.7954 | Real |
| EfficientNet-Lite | 2 | 0.8046 | 0.8281 | Real |
| EfficientNet-Lite | 3 | 0.7931 | 0.7868 | Real |
| EfficientNet-Lite | 4 | 0.6839 | 0.6778 | Real |
| EfficientNet-Lite | 5 | 0.7341 | 0.7384 | Real |
| MobileNetV2 | 1 | 0.7816 | 0.7960 | Real |
| MobileNetV2 | 2 | 0.7989 | 0.8113 | Real |
| MobileNetV2 | 3 | 0.7759 | 0.7563 | Real |
| MobileNetV2 | 4 | 0.7759 | 0.7739 | Real |
| MobileNetV2 | 5 | 0.7630 | 0.7802 | Real |
| ShuffleNetV2 | 1 | 0.6667 | 0.6990 | Real |
| ShuffleNetV2 | 2 | 0.6437 | 0.6601 | Real |
| ShuffleNetV2 | 3 | 0.7184 | 0.7130 | Real |
| ShuffleNetV2 | 4 | 0.6667 | 0.6411 | Real |
| ShuffleNetV2 | 5 | 0.6705 | 0.6945 | Real |
| VGG16 | 1 | 0.7069 | 0.7054 | Real |
| VGG16 | 2 | 0.7011 | 0.7166 | Real |
| VGG16 | 3 | 0.6831 | 0.7229 | ⚠ ESTIMATED |
| VGG16 | 4 | 0.6866 | 0.7265 | ⚠ ESTIMATED |
| VGG16 | 5 | 0.7175 | 0.6977 | ⚠ ESTIMATED |
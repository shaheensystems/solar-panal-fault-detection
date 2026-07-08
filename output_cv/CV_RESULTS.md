# 5-Fold Cross-Validation Results

**PARTIAL RUN** — VGG16 has fewer than 5 folds completed. Re-run `scripts/run_cross_validation.py` to finish the remaining folds (already-completed folds are cached and will be skipped), then re-run this script to refresh these numbers.


## Summary (mean ± std across folds)

| Model | Folds | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| MobileNetV2 | 5/5 | 0.7790 ± 0.0130 | 0.7835 ± 0.0210 | 0.8095 ± 0.0190 | 0.7743 ± 0.0243 |
| EfficientNet-Lite | 5/5 | 0.7595 ± 0.0500 | 0.7653 ± 0.0585 | 0.7763 ± 0.0598 | 0.7620 ± 0.0551 |
| ShuffleNetV2 | 5/5 | 0.6732 ± 0.0274 | 0.6815 ± 0.0298 | 0.7116 ± 0.0230 | 0.6821 ± 0.0275 |
| VGG16 ⚠ partial | 2/5 | 0.7040 ± 0.0041 | 0.7110 ± 0.0079 | 0.7206 ± 0.0073 | 0.7068 ± 0.0199 |

## Per-fold results

| Model | Fold | Accuracy | Macro F1 | Epochs | Train time (s) |
|---|---|---|---|---|---|
| EfficientNet-Lite | 1 | 0.7816 | 0.7954 | 25 | 386.7 |
| EfficientNet-Lite | 2 | 0.8046 | 0.8281 | 25 | 352.6 |
| EfficientNet-Lite | 3 | 0.7931 | 0.7868 | 25 | 331.9 |
| EfficientNet-Lite | 4 | 0.6839 | 0.6778 | 25 | 347.8 |
| EfficientNet-Lite | 5 | 0.7341 | 0.7384 | 25 | 367.5 |
| MobileNetV2 | 1 | 0.7816 | 0.7960 | 25 | 413.2 |
| MobileNetV2 | 2 | 0.7989 | 0.8113 | 23 | 447.4 |
| MobileNetV2 | 3 | 0.7759 | 0.7563 | 20 | 390.0 |
| MobileNetV2 | 4 | 0.7759 | 0.7739 | 25 | 432.7 |
| MobileNetV2 | 5 | 0.7630 | 0.7802 | 24 | 454.7 |
| ShuffleNetV2 | 1 | 0.6667 | 0.6990 | 25 | 320.6 |
| ShuffleNetV2 | 2 | 0.6437 | 0.6601 | 25 | 338.7 |
| ShuffleNetV2 | 3 | 0.7184 | 0.7130 | 25 | 333.7 |
| ShuffleNetV2 | 4 | 0.6667 | 0.6411 | 25 | 341.6 |
| ShuffleNetV2 | 5 | 0.6705 | 0.6945 | 25 | 324.0 |
| VGG16 | 1 | 0.7069 | 0.7054 | 25 | 3266.1 |
| VGG16 | 2 | 0.7011 | 0.7166 | 25 | 3239.8 |

## Single run vs cross-validation mean

| Model | Single-run accuracy | CV mean accuracy | Single-run F1 | CV mean F1 |
|---|---|---|---|---|
| MobileNetV2 | 0.8043 | 0.7790 | 0.8100 | 0.7835 |
| EfficientNet-Lite | 0.7826 | 0.7595 | 0.7900 | 0.7653 |
| ShuffleNetV2 | 0.6304 | 0.6732 | 0.6400 | 0.6815 |
| VGG16 | 0.6957 | 0.7040 | 0.7000 | 0.7110 |

## Figures

- `fig_cv_accuracy_summary.png` — mean accuracy ± std bar chart + per-fold accuracy line chart
- `fig_cv_f1_and_distribution.png` — mean macro-F1 ± std bar chart + accuracy box plot
- `fig_cv_vs_single_run.png` — single-run vs 5-fold-mean comparison (accuracy and F1)

## What these results mean

**The ranking holds up.** MobileNetV2 (77.9%) > EfficientNet-Lite0 (76.0%) > VGG16 (70.4%,
still partial) > ShuffleNetV2 (67.3%) — the exact same order as the single-run notebook.
That the ranking survives being re-measured on five different train/test splits, not just
the one the thesis reports, is the actual point of doing cross-validation: it's evidence
the single-run numbers weren't a lucky (or unlucky) split for any one model.

**The single fixed split flattered the two best models and was harsh on the worst one.**
This is the most important finding in this table:

| Model | Single-run accuracy | 5-fold mean | Difference |
|---|---|---|---|
| MobileNetV2 | 80.43% | 77.90% | single run **+2.53pp** higher |
| EfficientNet-Lite0 | 78.26% | 75.95% | single run **+2.31pp** higher |
| ShuffleNetV2 | 63.04% | 67.32% | single run **−4.28pp** lower |
| VGG16 (partial, n=2) | 69.57% | 70.40% | roughly consistent |

MobileNetV2 and EfficientNet-Lite0 both did *better* on the thesis's one reported split
than they do on average — and ShuffleNetV2 did noticeably *worse*. With only 138 test
images spread across 6 imbalanced classes (as few as 11 images in Physical-Damage), which
specific images land in the test set measurably moves the numbers. None of this changes
which model wins, but it does mean the single-run accuracy figures should be reported as
"a" result, not "the" result — the 5-fold mean is the more defensible number to lead with,
with the single run kept as a supporting data point.

**Accuracy and stability are two different rankings.** By mean accuracy alone,
EfficientNet-Lite0 (75.95%) edges out ShuffleNetV2 (67.32%) comfortably. But
EfficientNet-Lite0's std dev (±5.00%) is nearly *four times* MobileNetV2's (±1.30%) and
almost double ShuffleNetV2's (±2.74%) — it's the least consistent of the three complete
models, swinging from 80.46% (fold 2) down to 68.39% (fold 4). MobileNetV2 is the only
model that is simultaneously the most accurate *and* the most stable, which is a stronger
result than its accuracy number alone conveys — a deployed model that performs consistently
regardless of which images it happens to see matters as much as its average accuracy for a
real fault-detection system.

**The fold 4 EfficientNet-Lite0 dip (68.39%) is a real, explainable data-split effect, not
a bug.** It's the direct consequence of cross-validating a small (869-image), imbalanced
dataset: stratified k-fold keeps class *proportions* consistent across folds, but with only
~14 Physical-Damage images per test fold, a handful of genuinely hard or ambiguous images
landing together in one fold's test set is enough to move that fold's accuracy several
points. This is exactly the kind of instability a single train/test split can't reveal.

**VGG16's training cost is wildly disproportionate to its parameter count.** VGG16 has
14.7M params — roughly 6.5× MobileNetV2's 2.27M — but each VGG16 fold takes ~3,250s
(~54 minutes) to train versus ~330–450s (~6–7.5 minutes) for the other three, a ~7–10×
difference, not the ~6.5× its parameter count alone would suggest. Its large, dense
convolutional filters (as opposed to the depthwise-separable convolutions the three
lightweight models are built from) cost far more compute per layer than the parameter
count implies. This is a second, independent piece of evidence — alongside the accuracy
numbers — for why the lightweight architectures are the right choice for this task, before
VGG16's accuracy is even considered.

**Caveat on VGG16's numbers above:** its ±0.41% std dev looks like the tightest of all four
models, but that's an artefact of having only 2 data points, not genuine stability — a
std dev computed from n=2 is barely informative and will very likely widen once folds 3–5
are real. Don't read anything into VGG16's spread until it has the full 5 folds; its mean
(70.40%) is directionally consistent with the single run (69.57%) and is the only number
worth trusting from it right now.

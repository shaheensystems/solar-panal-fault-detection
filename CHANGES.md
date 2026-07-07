# Changes Made — Code Fixes and Where They Affect the Thesis Document

This documents every change applied to `solar_panel_fault_detection.ipynb`, why, and
—critically—**exactly which parts of `Master_Degree_Project_Report_2026_Feedback.pdf`
now disagree with the code** and need updating before submission. Real numbers from the
corrected run are in `output/RESULTS.md`; all figures/CSVs referenced below are in `output/`.

Nothing was "optimised" for its own sake — every change below is a correctness fix (a bug
that produced wrong or leaked numbers) or a fix required to make previously-broken cells
run at all. Cells that already worked (all EDA, dataset splitting, TFLite conversion
logic, etc.) were left untouched.

---

## Headline: what actually changed

| Model | Report's number (validation, leaked) | Corrected number (held-out test) |
|---|---|---|
| MobileNetV2 | 93.22% | **80.43%** |
| EfficientNet-Lite | 91.53% | **78.26%** |
| VGG16 | 91.53% | **69.57%** |
| ShuffleNetV2 | 27.12% (from scratch) | **63.04%** (now genuinely pretrained — see Issue 5) |

Three of the four numbers are lower because they were previously measured on leaked or
validation-selected data (Issues 1–2). ShuffleNetV2's number moved the other direction —
it now has real pretrained ImageNet weights it never had before, so 63.04% is a
substantially *stronger*, and for the first time genuinely comparable, result. Full
per-model precision/recall/F1, resource usage, and TFLite numbers are in `output/RESULTS.md`.

---

## Issue 1 — No real test set / train-val leakage
**Thesis sections affected:** Abstract ("93.22% validation accuracy"), 3.3.6 *Data Split*
(claims stratified 70/15/15 test split existed), Table 1 in the proposal.

The notebook had two dataset-loading cells: a correct 70/15/15 stratified split via
`splitfolders`, immediately followed by a leftover, unused 80/20 `validation_split` call
with **mismatched seeds** (`seed=100` vs `seed=200`) that silently overwrote `train`/`validation`
right after — meaning every subsequent cell trained and evaluated on leaked, non-stratified
data despite the correct split code sitting right above it.
**Fix:** deleted the leftover cell; `train`/`validation`/`test` now come only from the
stratified split (605/126/138 images — see `output/RESULTS.md` §1).
**Where to edit the thesis:** every accuracy number quoted anywhere is affected (see table above).

## Issue 2 — Final metrics computed on validation data, not test data
**Thesis sections affected:** Abstract, Table 5 *Performance Metrics*, Table 10
*Consolidated Results Overview*, Table 11 *Accuracy Ranking (RQ1)*, Figures 12–20
(per-model loss/accuracy + confusion matrices, comparison bar chart), §4.3–4.4, §5.1–5.3.

`ModelCheckpoint`/`EarlyStopping` both select the epoch that scores best **on validation**;
every "accuracy" figure and confusion matrix in the report was then computed on that same
validation set — the equivalent of grading on the practice exam.
**Fix:** a genuine `test` split (never touched by any callback or tuning decision) is now
built once and used for every reported metric, confusion matrix, F1 chart, radar chart,
and the TFLite/Raspberry-Pi-style evaluation. Validation is still used correctly for early
stopping/checkpointing (that part of the methodology, §3.6.3, was already correct).
**Where to edit the thesis:** replace every "validation accuracy" figure/table with the
"test accuracy" numbers in `output/RESULTS.md` §2; rename the column/labels from "Val
Accuracy" to "Test Accuracy" throughout (Tables 5/10/11, Figures 12–20, 28).
New figures: `output/confusion_matrices_all_models.png`, `output/f1_per_class_all_models.png`,
`output/radar_chart_metrics.png`, `output/params_vs_accuracy_scatter.png`.

## Issue 3 — Augmentation & class-imbalance handling described but never actually applied
**Thesis sections affected:** §3.4.2 *Data Augmentation Strategy* (Algorithm 2), §3.4.3
*Handling Imbalance Data*.

The augmentation algorithm and SMOTE-based imbalance handling were documented and
visualised in EDA, but no training cell actually used them — every model trained on raw,
unaugmented, unweighted batches.
**Fix:** a shared `data_augmentation` Keras block (flip, ±20° rotation, 0.8–1.2× zoom,
brightness, Gaussian noise — matching §3.4.2's own description) is now the first layer of
every pretrained model, and `class_weight` (computed via `sklearn.compute_class_weight`
from the actual train-split folder counts) is passed to every `.fit()` call.
**Important discrepancy to fix in the text, not the code:** §3.4.3 specifically claims
**SMOTE on post-GAP feature vectors**. The notebook does not do this anywhere (nor did
the original), and implementing true feature-space SMOTE inside a Keras training loop is
a non-trivial undertaking beyond this pass's scope. Two honest options: (a) change §3.4.3's
text to describe `class_weight`-based reweighting (what the code actually does — a
standard, legitimate imbalance-handling technique, just not SMOTE), or (b) if SMOTE is a
hard requirement, that is additional implementation work not covered here. The class
weights actually computed are in `output/RESULTS.md` §1.

## Issue 4 — VGG16 handicapped: wrong preprocessing + oversized head
**Thesis sections affected:** Table 4 *Architectural comparison* (VGG16 row: "13 Conv
Layers" / no head details given), §5.4 *Computational overhead comparison*.

VGG16 fed raw `[0,255]` pixels into ImageNet-pretrained weights that expect
`vgg16.preprocess_input`'s BGR + mean-subtraction, and used a
`Flatten→Dense(256)` head (~6.4M extra trainable params) versus the
`GlobalAveragePooling2D` head every other model uses.
**Fix:** rebuilt as a Functional model with `vgg16.preprocess_input` and the same GAP head
as the other three models (trainable head now ~3K params instead of ~6.4M).
**Where to edit the thesis:** Table 4's VGG16 row should note "GlobalAveragePooling2D head
(post-fix)" to match the other rows; the 14.7M total-parameter figure in `output/RESULTS.md`
already reflects this (vs. the thesis's 21.14M, which included the old oversized head).

## Issue 5 — ShuffleNetV2 not comparable to the other three (no pretraining) — RESOLVED

**Thesis sections affected:** Table 4 (ShuffleNetV2 "Transfer Learning" row — now
genuinely true), §4.3.3, §5.6 *EDA effect on Performance* / Table 14 *Transfer Learning
Accuracy*, Figure 29 (**retired**, see below), Table 12 *computational overhead metrics*.

**Original finding:** no official TensorFlow/Keras ImageNet-pretrained ShuffleNetV2
checkpoint exists. The only one initially found was an unofficial, unmaintained
**TensorFlow 1.10** `tf.estimator` checkpoint on a personal Google Drive link (68.8%
top-1, no provenance guarantee), which was rejected as too risky to convert.

**Follow-up (this session, on request):** torchvision does officially maintain a
pretrained ShuffleNetV2 x1.0 checkpoint (69.4% ImageNet top-1, hosted on
`download.pytorch.org`, part of torchvision's maintained model zoo — a genuinely
trustworthy source, unlike the TF1 Google Drive checkpoint). It was converted to Keras 3
and **validated to machine precision** against the original PyTorch model (cosine
similarity 1.0, max absolute difference ~1.8×10⁻⁷ on backbone feature vectors — see
`scripts/convert_shufflenet_weights.py` for the full conversion + validation code).

The conversion required rebuilding ShuffleNetV2's block from scratch, because the
original from-scratch implementation in this notebook had **two real architecture bugs**
that also happened to be *why* no direct weight transplant into it was ever possible:

1. Its internal bottleneck width was `out_channels // 4`; the real ShuffleNetV2 (and the
   pretrained weights) use `out_channels // 2` throughout — a bottleneck twice as narrow
   as the actual architecture.
2. Its stride-1 residual path processed the **full** input feature map; real ShuffleNetV2
   splits the input in half and only processes one half (`x2`), keeping the other (`x1`)
   as an identity shortcut — the "channel split" that gives ShuffleNet its efficiency.

Two further PyTorch→TensorFlow porting pitfalls had to be fixed before the conversion
validated (each initially produced a plausible-but-wrong result — cosine similarity 0.84,
then 0.999, before reaching 1.0): TF's `padding='same'` is not always identical to
PyTorch's explicit symmetric `padding=1` for stride>1 convs (fixed with explicit
`ZeroPadding2D` + `'valid'`), and Keras's `BatchNormalization` defaults to
`epsilon=1e-3` vs PyTorch's `1e-5` (small per-layer drift that compounds across 112 BN
layers into a real difference).

**Fix:** ShuffleNetV2 now uses the same frozen-backbone + augmentation + class_weight +
single-phase training recipe as the other three models (the old two-phase/label-smoothing/
cosine-LR scheme was a from-scratch-only workaround, no longer needed). Test accuracy rose
from 21.74% (near-random) to **63.04%**. The pretrained backbone is committed at
`pretrained/shufflenetv2_x1_0_imagenet_backbone.keras` (5.8 MB); the notebook loads it
directly and never needs `torch`/`torchvision` at runtime — those are only required to
re-run the one-time conversion script.

**Where to edit the thesis:**

- Table 4's ShuffleNetV2 row can now honestly say "Pre-trained ImageNet weights
  (torchvision conversion)".
- Table 12 / §5.4 computational-overhead discussion should note ShuffleNetV2's unusually
  high one-time graph-compilation cost on CPU (see `output/RESULTS.md` §3) — real, and an
  interesting practical-deployment finding distinct from its accuracy.
- **Figure 29 / the RQ4 "Transfer Learning Effect" chart no longer has data to show** —
  it specifically compared "with" vs "without" pretraining using ShuffleNetV2 as the sole
  "without" data point. With all four models now pretrained, that comparison group is
  gone. The notebook cell that used to produce Figure 29 now prints an explanatory note
  instead of a chart (see the retired-chart entry below) — §5.6/Table 14 need to either
  drop this specific figure or reframe RQ4 using the *historical* from-scratch result
  (21.74% vs 63.04%, both real measurements taken during this work, see
  `output/RESULTS.md` §5) as a point-in-time ablation rather than a live chart.

## Issue 6 — Hard-coded Raspberry Pi results, no measurement artefacts in the repo
**Thesis sections affected:** §3.8 *Edge Deployment Methodology* (whole section describes
a Pi 3B+ deployment), §3.8.2, Figure 4, §4.7 *Edge Deployment Results*, Figures 26–27,
Tables 9/13, Abstract's "104.9ms... 80.7MB" claim.

The repository contains no Raspberry Pi benchmark script, no raw CSV, and no evidence the
notebook was ever run on Pi hardware — the Pi numbers were dictionary literals. There is
no physical Raspberry Pi in this environment either, so genuine hardware numbers still
cannot be produced here.
**Fix applied:** removed the fabricated Pi numbers entirely rather than leave
unverifiable figures in place; parameter counts in the resource summary are now always
`model.count_params()` reads, never hand-typed (`output/final_resource_summary.csv`).
**What you need to decide for the thesis:** §3.8/§4.7/Figures 26-27/Tables 9,13 describe a
real deployment workflow (TFLite conversion → Pi 3B+ → measured latency/RAM) that this
session could not execute (no hardware). Options: (a) actually run
`output/tflite/*.tflite` on a real Pi 3B+ and report genuine numbers — the TFLite files
are ready and this notebook's own "Strategy 2" batch-inference code
(§ Batch TFLite Inference cell) is the measurement harness, it just needs to run on Pi
hardware instead of this dev machine; or (b) reframe §3.8/§4.7 as "future work" /
methodology-only sections until real hardware numbers exist. Leaving the old fabricated
104.9ms/80.7MB numbers in place is the one option that should not be chosen.

## Issue 7 — "EfficientNet-Lite" was actually EfficientNetB0
**Thesis sections affected:** everywhere "EfficientNet-Lite" is named (Table 4, Table 5,
Figures 14-15, 20, 28, Abstract) — all of these now describe a genuinely different, real
model, so **no restructuring is needed, only the numbers change** (see headline table above).

Real EfficientNet-Lite0 ImageNet-pretrained weights exist and are officially hosted on
**TensorFlow Hub** (`tensorflow/efficientnet/lite0/feature-vector/2`, ~75% top-1 on
ImageNet) — a different distribution channel than `tf.keras.applications`, which only
ships the standard (non-Lite) EfficientNetB0/B1/etc.
**Fix:** the model is now genuinely EfficientNet-Lite0, loaded via `tensorflow_hub`. Since
TF-Hub's module is a legacy `tf_keras` object, it's wrapped in a small
`HubFeatureExtractor(keras.layers.Layer)` adapter (registered serialisable) so it works
inside Keras 3 and can be saved/reloaded normally. Its backbone (3,413,024 params) isn't
auto-discovered by Keras 3's parameter counting when wrapped this way, so the notebook
explicitly adds it back in wherever total params are reported — the 3,420,710 total in
`output/RESULTS.md` already accounts for this.
**Where to edit the thesis:** no text changes needed beyond the numbers — "EfficientNet-Lite"
is accurate now, whereas before it was a mislabelled EfficientNetB0.

## Issue 8 — Dead scaffolding cell
Deleted a no-op cell that only existed to preserve cell numbering after an earlier draft
was removed. No thesis impact — this was a repo-cleanliness issue, not a results issue.

## Radar chart updated; Transfer Learning Effect chart retired (consequence of Issue 5)
**Thesis sections affected:** Figure 20/28-style radar chart (now shows all four models
instead of three), Figure 29 (retired — see Issue 5 above for the full explanation).

The multi-metric radar chart (`output/radar_chart_metrics.png`) previously excluded
ShuffleNetV2 with the comment "no pretraining, not comparable" — it now includes all four
models, since that's no longer true. The "Transfer Learning Effect" chart
(`transfer_learning_effect.png`) is retired: its notebook cell now prints an explanatory
note instead of generating a chart, since there is no longer a non-pretrained model to
form the "without transfer learning" comparison group. See Issue 5 for what to do about
this in the thesis text.

---

## Bugs found only while actually running the corrected notebook (not in the original review)

These weren't part of the original 8-issue review — they only surfaced once the notebook
was genuinely executed end-to-end for the first time in this environment:

1. **`TSNE(n_iter=...)` → `max_iter=...`** — scikit-learn renamed this parameter; the
   t-SNE feature-space plot (Figure/§3.3.5) would not run at all on a current
   scikit-learn install otherwise.
2. **MobileNetV2's legacy `.h5` save crashed** once the shared `data_augmentation` block
   (Issue 3) was nested inside its Sequential model — Keras 3's HDF5 path tries to
   `deepcopy` the model config, which fails on nested Sequential-in-Sequential models.
   The `.h5` save was dead code anyway (nothing downstream read it; the SavedModel export
   right below it is what TFLite conversion actually uses), so it was removed rather than
   worked around.
3. **A leftover `X_val`/`y_val` reference** in the ShuffleNetV2 resource-summary cell that
   the Issue 2 test-set rename had missed on the first pass — fixed to `X_test`/`y_test`.
4. **Two "Resource Usage Visualisations" cells and the final summary table** still read a
   `'Val Accuracy'` column that Issue 2's fix had renamed to `'Test Accuracy'` — fixed.
5. **`HubFeatureExtractor` (Issue 7) needed `@keras.saving.register_keras_serializable()`**
   for the notebook's own "Strategy 4: Memory-Efficient Keras Evaluation" cell to be able
   to reload `eff_model.keras` from disk — added, confirmed working (both the in-memory
   and file-based evaluation paths now report the same 78.26% test accuracy).

None of these change any reported number — they're the difference between the notebook
crashing partway through and actually producing the numbers in `output/RESULTS.md`.

---

## Scope not covered by this pass (flagging, not fixing)

- **§4.5 / Table 10 / Figures 21–23 — 5-fold cross-validation.** No cross-validation code
  exists anywhere in `solar_panel_fault_detection.ipynb`. These results in the thesis were
  produced by different code than what's in this repository. If you want this section to
  stay, it needs its own cross-validation implementation (wrap each model's build+train in
  a `StratifiedKFold` loop) — not something this pass added, since it wasn't part of the
  original review scope.
- **Dataset size discrepancy:** the thesis abstract says "875 images" (matches the actual
  dataset and this run), but §3.3/§6.1 *Limitations* say "1,902 RGB images". Worth
  reconciling — the code and `kaggle/Faulty_solar_panel/` folder both confirm 875.
- **GPU-based timing (§4.6, Tables 6-8, Figure 24, "Colab GPU" mentions throughout).**
  This session has no GPU; all training/inference timings in `output/RESULTS.md` are
  CPU-only and **not comparable** to GPU figures. If the thesis is meant to report Colab
  GPU numbers specifically, those need a GPU-equipped run — the numbers here are real but
  answer a different question (CPU-only edge-adjacent hardware, arguably closer to the
  thesis's actual Raspberry Pi deployment story than a Colab GPU number would be).

---

## Files changed / added in this repo

- `solar_panel_fault_detection.ipynb` — all fixes above, executed cleanly top-to-bottom,
  zero error outputs remaining.
- `requirements.txt` — added, pins the exact package versions used for this run.
- `pretrained/shufflenetv2_x1_0_imagenet_backbone.keras` — the validated, converted
  pretrained ShuffleNetV2 backbone (5.8 MB), loaded directly by the notebook. No
  `torch`/`torchvision` needed at notebook-run time.
- `scripts/convert_shufflenet_weights.py` — the one-time conversion script that produced
  the file above, with the full validation methodology documented in its docstring. Only
  needed if you want to reproduce or re-derive the conversion; requires
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- `output/` — all figures (PNG), `final_resource_summary.csv`, TFLite models
  (`output/tflite/*.tflite`), saved model weights/SavedModels, `RESULTS.md`.
- `dataset_split/` — the physical 70/15/15 split created by `splitfolders` (train/val/test
  folders of images), regenerable from `kaggle/Faulty_solar_panel/` at any time.

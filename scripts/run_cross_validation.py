"""
5-fold stratified cross-validation for all four models (MobileNetV2,
EfficientNet-Lite0, ShuffleNetV2, VGG16), using the exact same architectures,
augmentation, class-weighting, and training recipe as
solar_panel_fault_detection.ipynb — just wrapped in a StratifiedKFold loop
instead of a single fixed 70/15/15 split.

Why this exists: the single-run notebook reports one number per model. This
answers "how stable is that number across different train/test splits?" —
mean +/- std accuracy/F1 across 5 folds, each model rebuilt and retrained from
scratch per fold (K-fold CV is inherently redundant retraining, so none of
run_notebook.py's fit-caching applies here — each fold really does retrain).

Frozen pretrained backbones (MobileNetV2/VGG16/EfficientNet-Lite0-hub/
ShuffleNetV2) are each loaded ONCE and reused across all 5 folds — they never
change (trainable=False), so there's no correctness reason to reload them,
and it avoids repeating the ~9-minute one-time TensorFlow graph-tracing
overhead ShuffleNetV2's fragmented architecture incurs on this CPU-only
machine (see CHANGES.md / RESULTS.md for background on that overhead).

Resumable: each (model, fold) result is cached to output_cv/cache/ as it
completes, so a crash or interruption doesn't lose completed folds.

Output goes to output_cv/ — deliberately separate from output/ (the
single-run notebook's results), per instruction not to touch/overwrite that.

Run: python scripts/run_cross_validation.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import gc
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEED = 100
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = pathlib.Path("kaggle/Faulty_solar_panel")
OUT_DIR = pathlib.Path("output_cv")
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
EPOCHS = 25
BATCH_SIZE = 32

log_f = open(OUT_DIR / "run.log", "a", encoding="utf-8")


def log(msg):
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()


# ---------------------------------------------------------------------------
# Dataset: full path/label list (not the fixed dataset_split/ folders — CV
# needs to redraw the train/test boundary itself, fold by fold).
# ---------------------------------------------------------------------------
class_names = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
log(f"Classes: {class_names}")
CLEAN_IDX = class_names.index("Clean")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

paths, labels = [], []
for idx, cls in enumerate(class_names):
    for p in sorted((DATA_DIR / cls).glob("*.*")):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
            labels.append(idx)
paths = np.array(paths)
labels = np.array(labels)
log(f"Total images: {len(paths)}")


def make_dataset(idx, training):
    p = paths[idx]
    y = labels[idx]
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if training:
        ds = ds.shuffle(buffer_size=len(p), seed=SEED)

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [224, 224])
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Shared augmentation (identical to notebook cell 7) + one frozen backbone per
# model type, built once and reused across every fold.
# ---------------------------------------------------------------------------
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(20 / 360),
    keras.layers.RandomZoom((-0.2, 0.2)),
    keras.layers.RandomBrightness(0.2),
    keras.layers.GaussianNoise(0.05),
], name="augmentation")


@keras.saving.register_keras_serializable()
class HubFeatureExtractor(keras.layers.Layer):
    def __init__(self, handle, **kwargs):
        super().__init__(**kwargs)
        self.handle = handle
        self.hub_layer = hub.KerasLayer(handle, trainable=False)

    def call(self, x):
        return self.hub_layer(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["handle"] = self.handle
        return cfg


@keras.saving.register_keras_serializable()
class ChannelShuffle(keras.layers.Layer):
    def __init__(self, groups=2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, x):
        s = tf.shape(x)
        t = tf.reshape(x, [-1, s[1], s[2], self.groups, s[3] // self.groups])
        t = tf.transpose(t, perm=[0, 1, 2, 4, 3])
        return tf.reshape(t, [-1, s[1], s[2], s[3]])

    def get_config(self):
        cfg = super().get_config()
        cfg["groups"] = self.groups
        return cfg


@keras.saving.register_keras_serializable()
class ChannelSplit(keras.layers.Layer):
    def __init__(self, split="first", **kwargs):
        super().__init__(**kwargs)
        self.split = split

    def call(self, x):
        c = tf.shape(x)[-1] // 2
        return x[:, :, :, :c] if self.split == "first" else x[:, :, :, c:]

    def get_config(self):
        cfg = super().get_config()
        cfg["split"] = self.split
        return cfg


log("Loading frozen backbones (once, shared across all folds)...")
mobilenet_base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
mobilenet_base.trainable = False

eff_base = HubFeatureExtractor("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2")

shuffle_backbone = keras.models.load_model("pretrained/shufflenetv2_x1_0_imagenet_backbone.keras")
shuffle_backbone.trainable = False

vgg_base = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
vgg_base.trainable = False
log("Backbones loaded.")


def build_mobilenet_model():
    return keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        data_augmentation,
        keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        mobilenet_base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(6, activation="softmax"),
    ])


def build_efficientnet_model():
    m = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        data_augmentation,
        keras.layers.Rescaling(scale=1.0 / 255),
        eff_base,
        keras.layers.Dropout(0.5),
        keras.layers.Dense(6, activation="softmax"),
    ])
    m.build((None, 224, 224, 3))
    return m


def build_shufflenet_model():
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = keras.layers.Rescaling(scale=1.0 / 255)(x)
    x = keras.layers.Normalization(mean=[0.485, 0.456, 0.406],
                                    variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(x)
    x = shuffle_backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(6, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_vgg_model():
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = vgg_base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(6, activation="softmax")(x)
    return keras.Model(inputs, outputs)


MODEL_BUILDERS = {
    "MobileNetV2": build_mobilenet_model,
    "EfficientNet-Lite": build_efficientnet_model,
    "ShuffleNetV2": build_shufflenet_model,
    "VGG16": build_vgg_model,
}

# ---------------------------------------------------------------------------
# 5-fold stratified split (fixed seed -> reproducible fold assignment).
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(paths, labels))
log(f"Prepared {N_SPLITS} stratified folds.")

results = []
for model_name, builder in MODEL_BUILDERS.items():
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        cache_path = CACHE_DIR / f"{model_name}_fold{fold_idx}.json"
        if cache_path.exists():
            entry = json.loads(cache_path.read_text())
            log(f"[cache] {model_name} fold {fold_idx}: acc={entry['accuracy']:.4f} f1={entry['f1_macro']:.4f} (skipped)")
            results.append(entry)
            continue

        log(f"\n{'=' * 70}\n{model_name} — fold {fold_idx + 1}/{N_SPLITS}\n{'=' * 70}")

        train_inner_idx, val_inner_idx = train_test_split(
            train_idx, test_size=0.15, stratify=labels[train_idx], random_state=SEED
        )

        train_ds = make_dataset(train_inner_idx, training=True)
        val_ds = make_dataset(val_inner_idx, training=False)
        test_ds = make_dataset(test_idx, training=False)

        cw = dict(enumerate(compute_class_weight(
            "balanced", classes=np.unique(labels[train_inner_idx]), y=labels[train_inner_idx]
        )))

        model = builder()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        t0 = time.perf_counter()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                          class_weight=cw, callbacks=[early_stop], verbose=2)
        train_time = time.perf_counter() - t0

        y_true, y_pred = [], []
        for xb, yb in test_ds:
            preds = model.predict(xb, verbose=0)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(np.argmax(preds, axis=1).tolist())

        # Fault-detection recall: of the genuinely faulty test images (true
        # label != Clean), what fraction did the model NOT wave through as
        # "Clean"? This is a binary catch-rate that macro precision/recall/F1
        # don't capture (they weight "wrong fault type" the same as "missed
        # entirely") — see output/RESULTS.md section 2a for the single-run
        # version of this metric and why it matters for this problem.
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        faulty_mask = np.array(y_true) != CLEAN_IDX
        n_faulty = int(faulty_mask.sum())
        n_missed = int(sum(1 for t, p in zip(y_true, y_pred) if t != CLEAN_IDX and p == CLEAN_IDX))
        fault_detection_recall = (n_faulty - n_missed) / n_faulty if n_faulty else None

        entry = {
            "model": model_name,
            "fold": fold_idx,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "epochs_run": len(hist.history["loss"]),
            "train_time_s": round(train_time, 1),
            "n_train": int(len(train_inner_idx)),
            "n_val": int(len(val_inner_idx)),
            "n_test": int(len(test_idx)),
            "n_faulty": n_faulty,
            "n_faulty_missed_as_clean": n_missed,
            "fault_detection_recall": fault_detection_recall,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }
        cache_path.write_text(json.dumps(entry))
        results.append(entry)
        log(f"{model_name} fold {fold_idx}: acc={entry['accuracy']:.4f} f1={entry['f1_macro']:.4f} "
            f"fault_recall={fault_detection_recall:.4f} ({train_time:.1f}s, {entry['epochs_run']} epochs)")

        del model, train_ds, val_ds, test_ds
        gc.collect()

# ---------------------------------------------------------------------------
# Aggregate + save
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "cv_per_fold_results.csv", index=False)

summary = df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
    mean_f1=("f1_macro", "mean"), std_f1=("f1_macro", "std"),
    mean_precision=("precision_macro", "mean"), std_precision=("precision_macro", "std"),
    mean_recall=("recall_macro", "mean"), std_recall=("recall_macro", "std"),
    mean_train_time_s=("train_time_s", "mean"),
).reset_index()
summary.to_csv(OUT_DIR / "cv_summary.csv", index=False)

log("\n\n" + "=" * 70)
log("5-FOLD CROSS-VALIDATION SUMMARY")
log("=" * 70)
log(summary.to_string(index=False))
log("\nALL FOLDS COMPLETE.")

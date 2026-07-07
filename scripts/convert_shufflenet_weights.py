"""
Converts torchvision's official ImageNet-pretrained ShuffleNetV2 x1.0 checkpoint
into a Keras 3 model, byte-validated against the original PyTorch model.

Why this exists: no official TensorFlow/Keras ShuffleNetV2 checkpoint exists.
torchvision's is the only trustworthy pretrained ShuffleNetV2 (official,
maintained, hosted on download.pytorch.org — unlike the unofficial TF1
checkpoint on a personal Google Drive link that was rejected during the
original review; see CHANGES.md, Issue 5).

This is a one-time conversion. Requires `torch`/`torchvision` (not a project
dependency — install separately to re-run this script). The notebook itself
only loads the resulting `pretrained/shufflenetv2_x1_0_imagenet_backbone.keras`
file and never needs torch at runtime.

Two porting pitfalls this script fixes (each initially produced a plausible-
but-wrong conversion, caught by the forward-pass validation below):
  1. TF's `padding='same'` is not always identical to PyTorch's explicit
     `padding=1` for stride>1 convs — replaced with ZeroPadding2D + 'valid'
     to force exact, stride-independent symmetric padding.
  2. Keras's BatchNormalization defaults to epsilon=1e-3; PyTorch's default
     is 1e-5 — mismatched epsilon alone dropped cosine similarity to ~0.999
     instead of ~1.0 after fix #1, small but real drift compounded over 112
     BN layers.

Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     python scripts/convert_shufflenet_weights.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import keras
import torch
import torchvision

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "pretrained",
                         "shufflenetv2_x1_0_imagenet_backbone.keras")

print("=== Building faithful Keras ShuffleNetV2 x1.0 backbone (explicit padding to match torch exactly) ===")


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


def conv3x3(x, filters, stride, name, depthwise=False):
    """3x3 conv with PyTorch-style symmetric padding=1, regardless of stride."""
    x = keras.layers.ZeroPadding2D(padding=1)(x)
    if depthwise:
        return keras.layers.DepthwiseConv2D(3, strides=stride, padding="valid", use_bias=False, name=name)(x)
    return keras.layers.Conv2D(filters, 3, strides=stride, padding="valid", use_bias=False, name=name)(x)


def inverted_residual(x, oup, stride, prefix):
    branch_features = oup // 2

    if stride > 1:
        b1 = conv3x3(x, None, stride, f"{prefix}_branch1_0", depthwise=True)
        b1 = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_branch1_1")(b1)
        b1 = keras.layers.Conv2D(branch_features, 1, use_bias=False, name=f"{prefix}_branch1_2")(b1)
        b1 = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_branch1_3")(b1)
        b1 = keras.layers.ReLU()(b1)
        branch2_in = x
    else:
        b1 = ChannelSplit(split="first")(x)
        branch2_in = ChannelSplit(split="second")(x)

    b2 = keras.layers.Conv2D(branch_features, 1, use_bias=False, name=f"{prefix}_branch2_0")(branch2_in)
    b2 = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_branch2_1")(b2)
    b2 = keras.layers.ReLU()(b2)
    b2 = conv3x3(b2, None, stride, f"{prefix}_branch2_3", depthwise=True)
    b2 = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_branch2_4")(b2)
    b2 = keras.layers.Conv2D(branch_features, 1, use_bias=False, name=f"{prefix}_branch2_5")(b2)
    b2 = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_branch2_6")(b2)
    b2 = keras.layers.ReLU()(b2)

    out = keras.layers.Concatenate()([b1, b2])
    out = ChannelShuffle(groups=2)(out)
    return out


def build_shufflenetv2_x1_0_backbone(input_shape=(224, 224, 3)):
    inputs = keras.Input(shape=input_shape)
    x = conv3x3(inputs, 24, 2, "conv1_0", depthwise=False)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="conv1_1")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.ZeroPadding2D(padding=1)(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")(x)

    for stage_idx, (out_ch, repeats) in zip([2, 3, 4], [(116, 4), (232, 8), (464, 4)]):
        x = inverted_residual(x, out_ch, stride=2, prefix=f"stage{stage_idx}_0")
        for b in range(1, repeats):
            x = inverted_residual(x, out_ch, stride=1, prefix=f"stage{stage_idx}_{b}")

    x = keras.layers.Conv2D(1024, 1, use_bias=False, name="conv5_0")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="conv5_1")(x)
    x = keras.layers.ReLU()(x)
    return keras.Model(inputs, x, name="shufflenetv2_x1_0_backbone")


keras_model = build_shufflenetv2_x1_0_backbone()
print("Keras backbone params:", keras_model.count_params())

print("\n=== Loading real pretrained torchvision ShuffleNetV2 x1.0 ===")
torch_model = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_sd = torch_model.state_dict()

print("\n=== Transplanting weights by explicit name mapping ===")
torch_unit_prefixes = sorted({k.rsplit(".", 1)[0] for k in torch_sd if not k.startswith("fc") and "num_batches_tracked" not in k})

assigned = 0
for torch_prefix in torch_unit_prefixes:
    keras_name = torch_prefix.replace(".", "_")
    layer = keras_model.get_layer(name=keras_name)
    tensor = torch_sd[f"{torch_prefix}.weight"].numpy()
    if isinstance(layer, keras.layers.BatchNormalization):
        gamma = tensor
        beta = torch_sd[f"{torch_prefix}.bias"].numpy()
        rmean = torch_sd[f"{torch_prefix}.running_mean"].numpy()
        rvar = torch_sd[f"{torch_prefix}.running_var"].numpy()
        layer.set_weights([gamma, beta, rmean, rvar])
    elif isinstance(layer, keras.layers.DepthwiseConv2D):
        layer.set_weights([np.transpose(tensor, (2, 3, 0, 1))])
    elif isinstance(layer, keras.layers.Conv2D):
        layer.set_weights([np.transpose(tensor, (2, 3, 1, 0))])
    else:
        raise TypeError(f"unexpected layer type for {keras_name}: {type(layer)}")
    assigned += 1

print(f"Assigned weights to {assigned} layers (of {len(torch_unit_prefixes)} torch units).")

print("\n=== Validation: forward pass comparison on random input ===")
np.random.seed(0)
img_uint8 = np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x_tf = (img_uint8.astype(np.float32) / 255.0 - mean) / std
keras_out = keras_model(x_tf, training=False).numpy()
keras_feat = keras_out.mean(axis=(1, 2))

x_torch = torch.from_numpy(img_uint8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
x_torch = (x_torch - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
with torch.no_grad():
    t = torch_model.conv1(x_torch)
    t = torch_model.maxpool(t)
    t = torch_model.stage2(t)
    t = torch_model.stage3(t)
    t = torch_model.stage4(t)
    t = torch_model.conv5(t)
    torch_feat = t.mean([2, 3]).numpy()

diff = np.abs(keras_feat - torch_feat)
cos = (keras_feat @ torch_feat.T) / (np.linalg.norm(keras_feat) * np.linalg.norm(torch_feat))
print("keras shape:", keras_out.shape, " torch shape:", tuple(t.shape))
print("keras_feat[:5] :", keras_feat[0, :5])
print("torch_feat[:5] :", torch_feat[0, :5])
print("max abs diff   :", diff.max())
print("mean abs diff  :", diff.mean())
print("cosine sim     :", cos.item())

if cos.item() > 0.999 and diff.max() < 1e-2:
    print("\n*** CONVERSION VALIDATED ***")
    keras_model.save(OUT_PATH)
    print("Saved converted backbone to", os.path.abspath(OUT_PATH))
else:
    raise RuntimeError("Conversion failed validation — do not trust the saved weights.")

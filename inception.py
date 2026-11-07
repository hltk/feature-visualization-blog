"""Original Lucid InceptionV1, truncated at mixed5a. RGB images enter in [0, 1].

Architecture and weights come from the original frozen InceptionV1 graph:
https://storage.googleapis.com/modelzoo/vision/other_models/InceptionV1.pb
The forward pass uses ordinary ReLU; Lucid's startup gradient warm-up is optional.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from relu import ReLUWarmup

CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/inception_v1_mixed5a.npz"


def same_pad(x, kernel, stride=1, value=0):
    """SAME padding: any extra pixel goes on the bottom/right."""
    h, w = x.shape[-2:]
    ph = max((h + stride - 1) // stride * stride - stride + kernel - h, 0)
    pw = max((w + stride - 1) // stride * stride - stride + kernel - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=value)


def pool(x, stride=1):
    return F.max_pool2d(same_pad(x, 3, stride, float("-inf")), 3, stride)


def normalize(x):
    # A channel radius of 5 sums 11 channels; PyTorch divides alpha by size.
    return F.local_response_norm(x, 11, alpha=11e-4, beta=0.5, k=2)


class InceptionV1(nn.Module):
    blocks = ("mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a")

    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Model weights missing: {checkpoint}")
        self.convs = nn.ModuleDict()
        self.relu_warmup = False
        with np.load(checkpoint) as weights:
            for name in weights.files:
                if not name.endswith("_w") or not name.startswith(
                    ("conv2d", *self.blocks)
                ):
                    continue
                weight = torch.from_numpy(weights[name].copy()).permute(3, 2, 0, 1)
                out_channels, in_channels, k, _ = weight.shape
                conv = nn.Conv2d(in_channels, out_channels, k)
                conv.weight.data.copy_(weight)
                conv.bias.data.copy_(torch.from_numpy(weights[name[:-2] + "_b"].copy()))
                self.convs[name[:-2]] = conv
        self.eval().requires_grad_(False)

    def conv(self, x, name, stride=1):
        layer = self.convs[name]
        pre = F.conv2d(same_pad(x, layer.kernel_size[0], stride),
                       layer.weight, layer.bias, stride=stride)
        return ReLUWarmup.apply(pre) if self.relu_warmup else F.relu(pre)

    def forward(self, image):
        """Return named activations, each shaped [batch, channels, height, width]."""
        layers = {}
        x = self.conv(image * 255 - 117, "conv2d0", stride=2)
        layers["conv2d0"] = x
        x = normalize(pool(x, 2))
        layers["localresponsenorm0"] = x
        x = self.conv(self.conv(x, "conv2d1"), "conv2d2")
        x = normalize(x)
        layers["localresponsenorm1"] = x
        x = pool(x, 2)
        for name in self.blocks:
            if name in ("mixed4a", "mixed5a"):
                x = pool(x, 2)
            x = torch.cat([
                self.conv(x, name + "_1x1"),
                self.conv(self.conv(x, name + "_3x3_bottleneck"), name + "_3x3"),
                self.conv(self.conv(x, name + "_5x5_bottleneck"), name + "_5x5"),
                self.conv(pool(x), name + "_pool_reduce"),
            ], dim=1)
            layers[name] = x
        return layers

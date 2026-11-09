"""2023: MACO with the 2017 views and Adam schedule used in the blog."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from shared.cli import run
from shared.common import ASSETS, SIZE, maximize
from shared.fourier import color_mix
from shared.views import distill_2017_view


def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    views = torch.Generator().manual_seed(1001)
    shape = (1, 3, SIZE, SIZE // 2 + 1)
    phase = ((2 * torch.rand(shape) - 1) * math.pi).requires_grad_()
    magnitude = torch.from_numpy(np.load(ASSETS / "maco_imagenet.npy"))[None]
    magnitude = F.interpolate(magnitude, shape[-2:], mode="bilinear", align_corners=False)

    def image():
        # r exp(i phi), with fixed ImageNet magnitudes and trainable phases.
        spectrum = torch.polar(magnitude, phase)
        rgb = torch.fft.irfft2(spectrum, s=(SIZE, SIZE))
        rgb = (rgb - rgb.mean()) / (rgb.std(correction=0) + 1e-5)
        return (2 * color_mix(rgb)).sigmoid()

    def score(_step):
        view = distill_2017_view(image().to(device), views)
        return model(view)[layer][:, channel].mean()

    maximize(model, [phase], score, steps, .05)
    with torch.no_grad():
        return image()


if __name__ == "__main__":
    run(render, __doc__, "maco_2023", steps=2560)

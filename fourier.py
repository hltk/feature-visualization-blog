import math

import numpy as np
import torch
from torch.nn import functional as F

from common import ASSETS, SIZE, maximize
from views import distill_2017_view

# Empirical Lucid color matrix. See LICENSES/Lucid.txt.
COLOR = torch.tensor([[.26, .09, .02], [.27, 0., -.05], [.27, -.09, .03]])
COLOR /= torch.linalg.vector_norm(COLOR, dim=0).max()


def color_mix(image):
    return torch.einsum("ij,bjhw->bihw", COLOR, image)


def activation(features, channel, objective):
    if objective == "layer":
        return features.square().mean()
    feature = features[:, channel]
    height, width = feature.shape[-2:]
    if objective == "neuron":
        return feature[:, height // 2, width // 2].mean()
    if objective == "centered":
        y = (torch.arange(height, device=feature.device) - (height - 1) / 2) / height
        x = (torch.arange(width, device=feature.device) - (width - 1) / 2) / width
        weights = (-(y[:, None].square() + x.square()) / (2 * .2 ** 2)).exp()
        return (feature * (weights / weights.sum())).sum()
    return feature.mean()


def render(model, layer, channel, method, steps, *, objective="channel", strength=1.5):
    device = next(model.parameters()).device
    views = torch.Generator().manual_seed(1001)
    shape = (1, 3, SIZE, SIZE // 2 + 1)
    centered = method == "centered-2026"

    # Keep the FFT on CPU for consistent support across accelerators.
    if method == "maco":
        parameter = ((2 * torch.rand(shape) - 1) * math.pi).requires_grad_()
        magnitude = torch.from_numpy(np.load(ASSETS / "maco_imagenet.npy"))[None]
        magnitude = F.interpolate(magnitude, shape[-2:], mode="bilinear", align_corners=False)

        def logits(_step):
            spectrum = torch.polar(magnitude, parameter)
            image = torch.fft.irfft2(spectrum, s=(SIZE, SIZE))
            image = (image - image.mean()) / (image.std(correction=0) + 1e-5)
            return 2 * color_mix(image)
    else:
        parameter = (.01 * torch.randn(*shape, 2)).requires_grad_()
        initial = parameter.detach().clone()
        radius = (torch.fft.fftfreq(SIZE)[:, None].square()
                  + torch.fft.rfftfreq(SIZE).square()).sqrt()
        scale = radius.clamp_min(1 / SIZE).reciprocal()

        def logits(step):
            progress = step / max(steps - 1, 1)
            coefficients = parameter
            if centered and progress < .6:
                cutoff = .05 + (float(radius.max()) - .05) * progress / .6
                edge = ((radius - cutoff) / .02).clamp(0, 1)
                mask = (1 + (math.pi * edge).cos()) / 2
                coefficients = initial + (parameter - initial) * mask[..., None]
            spectrum = torch.view_as_complex(coefficients) * scale
            return color_mix(torch.fft.irfft2(spectrum, s=(SIZE, SIZE), norm="ortho") / 4)

    prior = torch.tensor([.72, .75, .78]).logit()[None, :, None, None]

    def score(step):
        rgb_logits = logits(step)
        view = distill_2017_view(rgb_logits.sigmoid().to(device), views)
        value = activation(model(view)[layer], channel, objective)
        if centered:
            penalty = (rgb_logits - prior).square().mean()
            value = strength * value - 20 * penalty.to(device)
        return value

    def rate(step):
        progress = step / max(steps - 1, 1)
        finish = max((progress - .6) / .4, 0)
        return .005 + .045 * (1 + math.cos(math.pi * finish)) / 2

    maximize(model, [parameter], score, steps, rate if centered else .05)
    with torch.no_grad():
        return logits(steps).sigmoid()

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from shared.common import ASSETS


def resize(image, size):
    return F.interpolate(image, (size, size), mode="bilinear",
                         align_corners=False, antialias=True)


def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    with Image.open(ASSETS / "clouds.png") as photo:
        pixels = np.array(photo.convert("RGB"))
    original = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device).float() / 255

    rng = torch.Generator().manual_seed(1)
    detail = torch.zeros_like(resize(original, 160))
    for size in (160, 224, 320):
        base = resize(original, size)
        image = (base + resize(detail, size)).clamp(0, 1)
        for _ in range(steps):
            dy, dx = torch.randint(-16, 17, (2,), generator=rng).tolist()
            view = image.roll((dy, dx), dims=(-2, -1)).requires_grad_()
            energy = model(view)[layer][:, channel].square().mean()
            gradient, = torch.autograd.grad(energy, view)
            update = .018 * gradient / gradient.abs().mean().clamp_min(1e-12)
            image = (view.detach() + update).clamp(0, 1).roll((-dy, -dx), dims=(-2, -1))
        detail = image - base
    return image

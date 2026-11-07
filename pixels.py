import torch
from torch.nn import functional as F

from common import SIZE, maximize

GRAY = 117 / 255


def total_variation(image):
    corner = image[..., :-1, :-1]
    dx = image[..., :-1, 1:] - corner
    dy = image[..., 1:, :-1] - corner
    return (dx.square() + dy.square()).mean()


def gaussian_blur(image):
    axis = torch.arange(-4, 5, device=image.device, dtype=image.dtype)
    kernel = (-axis.square() / 2).exp()
    kernel /= kernel.sum()
    weights = torch.outer(kernel, kernel).expand(3, 1, 9, 9)
    return F.conv2d(F.pad(image, (4,) * 4, mode="reflect"), weights, groups=3)


def resize(image, size):
    return F.interpolate(image, (size, size), mode="bilinear", align_corners=False)


def render(model, layer, channel, method, steps):
    device = next(model.parameters()).device
    l2 = method in ("l2-pixels", "l2-gaussian-blur")
    jitter = method == "jitter-multiscale"
    base = torch.full((1, 3, SIZE, SIZE), GRAY) if l2 else torch.rand(1, 3, SIZE, SIZE)
    base = base.to(device)
    detail = torch.zeros_like(base)
    octaves = min(steps, 4) if jitter else 1
    shifts = torch.Generator().manual_seed(1001)

    for octave in range(octaves):
        size = round(SIZE / 1.4 ** (octaves - octave - 1))
        background = resize(base, size)
        image = (background + resize(detail, size)).clamp(0, 1).requires_grad_()

        def score(_step):
            view = image
            if jitter:
                shift = torch.randint(-32, 33, (2,), generator=shifts).tolist()
                view = image.roll(tuple(shift), dims=(-2, -1))
            value = model(view)[layer][:, channel].mean()
            if l2:
                value = value - 1000 * (image - GRAY).square().mean()
            if method == "tv-pixels":
                value = value - 1000 * total_variation(image)
            return value

        def project(step):
            image.clamp_(0, 1)
            if method == "l2-gaussian-blur" and (step + 1) % 4 == 0:
                image.lerp_(gaussian_blur(image), 0.5)

        updates = steps // octaves + (octave < steps % octaves)
        maximize(model, [image], score, updates, .05, project=project, warmup=octave == 0)
        detail = image.detach() - background

    return image.detach()

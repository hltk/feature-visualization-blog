"""2015: Random Shifts."""
import torch
from torch.nn import functional as F

from shared.cli import run
from shared.common import SIZE, maximize


def resize(image, size):
    return F.interpolate(image, (size, size), mode="bilinear", align_corners=False)



def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    base = torch.rand(1, 3, SIZE, SIZE).to(device)
    detail = torch.zeros_like(base)
    octaves = min(steps, 4)
    shifts = torch.Generator().manual_seed(1001)

    for octave in range(octaves):
        size = round(SIZE / 1.4 ** (octaves - octave - 1))
        background = resize(base, size)
        image = (background + resize(detail, size)).clamp(0, 1).requires_grad_()

        def score(_step):
            shift = torch.randint(-32, 33, (2,), generator=shifts).tolist()
            view = image.roll(tuple(shift), dims=(-2, -1))
            return model(view)[layer][:, channel].mean()

        def project(_step):
            image.clamp_(0, 1)

        updates = steps // octaves + (octave < steps % octaves)
        maximize(model, [image], score, updates, .05, project=project, warmup=octave == 0)
        # Carry the optimized detail into the next resolution.
        detail = image.detach() - background

    return image.detach()


if __name__ == "__main__":
    run(render, __doc__, "random_shifts_2015", steps=2560, target="mixed4d:475")

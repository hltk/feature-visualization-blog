"""2015: Gaussian Blur."""
import torch
from torch.nn import functional as F

from shared.cli import run
from shared.common import SIZE, maximize


GRAY = 117 / 255


def gaussian_blur(image):
    axis = torch.arange(-4, 5, device=image.device, dtype=image.dtype)
    kernel = (-axis.square() / 2).exp()
    kernel /= kernel.sum()
    weights = torch.outer(kernel, kernel).expand(3, 1, 9, 9)
    return F.conv2d(F.pad(image, (4,) * 4, mode="reflect"), weights, groups=3)

def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    image = torch.full((1, 3, SIZE, SIZE), GRAY).to(device).requires_grad_()

    def score(_step):
        activation = model(image)[layer][:, channel].mean()
        return activation - 1000 * (image - GRAY).square().mean()

    def project(step):
        image.clamp_(0, 1)
        if (step + 1) % 4 == 0:
            image.lerp_(gaussian_blur(image), 0.5)

    maximize(model, [image], score, steps, .05, project=project)
    return image.detach()


if __name__ == "__main__":
    run(render, __doc__, "gaussian_blur_2015", steps=2560, target="mixed4d:475")

"""2009: First Images."""
import torch

from shared.cli import run
from shared.common import SIZE, maximize


def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    image = torch.rand(1, 3, SIZE, SIZE).to(device).requires_grad_()

    def score(_step):
        activation = model(image)[layer][:, channel].mean()
        return activation

    def project(_step):
        image.clamp_(0, 1)

    maximize(model, [image], score, steps, .05, project=project)
    return image.detach()


if __name__ == "__main__":
    run(render, __doc__, "first_images_2009", steps=2560, target="mixed4d:475")

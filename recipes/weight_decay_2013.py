"""2013: Weight Decay."""
import torch

from shared.cli import run
from shared.common import SIZE, maximize


GRAY = 117 / 255

def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    image = torch.full((1, 3, SIZE, SIZE), GRAY).to(device).requires_grad_()

    def score(_step):
        activation = model(image)[layer][:, channel].mean()
        return activation - 1000 * (image - GRAY).square().mean()

    def project(_step):
        image.clamp_(0, 1)

    maximize(model, [image], score, steps, .05, project=project)
    return image.detach()


if __name__ == "__main__":
    run(render, __doc__, "weight_decay_2013", steps=2560, target="mixed4d:475")

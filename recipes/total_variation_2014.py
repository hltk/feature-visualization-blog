"""2014: Total Variation."""
import torch

from shared.cli import run
from shared.common import SIZE, maximize


def total_variation(image):
    # TV with beta = 2, averaged over neighboring RGB differences.
    corner = image[..., :-1, :-1]
    dx = image[..., :-1, 1:] - corner
    dy = image[..., 1:, :-1] - corner
    return (dx.square() + dy.square()).mean()

def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    image = torch.rand(1, 3, SIZE, SIZE).to(device).requires_grad_()

    def score(_step):
        activation = model(image)[layer][:, channel].mean()
        return activation - 1000 * total_variation(image)

    def project(_step):
        image.clamp_(0, 1)

    maximize(model, [image], score, steps, .05, project=project)
    return image.detach()


if __name__ == "__main__":
    run(render, __doc__, "total_variation_2014", steps=2560, target="mixed4d:475")

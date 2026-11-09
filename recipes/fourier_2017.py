"""2017: Fourier."""
import torch

from shared.cli import run
from shared.common import SIZE, maximize
from shared.fourier import fourier_logits, frequency_radius
from shared.views import distill_2017_view


def render(model, layer, channel, steps):
    device = next(model.parameters()).device
    views = torch.Generator().manual_seed(1001)
    # Keep the inverse FFT on CPU for consistent accelerator support.
    coefficients = (.01 * torch.randn(1, 3, SIZE, SIZE // 2 + 1, 2)).requires_grad_()
    scale = frequency_radius().clamp_min(1 / SIZE).reciprocal()

    def image():
        return fourier_logits(coefficients, scale).sigmoid()

    def score(_step):
        view = distill_2017_view(image().to(device), views)
        return model(view)[layer][:, channel].mean()

    maximize(model, [coefficients], score, steps, .05)
    with torch.no_grad():
        return image()


if __name__ == "__main__":
    run(render, __doc__, "fourier_2017", steps=2560)

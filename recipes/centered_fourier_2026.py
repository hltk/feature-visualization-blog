"""2026 Gaussian-centered Fourier and the Different Objectives examples."""
import math

import torch

from shared.cli import load_model, parse, parser_for
from shared.common import SIZE, maximize, save
from shared.fourier import fourier_logits, frequency_radius
from shared.views import distill_2017_view


def activation(features, channel, objective):
    # The blog's layer, neuron, and channel objectives reduce A[batch, c, u, v].
    if objective == "layer":
        return features.square().mean()
    feature = features[:, channel]
    height, width = feature.shape[-2:]
    if objective == "neuron":
        return feature[:, height // 2, width // 2].mean()
    if objective == "centered":
        # Replace the uniform spatial average with a Gaussian around the center.
        y = (torch.arange(height, device=feature.device) - (height - 1) / 2) / height
        x = (torch.arange(width, device=feature.device) - (width - 1) / 2) / width
        weights = (-(y[:, None].square() + x.square()) / (2 * .2 ** 2)).exp()
        return (feature * (weights / weights.sum())).sum()
    return feature.mean()


def render(model, layer, channel, steps, *, objective="centered", strength=1.5):
    device = next(model.parameters()).device
    views = torch.Generator().manual_seed(1001)
    parameter = (.01 * torch.randn(1, 3, SIZE, SIZE // 2 + 1, 2)).requires_grad_()
    initial = parameter.detach().clone()
    radius = frequency_radius()
    scale = radius.clamp_min(1 / SIZE).reciprocal()
    max_radius = float(radius.max())
    prior = torch.tensor([.72, .75, .78]).logit()[None, :, None, None]

    def logits(step):
        progress = step / max(steps - 1, 1)
        if progress >= .6:
            return fourier_logits(parameter, scale)
        # Open higher frequencies to updates as optimization progresses.
        cutoff = .05 + (max_radius - .05) * progress / .6
        edge = ((radius - cutoff) / .02).clamp(0, 1)
        mask = (1 + (math.pi * edge).cos()) / 2
        coefficients = initial + (parameter - initial) * mask[..., None]
        return fourier_logits(coefficients, scale)

    def score(step):
        rgb_logits = logits(step)
        view = distill_2017_view(rgb_logits.sigmoid().to(device), views)
        value = activation(model(view)[layer], channel, objective)
        penalty = (rgb_logits - prior).square().mean()
        return strength * value - 20 * penalty.to(device)

    def rate(step):
        progress = step / max(steps - 1, 1)
        finish = max((progress - .6) / .4, 0)
        return .005 + .045 * (1 + math.cos(math.pi * finish)) / 2

    maximize(model, [parameter], score, steps, rate)
    with torch.no_grad():
        return logits(steps).sigmoid()


def main():
    parser = parser_for(__doc__, "centered_fourier_2026", steps=2560)
    parser.add_argument("--objective", choices=("centered", "neuron", "channel", "layer"),
                        default="centered")
    parser.add_argument("--strength", type=float, default=1.5)
    args = parse(parser)
    model = load_model(args)
    torch.manual_seed(1)
    image = render(model, args.layer, args.channel, args.steps,
                   objective=args.objective, strength=args.strength)
    name = args.layer if args.objective == "layer" else f"{args.layer}-{args.channel}"
    save(image, args.output / f"{args.objective}-{name}.png")


if __name__ == "__main__":
    main()

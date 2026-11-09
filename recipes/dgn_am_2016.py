"""The published 2016 fc6 upconvolutional image prior, implemented in PyTorch.

Architecture: Evolving-AI-Lab/synthesizing, nets/upconv/fc6/generator.prototxt.
Weights: Dosovitskiy & Brox's release_deepsim_v0, stored in ignored data/dgn/.
"""
import numpy as np
import torch
from torch import nn

from shared.cli import load_model, parse, parser_for
from shared.common import ASSETS, save

ROOT = ASSETS.parent


class DGN(nn.Module):
    def __init__(self, weights=ROOT / 'data/dgn/fc6.npz'):
        super().__init__()
        layers = {}
        for name in ('defc7', 'defc6', 'defc5'):
            layers[name] = nn.Linear(4096, 4096)
            layers['relu_' + name] = nn.LeakyReLU(.3)
        layers['reshape'] = nn.Unflatten(1, (256, 4, 4))
        for name, inputs, outputs, kernel, stride in (
            ('deconv5', 256, 256, 4, 2), ('conv5_1', 256, 512, 3, 1),
            ('deconv4', 512, 256, 4, 2), ('conv4_1', 256, 256, 3, 1),
            ('deconv3', 256, 128, 4, 2), ('conv3_1', 128, 128, 3, 1),
            ('deconv2', 128, 64, 4, 2), ('deconv1', 64, 32, 4, 2),
            ('deconv0', 32, 3, 4, 2),
        ):
            layers[name] = nn.ConvTranspose2d(inputs, outputs, kernel, stride, padding=1)
            if name != 'deconv0':
                layers['relu_' + name] = nn.LeakyReLU(.3)
        self.layers = nn.ModuleDict(layers)
        with np.load(weights, allow_pickle=False) as saved:
            expected = {f'{name}.{parameter}' for name, layer in self.layers.items()
                        for parameter, _ in layer.named_parameters()}
            assert set(saved.files) == expected
            for name, layer in self.layers.items():
                layer.load_state_dict({parameter: torch.from_numpy(saved[f'{name}.{parameter}'].copy())
                                       for parameter, _ in layer.named_parameters()})
        self.eval().requires_grad_(False)

    def forward(self, code):
        image = code
        for layer in self.layers.values():
            image = layer(image)
        # The original decoder outputs mean-subtracted BGR at 256 x 256.
        rgb = image[:, [2, 1, 0]] + image.new_tensor([123, 117, 104])[None, :, None, None]
        return (rgb / 255).clamp(0, 1)


def render(model, generator, upper, layer, channel, steps, seed):
    initial = np.random.RandomState(seed).normal(size=(1, 4096)).astype("float32")
    code = torch.from_numpy(initial).to(upper.device).requires_grad_()
    for step in range(steps):
        model.relu_warmup = step < 17
        image = generator(code)[:, :, 16:240, 16:240]
        activation = model(image)[layer][:, channel].mean()
        gradient, = torch.autograd.grad(activation, code)
        with torch.no_grad():
            rate = 8 * (1 - step / steps)
            code.add_(rate * gradient / gradient.abs().mean().clamp_min(1e-12))
            code.clamp_(min=0)
            code.copy_(torch.minimum(code, upper))
            code.mul_(.99)
    model.relu_warmup = False
    with torch.no_grad():
        return generator(code)[:, :, 16:240, 16:240]


def main():
    parser = parser_for("2016: DGN-AM", "dgn_am_2016", steps=200)
    parser.add_argument("--seed", type=int, default=3)
    args = parse(parser)
    cache = ROOT / "data/dgn"
    weights, bounds = cache / "fc6.npz", cache / "fc6-upper.txt"
    if not weights.is_file() or not bounds.is_file():
        parser.error("Run python -m weights.dgn to prepare the fc6 generator and bounds.")
    model = load_model(args)
    generator = DGN(weights).to(args.device)
    upper = torch.from_numpy(np.loadtxt(bounds, dtype=np.float32)).reshape(1, 4096).to(args.device)
    image = render(model, generator, upper, args.layer, args.channel, args.steps, args.seed)
    save(image, args.output / f"{args.layer}-{args.channel}.png")


if __name__ == "__main__":
    main()

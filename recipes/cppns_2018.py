"""2018: CPPNs. Coordinate-to-RGB parameterization from Distill.

Copyright 2018 The Lucid Authors. Apache-2.0; see LICENSES/Lucid.txt.
Port of notebooks/differentiable-parameterizations/xy2rgb.ipynb at
https://github.com/tensorflow/lucid/tree/6dcc927e4ff4e7ef4d9c54d27b0352849dadd1bb
"""
import math

import torch
from torch import nn


from shared.cli import run
from shared.common import SIZE, maximize


class CPPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList()
        inputs = 2
        for _ in range(8):
            # A 1x1 convolution applies the same MLP layer at every (u, v).
            layer = nn.Conv2d(inputs, 24, 1)
            nn.init.normal_(layer.weight, std=1 / math.sqrt(inputs))
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            inputs = 48
        self.rgb = nn.Conv2d(inputs, 3, 1)
        nn.init.zeros_(self.rgb.weight)
        nn.init.zeros_(self.rgb.bias)

        axis = torch.linspace(-math.sqrt(3), math.sqrt(3), SIZE)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y))[None])

    def forward(self):
        values = self.coordinates
        for layer in self.hidden:
            a = layer(values).atan()
            # The unbiased variant centers the squared arctangent branch.
            values = torch.cat((a / .67, (a.square() - .45) / .396), dim=1)
        return self.rgb(values).sigmoid()


def render(model, layer, channel, steps):
    image = CPPN().to(next(model.parameters()).device)

    def score(_step):
        return model(image())[layer][:, channel].mean()

    maximize(model, image.parameters(), score, steps, .005)
    with torch.no_grad():
        return image()


if __name__ == "__main__":
    run(render, "2018: CPPNs", "cppns_2018", steps=512)

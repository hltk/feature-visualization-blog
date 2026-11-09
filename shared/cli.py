"""Common arguments and image I/O for the chapter examples."""
import argparse
from pathlib import Path

import torch

from shared.common import device, save
from shared.inception import InceptionV1


def parser_for(description, name, *, steps, target="mixed4d:475"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target", default=target, metavar="LAYER:CHANNEL")
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=device())
    parser.add_argument("--output", type=Path, default=Path("output") / name)
    return parser


def parse(parser):
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    try:
        layer, index = args.target.split(":")
        channel = int(index)
        if layer not in ("conv2d0", *InceptionV1.blocks) or channel < 0:
            raise ValueError
    except ValueError:
        parser.error("--target must be a known layer and a nonnegative channel, e.g. mixed4d:475")
    args.layer, args.channel = layer, channel
    return args


def load_model(args):
    torch.set_num_threads(4)
    return InceptionV1().to(args.device)


def run(render, description, name, *, steps, target="mixed4d:475"):
    args = parse(parser_for(description, name, steps=steps, target=target))
    model = load_model(args)
    torch.manual_seed(1)
    image = render(model, args.layer, args.channel, args.steps)
    save(image, args.output / f"{args.layer}-{args.channel}.png")

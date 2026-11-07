import argparse
from pathlib import Path

import torch

from recipes import cppn, dream, fourier, pixels
from shared.common import device, save
from shared.inception import InceptionV1

BIRDS = ("mixed4d", 475)
DAISIES = ("mixed4d", 485)
HOUSES = ("mixed4c", 509)
DOG = ("mixed5a", 25)
CHANNELS = (BIRDS, DAISIES, HOUSES)
COVER = (
    ("conv2d0", 60), ("mixed3a", 196), ("mixed4a", 11),
    ("mixed4b", 495), ("mixed4e", 55),
    ("conv2d0", 62), ("mixed3a", 48), ("mixed4a", 97),
    ("mixed4c", 420), ("mixed4d", 161),
)
PIXEL_METHODS = ("pixels", "l2-pixels", "tv-pixels", "l2-gaussian-blur", "jitter-multiscale")
METHODS = (*PIXEL_METHODS, "fourier", "maco", "centered-2026", "cppn", "dream", "objectives", "cover")


def render(model, method, steps, output, target=None):
    if method == "objectives":
        for objective in ("neuron", "channel"):
            for layer, channel in (BIRDS, DOG):
                torch.manual_seed(1)
                image = fourier.render(model, layer, channel, "centered-2026", steps, objective=objective)
                save(image, output / f"{objective}-{layer}-{channel}.png")
        for layer, strength in (("mixed3a", .15), ("mixed5a", .015)):
            torch.manual_seed(1)
            image = fourier.render(model, layer, None, "centered-2026", steps,
                                   objective="layer", strength=strength)
            save(image, output / f"layer-{layer}.png")
        return

    targets = (target,) if target is not None else COVER if method == "cover" else CHANNELS
    for index, (layer, channel) in enumerate(targets, 1):
        torch.manual_seed(1)
        name = f"{layer}-{channel}"
        print(f"{method} {name}", flush=True)
        if method in PIXEL_METHODS:
            image = pixels.render(model, layer, channel, method, steps)
        elif method == "cppn":
            image = cppn.render(model, layer, channel, steps)
        elif method == "dream":
            image = dream.render(model, layer, channel, steps)
        else:
            centered = method in ("cover", "centered-2026")
            image = fourier.render(model, layer, channel, "centered-2026" if centered else method,
                                   steps, objective="centered" if centered else "channel")
        if method == "cover":
            name = f"{index:02}-{name}"
        save(image, output / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description="Reproduce the blog's feature visualizations.")
    parser.add_argument("method", choices=("all", *METHODS))
    parser.add_argument("--target", metavar="LAYER:CHANNEL", help="render one channel for a method")
    parser.add_argument("--steps", type=int, help="override updates (per octave for DeepDream)")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=device())
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    target = None
    if args.target:
        if args.method in ("all", "cover", "objectives"):
            parser.error("--target requires a single channel method")
        try:
            layer, channel = args.target.split(":")
            target = layer, int(channel)
            if layer not in ("conv2d0", *InceptionV1.blocks) or target[1] < 0:
                raise ValueError
        except ValueError:
            parser.error("--target must be a known layer and a nonnegative channel, e.g. mixed4d:475")

    torch.set_num_threads(4)
    model = InceptionV1().to(args.device)
    for method in METHODS if args.method == "all" else (args.method,):
        steps = args.steps or {"cppn": 512, "dream": 14}.get(method, 2560)
        render(model, method, steps, args.output / method, target)


if __name__ == "__main__":
    main()

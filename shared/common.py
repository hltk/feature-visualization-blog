from pathlib import Path

import torch
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SIZE = 224


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def maximize(model, parameters, score, steps, rate, *, project=None, warmup=True):
    optimizer = torch.optim.Adam(parameters, lr=rate(0) if callable(rate) else rate)
    for step in range(steps):
        model.relu_warmup = warmup and step < 17
        if callable(rate):
            optimizer.param_groups[0]["lr"] = rate(step)
        optimizer.zero_grad()
        value = score(step)
        # Adam minimizes, so descend on the negative activation objective.
        (-value).backward()
        optimizer.step()
        if project is not None:
            with torch.no_grad():
                project(step)
        if (step + 1) % 256 == 0 or step + 1 == steps:
            print(f"  {step + 1:4}/{steps}  objective {value.item():.2f}", flush=True)
    model.relu_warmup = False


def save(image, path):
    pixels = image.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((255 * pixels).round().byte().cpu().numpy()).save(path)

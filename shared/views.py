"""The 2017 recipe's jitter, scale, rotation and crop sequence."""
import math

import torch
from torch.nn import functional as F


def _jitter(image, amount, generator):
    height, width = image.shape[-2:]
    dy, dx = torch.randint(amount + 1, (2,), generator=generator).tolist()
    return image[:, :, dy:dy + height - amount, dx:dx + width - amount]


def distill_2017_view(image, generator):
    """The article's ordered transforms, returning the original image size."""
    size = image.shape[-1]
    view = F.pad(image, (16,) * 4, value=0.5)
    view = _jitter(view, 16, generator)
    scales = (1.0, 0.975, 1.025, 0.95, 1.05)
    scale = scales[int(torch.randint(len(scales), (), generator=generator))]
    height, width = view.shape[-2:]
    new_height, new_width = int(height * scale), int(width * scale)
    # Legacy bilinear resize samples at output_index * input_size / output_size.
    y = torch.arange(new_height, device=image.device, dtype=image.dtype) * height / new_height
    x = torch.arange(new_width, device=image.device, dtype=image.dtype) * width / new_width
    yy, xx = torch.meshgrid(2 * (y + 0.5) / height - 1, 2 * (x + 0.5) / width - 1, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    view = F.grid_sample(view, grid, mode="bilinear", padding_mode="border", align_corners=False)
    angle = int(torch.randint(-5, 6, (), generator=generator)) * math.pi / 180
    theta = image.new_tensor([[[math.cos(angle), -math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0]]])
    grid = F.affine_grid(theta, view.shape, align_corners=False)
    # Lucid uses the rotation operator's default nearest-neighbor interpolation.
    view = F.grid_sample(view - 0.5, grid, mode="nearest", padding_mode="zeros", align_corners=False) + 0.5
    view = _jitter(view, 8, generator)
    # Crop the padding; small scaled crops are centered and gray-padded.
    pad_y = max(size - view.shape[-2], 0)
    pad_x = max(size - view.shape[-1], 0)
    view = F.pad(view, (pad_x // 2, pad_x - pad_x // 2,
                        pad_y // 2, pad_y - pad_y // 2), value=0.5)
    offset_y = (view.shape[-2] - size) // 2
    offset_x = (view.shape[-1] - size) // 2
    return view[:, :, offset_y:offset_y + size, offset_x:offset_x + size]

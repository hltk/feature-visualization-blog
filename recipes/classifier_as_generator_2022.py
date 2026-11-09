"""2022: Classifier as Generator, using the blog's frozen MAE adaptation."""
import importlib.util
import sys

import torch
from torch import nn
from torch.nn import functional as F

from weights.mae import CACHE
from shared.cli import load_model, parse, parser_for
from shared.common import maximize, save


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        runtime = CACHE / 'runtime'
        sys.path.insert(0, str(runtime))
        try:
            spec = importlib.util.spec_from_file_location('cag_official_mae', runtime / 'models_mae.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.mae = module.mae_vit_base_patch16()
        finally:
            sys.path.remove(str(runtime))
        weights = torch.load(CACHE / 'mae_visualize_vit_base.pth', map_location='cpu', weights_only=True)
        self.mae.load_state_dict(weights['model'], strict=True)
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.eval().requires_grad_(False)

    @staticmethod
    def view(image):
        # Fixed pretrained-MAE view; optimized pixel resolution is a separate axis.
        return F.interpolate(image, (224, 224), mode='bilinear', align_corners=False, antialias=True)

    def reconstruct(self, image, generator, mask_ratio=.75):
        """Reconstruct RGB from the unmasked patches.

        Encoder matches official MAE, with CPU-generated random masks for explicit
        reproducible streams on both CPU and MPS. Decoder/unpatchify are unmodified.
        """
        rgb = self.view(image)
        normalized = (rgb - self.rgb_mean) / self.rgb_std
        tokens = self.mae.patch_embed(normalized) + self.mae.pos_embed[:, 1:]
        batch, count, width = tokens.shape
        keep = int(count * (1-mask_ratio))
        noise = torch.rand(batch, count, generator=generator)
        order = noise.argsort(dim=1).to(tokens.device)
        restore = order.argsort(dim=1)
        tokens = tokens.gather(1, order[:, :keep, None].expand(-1, -1, width))
        cls = (self.mae.cls_token + self.mae.pos_embed[:, :1]).expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        for block in self.mae.blocks:
            tokens = block(tokens)
        prediction = self.mae.forward_decoder(self.mae.norm(tokens), restore)
        reconstructed = self.mae.unpatchify(prediction) * self.rgb_std + self.rgb_mean
        return reconstructed


def render(model, mae, layer, channel, steps):
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(1001)
    initial = torch.Generator().manual_seed(11)
    image = (.5 + .1 * torch.randn(1, 3, 64, 64, generator=initial)).clamp(0, 1).to(device)
    for size, rate in ((64, .05), (128, .025), (256, .0125)):
        image = F.interpolate(image.detach(), (size, size), mode="bilinear", align_corners=False)
        image.requires_grad_()

        def score(_step):
            reconstruction = mae.reconstruct(image, rng)
            feature = model(reconstruction)[layer][:, channel]
            return feature[:, feature.shape[-2] // 2, feature.shape[-1] // 2].mean()

        def project(_step):
            image.clamp_(0, 1)

        maximize(model, [image], score, steps, rate, project=project, warmup=size == 64)

    with torch.no_grad():
        return mae.reconstruct(image, torch.Generator().manual_seed(9001))


def main():
    parser = parser_for(__doc__, "classifier_as_generator_2022", steps=1000)
    args = parse(parser)
    if not (CACHE / "runtime/models_mae.py").is_file() or not (CACHE / "mae_visualize_vit_base.pth").is_file():
        parser.error("Run python -m weights.mae to prepare the MAE.")
    model = load_model(args)
    mae = MaskedAutoencoder().to(args.device)
    image = render(model, mae, args.layer, args.channel, args.steps)
    save(image, args.output / f"{args.layer}-{args.channel}.png")


if __name__ == "__main__":
    main()

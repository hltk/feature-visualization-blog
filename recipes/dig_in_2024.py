"""2024: DiG-IN, with a supplied text prompt and an Inception v1 target."""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from shared.cli import load_model, parse, parser_for
from shared.common import save

PROMPTS = {
    "mixed4d:475": "a bird standing in the water",
    "mixed4d:485": "a bunch of white flowers sitting on top of each other flowers",
    "mixed4c:509": "the building is made of stone",
}


def activation(classifier, image, layer, channel):
    view = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    return classifier(view)[layer][:, channel].mean()


class Denoiser:
    def __init__(self, pipe, args, scheduler_class):
        self.pipe = pipe
        self.device = torch.device(args.device)
        self.guidance = args.guidance
        self.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        self.scheduler.set_timesteps(args.ddim_steps)
        self.timesteps = self.scheduler.timesteps.tolist()
        self.side = args.resolution // pipe.vae_scale_factor

    @torch.no_grad()
    def encode(self, prompt):
        positive, negative = self.pipe.encode_prompt(prompt, self.device, 1, True)
        return torch.cat((negative, positive)).float()

    def noise(self, seed):
        rng = torch.Generator().manual_seed(seed)
        shape = (1, self.pipe.unet.config.in_channels, self.side, self.side)
        return torch.randn(shape, generator=rng).to(self.device) * self.scheduler.init_noise_sigma

    def predict(self, latent, embeddings, timestep):
        dtype = self.pipe.unet.dtype
        inputs = self.scheduler.scale_model_input(torch.cat((latent, latent)).to(dtype), timestep)
        unconditional, conditional = self.pipe.unet(
            inputs, timestep, encoder_hidden_states=embeddings.to(dtype), return_dict=False,
        )[0].float().chunk(2)
        return unconditional + self.guidance * (conditional - unconditional)

    def decode(self, latent):
        vae = self.pipe.vae
        return (vae.decode(latent / vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)

    def denoise(self, noise, condition, offsets=None, with_grad=False):
        latent = noise
        clean = []
        for i, timestep in enumerate(self.timesteps):
            embeddings = condition if offsets is None else condition + offsets[i]
            if with_grad:
                prediction = checkpoint(self.predict, latent, embeddings, timestep, use_reentrant=False)
            else:
                prediction = self.predict(latent, embeddings, timestep)
            step = self.scheduler.step(prediction, timestep, latent, eta=0)
            latent = step.prev_sample
            if i >= len(self.timesteps) - 3:
                clean.append(step.pred_original_sample)
        return clean


def render(denoiser, classifier, layer, channel, steps, seed, prompt):
    condition = denoiser.encode(prompt)
    noise = denoiser.noise(seed).requires_grad_()
    noise_norm = noise.detach().norm()
    offsets = torch.zeros(
        (len(denoiser.timesteps), *condition.shape), device=denoiser.device, requires_grad=True,
    )
    optimizer = torch.optim.Adam([
        {'params': [noise], 'lr': .001},
        {'params': [offsets], 'lr': .01},
    ])

    def score_image(latent):
        image = denoiser.decode(latent)
        return image, activation(classifier, image, layer, channel)

    best_score = -float("inf")
    for update in range(steps + 1):
        train = update < steps
        with torch.set_grad_enabled(train):
            clean = denoiser.denoise(noise, condition, offsets, train)
            if train:
                images_and_scores = [
                    checkpoint(score_image, latent, use_reentrant=False) for latent in clean
                ]
            else:
                images_and_scores = [score_image(clean[-1])]
            image, score = images_and_scores[-1]
            value = float(score.detach())
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite activation at update {update}")
            if value > best_score:
                best_score, best_image = value, image.detach()

            if train:
                # Score the last three predicted clean images, favoring later ones.
                weights = range(1, len(images_and_scores) + 1)
                objective = sum(w * s for w, (_, s) in zip(weights, images_and_scores)) / sum(weights)
                optimizer.zero_grad(set_to_none=True)
                (-objective / 100).backward()
                optimizer.step()
                with torch.no_grad():
                    noise.mul_(noise_norm / noise.norm().clamp_min(1e-12))
        print(f"{update}/{steps}: activation {value:.2f}", flush=True)

    return best_image


def main():
    parser = parser_for(__doc__, "dig_in_2024", steps=20)
    parser.add_argument("--prompt", help="Text prompt. Blog prompts are the defaults for its three targets.")
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4",
                        help="Diffusers model ID or local fp16 checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3)
    args = parse(parser)
    prompt = args.prompt or PROMPTS.get(f"{args.layer}:{args.channel}")
    if not prompt:
        parser.error("--prompt is required for targets without a blog prompt")
    if args.resolution < 64 or args.resolution % 64 or not 3 <= args.ddim_steps < 1000:
        parser.error("Resolution must be a positive multiple of 64, with 3–999 DDIM steps.")

    from diffusers import DDIMScheduler, StableDiffusionPipeline

    classifier = load_model(args)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16", use_safetensors=True,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(args.device)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    pipe.vae.to(dtype=torch.float32)
    denoiser = Denoiser(pipe, args, DDIMScheduler)
    image = render(denoiser, classifier, args.layer, args.channel, args.steps, args.seed, prompt)
    save(image, args.output / f"{args.layer}-{args.channel}.png")


if __name__ == "__main__":
    main()

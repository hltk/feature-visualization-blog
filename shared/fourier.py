import torch

from shared.common import SIZE

# Empirical Lucid color matrix. See LICENSES/Lucid.txt.
COLOR = torch.tensor([[.26, .09, .02], [.27, 0., -.05], [.27, -.09, .03]])
COLOR /= torch.linalg.vector_norm(COLOR, dim=0).max()


def color_mix(image):
    return torch.einsum("ij,bjhw->bihw", COLOR, image)


def frequency_radius():
    return (torch.fft.fftfreq(SIZE)[:, None].square()
            + torch.fft.rfftfreq(SIZE).square()).sqrt()


def fourier_logits(coefficients, scale):
    spectrum = torch.view_as_complex(coefficients) * scale
    image = torch.fft.irfft2(spectrum, s=(SIZE, SIZE), norm="ortho")
    return color_mix(image / 4)

"""Prepare the frozen MAE used by the Masked Autoencoders example."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/cag'
REVISION = 'efb2a8062c206524e35e47d04501ed4f544c0ae8'
MAE_URL = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth'
MAE_SHA256 = 'c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e'

SOURCE_HASHES = {
    "models_mae.py": "973c01f31fdf15febb5976baecd639b83a5cc011161af852e54ca68c6c2bafb4",
    "util/pos_embed.py": "a9e02a9a5e2587a42476bd827055b8f56ab3eaefef5e096d2fd8cadae5ddd099",
    "LICENSE": "ffe3b1e0b58c1f1250ce48f84d8c37328f9a1ff59bb77b63bc97480cf2778b64",
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(path, url, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        print("Downloading", url, flush=True)
        urllib.request.urlretrieve(url, temporary)
        if digest(temporary) != expected:
            raise ValueError(f"Checksum mismatch for {url}")
        temporary.replace(path)
    elif digest(path) != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def main():
    for name, expected in SOURCE_HASHES.items():
        url = f"https://raw.githubusercontent.com/facebookresearch/mae/{REVISION}/{name}"
        download(CACHE / "source" / name, url, expected)
    download(CACHE / "mae_visualize_vit_base.pth", MAE_URL, MAE_SHA256)
    runtime = CACHE / 'runtime'
    (runtime / 'util').mkdir(parents=True, exist_ok=True)
    # Mechanical compatibility edits only; preserve the downloaded originals.
    model_source = (CACHE / 'source/models_mae.py').read_text()
    assert model_source.count('qk_scale=None, ') == 2
    (runtime / 'models_mae.py').write_text(model_source.replace('qk_scale=None, ', ''))
    position_source = (CACHE / 'source/util/pos_embed.py').read_text()
    (runtime / 'util/pos_embed.py').write_text(position_source.replace('dtype=np.float)', 'dtype=np.float64)'))
    (runtime / 'util/__init__.py').write_text('')


if __name__ == "__main__":
    main()

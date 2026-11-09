"""Download and convert the original fc6 generator: uv run experiments/prepare_dgn.py.

No Caffe or protobuf dependency. Only tensor fields from the pinned published
Caffe checkpoint are read; each extracted tensor is checked against its hash.
Downloads and converted weights stay in ignored data/dgn/.
"""
import hashlib
import json
import urllib.request
import zipfile

import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def varint(data, position):
    value, shift = 0, 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
        shift += 7


def fields(data):
    """Yield protobuf field number, wire type and scalar or byte payload."""
    position = 0
    while position < len(data):
        tag, position = varint(data, position)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = varint(data, position)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[position:position + size]
            position += size
        elif wire == 2:
            size, position = varint(data, position)
            value = data[position:position + size]
            position += size
        else:
            raise ValueError(f'Unsupported wire type: {wire}')
        yield number, wire, value


def tensor(data):
    dimensions, packed = [], None
    for number, wire, value in fields(data):
        if number == 7:  # BlobShape, repeated int64 dim = 1.
            for field, kind, dimension in fields(value):
                assert field == 1
                if kind == 0:
                    dimensions.append(dimension)
                else:
                    position = 0
                    while position < len(dimension):
                        size, position = varint(dimension, position)
                        dimensions.append(size)
        elif number == 5:  # Packed float data.
            assert wire == 2
            packed = value
    assert dimensions and packed is not None
    return np.frombuffer(packed, dtype='<f4').reshape(dimensions).copy()


def main():
    metadata = json.loads(Path(__file__).with_name('dgn_generator.json').read_text())
    cache = ROOT / 'data/dgn'
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / 'release_deepsim_v0.zip'
    bounds_path = cache / 'fc6-upper.txt'
    for path, url, expected in (
        (archive_path, metadata['archive_url'], metadata['archive_sha256']),
        (bounds_path, metadata['bounds_url'], metadata['bounds_sha256']),
    ):
        if not path.exists():
            print('Downloading', url, flush=True)
            temporary = path.with_suffix('.download')
            urllib.request.urlretrieve(url, temporary)
            assert digest(temporary) == expected
            temporary.replace(path)
        assert digest(path) == expected, f'Input checksum mismatch: {path}'
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(metadata['member'])
    assert hashlib.sha256(raw).hexdigest() == metadata['caffe_sha256']
    arrays = {}
    for number, wire, layer in fields(memoryview(raw)):
        if number != 100:  # NetParameter.layer.
            continue
        name, blobs = None, []
        for field, kind, value in fields(layer):
            if field == 1:
                name = bytes(value).decode()
            elif field == 7:
                blobs.append(tensor(value))
        if blobs:
            assert len(blobs) == 2
            arrays[name + '.weight'], arrays[name + '.bias'] = blobs
    assert set(arrays) == set(metadata['tensors'])
    for name, array in arrays.items():
        expected = metadata['tensors'][name]
        assert list(array.shape) == expected['shape']
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256']
    destination = cache / 'fc6.npz'
    np.savez(destination, **arrays)
    assert digest(destination) == metadata['npz_sha256']
    print('Validated original fc6 generator:', destination)


if __name__ == '__main__':
    main()

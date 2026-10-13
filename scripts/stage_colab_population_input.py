"""Assemble hash-bound imported inputs on Colab, without changing originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--parts-manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('VG_SAE_COLAB_CPU_ONLY') != '1':
        raise RuntimeError('Use the verified Colab CPU launcher')
    destination = args.destination.resolve()
    if destination.exists() or not destination.is_relative_to('/content'):
        raise ValueError('A new directory inside the Colab runtime is required')
    receipt = json.loads(args.receipt.read_text())
    parts = json.loads(args.parts_manifest.read_text())
    assembled = destination.with_suffix('.zip')
    start = time.monotonic()
    with assembled.open('xb') as output:
        for index, part in enumerate(parts):
            if part['name'] != f'part-{index:03d}.bin':
                raise ValueError('Unexpected transfer-part order or path')
            path = Path('/content')/('vg-sae-population-'+part['name'])
            if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                raise ValueError('Transfer-part bytes do not match the manifest')
            with path.open('rb') as source:
                shutil.copyfileobj(source, output, 1024*1024)
    if assembled.stat().st_size != receipt['archive_bytes'] or digest(assembled) != receipt['archive_sha256']:
        raise ValueError('Reassembled input does not match the frozen archive')
    with zipfile.ZipFile(assembled) as archive:
        names = archive.namelist()
        if names != receipt['members'] or len(names) != receipt['input_members']:
            raise ValueError('Archive member contract changed')
        if any(not (destination/name).resolve().is_relative_to(destination) for name in names):
            raise ValueError('Archive path escapes its new destination')
        if sum(info.file_size for info in archive.infolist()) != receipt['uncompressed_bytes']:
            raise ValueError('Unexpected uncompressed byte count')
        destination.mkdir()
        archive.extractall(destination)
    record = {'archive_sha256': receipt['archive_sha256'], 'input_members': len(names),
              'uncompressed_bytes': receipt['uncompressed_bytes'],
              'destination': str(destination), 'wall_seconds': time.monotonic()-start,
              'operation': 'lossless transfer staging; no scientific calculation'}
    output = Path(os.environ['COLAB_TASK_OUTPUT'])/'INPUT_STAGING.json'
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

"""Fetch the pinned public S+ asset before sealing; runtime never downloads weights."""
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'timm/vit_small_plus_patch16_dinov3.lvd1689m'
REVISION = 'c35074f0e1a65242948168c9358ce26705c4deb2'
SHA256 = '423e7b4b1103de4100a5c19a436cc00f1a994d82835ed51623b209ccfd1e9615'
SIZE = 114750056

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    directory = Path(__file__).resolve().parent / 'assets' / 'dinov3-splus'
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('model.safetensors', 'config.json', 'README.md', 'LICENSE.md'):
        target = directory / name
        if target.exists():
            if name == 'model.safetensors' and (target.stat().st_size != SIZE or digest(target) != SHA256):
                raise ValueError('Existing pretrained weights mismatch; refusing overwrite')
            continue
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        temporary = target.with_name(name + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'BezNomera-asset-preparation/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if name == 'model.safetensors' and (temporary.stat().st_size != SIZE or digest(temporary) != SHA256):
            raise ValueError('Downloaded pretrained weights mismatch')
        temporary.replace(target)
    receipt = {'schema_version': 1, 'repository': REPO, 'revision': REVISION,
               'architecture': 'vit_small_plus_patch16_dinov3', 'license': 'DINOv3',
               'files': {name: {'sha256': digest(directory / name), 'bytes': (directory / name).stat().st_size}
                         for name in ('model.safetensors', 'config.json', 'README.md', 'LICENSE.md')}}
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'verified', 'sha256': SHA256, 'bytes': SIZE}))

if __name__ == '__main__':
    main()

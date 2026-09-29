"""Keep CLI device checks strict; the web service manages its saved device."""
import hashlib
import json
import os
from pathlib import Path
import sys


def bind_source_manifest(proof_path=Path('/app/source-verification.json'),
                         manifest_path=Path('/app/web-source-manifest.json')):
    """Resolve the build-time marker to the digest actually verified in the image."""
    if os.getenv('VEHICLE_SOURCE_MANIFEST_SHA256') != 'computed-in-image':
        return
    proof = json.loads(Path(proof_path).read_text(encoding='utf-8'))
    actual = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if proof.get('status') != 'verified' or proof.get('source_manifest_sha256') != actual:
        raise ValueError('Image source verification does not match its source manifest.')
    os.environ['VEHICLE_SOURCE_MANIFEST_SHA256'] = actual


def main():
    if len(sys.argv) < 2:
        raise ValueError('A container command is required.')
    bind_source_manifest()
    command = sys.argv[1:]
    # The settings page must remain reachable when the selected GPU is absent.
    # Web inference readiness is exposed separately by /api/v1/status.
    web_command = command[:4] == ['python', '-m', 'uvicorn', 'vehicle.server.app:app']
    device = os.getenv('VEHICLE_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('VEHICLE_DEVICE must be cpu or cuda.')
    if device == 'cuda' and not web_command:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Check the NVIDIA driver and Docker GPU access; CPU fallback is disabled.')
        # Run an actual kernel: checking the wheel version alone is insufficient.
        probe = torch.ones((2, 2), device='cuda')
        if not torch.isfinite(probe @ probe).all().item():
            raise RuntimeError('CUDA startup kernel returned non-finite values.')
        torch.cuda.synchronize()
        del probe
    os.execvp(command[0], command)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Runtime rejected: {error}', file=sys.stderr)
        sys.exit(2)

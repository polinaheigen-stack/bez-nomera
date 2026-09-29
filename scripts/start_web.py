"""Verify, build and start the offline E27 web application. GPU is the default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def launch_plan(root, environment=None, *, device='cuda', port=8027):
    if device not in ('cuda', 'cpu') or type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Device must be cuda or cpu; port must be 1024..65535.')
    root = Path(root).resolve()
    manifest = root / 'SOURCE_SHA256.json'
    if not manifest.is_file():
        raise ValueError('SOURCE_SHA256.json is missing. Use the complete final delivery.')
    bundle = json.loads((root / 'models/e27/bundle.json').read_text(encoding='utf-8'))
    if bundle.get('adapter') != 'e27_compact':
        raise ValueError('Expected the matching E27 model bundle.')
    env = dict(os.environ if environment is None else environment)
    env.update(SOURCE_MANIFEST_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(), WEB_PORT=str(port))
    service = 'web' if device == 'cuda' else 'web-cpu'
    compose = ['docker', 'compose', '-f', 'docker-compose.yml', '--profile', 'cpu']
    commands = [[sys.executable, 'verify_context.py'],
                compose + ['build', service],
                compose + ['stop', 'web', 'web-cpu'],
                compose + ['up', '-d', '--no-build', '--force-recreate', service]]
    return commands, env


def start(root=ROOT, *, dry_run=False, device='cuda', port=8027):
    root = Path(root).resolve()
    commands, env = launch_plan(root, device=device, port=port)
    if not dry_run:
        for command in commands:
            subprocess.run(command, cwd=root, env=env, check=True)
    return {'status': 'planned_only' if dry_run else 'container_started',
            'verification_executed': not dry_run, 'device': device,
            'commands': [['python', *commands[0][1:]], *commands[1:]], 'source_manifest_sha256': env['SOURCE_MANIFEST_SHA256'],
            'url': f'http://127.0.0.1:{port}',
            'model_status': 'Check /api/v1/status: model.available must be true before searching.',
            'note': 'Every service start clears its galleries, history and reports. Download results before restarting.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--port', type=int, default=8027)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them.')
    args = parser.parse_args()
    try:
        report = start(dry_run=args.dry_run, device=args.device, port=args.port)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

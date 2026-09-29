"""Upload real control files through the same HTTP API as the browser; save evidence."""
import argparse
import http.client
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid


def get(base, path):
    with urllib.request.urlopen(base.rstrip('/') + path, timeout=120) as response:
        return json.load(response)


def multipart(base, path, fields, files):
    boundary = 'E27-' + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(), None))
    for name, file in files:
        file = Path(file)
        if any(char in file.name for char in ('"', '\r', '\n')):
            raise ValueError('Unsafe multipart filename.')
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{file.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(), file))
    closing = f'--{boundary}--\r\n'.encode()
    length = len(closing) + sum(len(header) + (file.stat().st_size + 2 if file else 0) for header, file in parts)
    url = urllib.parse.urlsplit(base)
    if url.scheme not in ('http', 'https'):
        raise ValueError('HTTP(S) base URL required.')
    connection = (http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection)(url.hostname, url.port, timeout=600)
    try:
        connection.putrequest('POST', url.path.rstrip('/') + path)
        connection.putheader('Content-Type', f'multipart/form-data; boundary={boundary}')
        connection.putheader('Content-Length', str(length))
        connection.endheaders()
        for header, file in parts:
            connection.send(header)
            if file:
                with file.open('rb') as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        connection.send(chunk)
                connection.send(b'\r\n')
        connection.send(closing)
        response = connection.getresponse()
        content = response.read()
        if response.status not in (200, 201, 202):
            raise RuntimeError(f'Upload rejected: HTTP {response.status}: {content.decode(errors="replace")}')
        return json.loads(content)
    finally:
        connection.close()


def wait_for(base, path, completed, timeout=7200):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        value = get(base, path)
        state = value.get('status')
        progress = (state, value.get('processed'))
        if progress != last:
            print(path, progress, flush=True)
            last = progress
        if state in completed:
            return value
        if state in ('failed', 'cancelled', 'unavailable'):
            raise RuntimeError(json.dumps(value, ensure_ascii=False))
        time.sleep(1)
    raise TimeoutError(f'Timed out: {path}')


def save_download(base, url, output):
    with urllib.request.urlopen(base.rstrip('/') + url, timeout=120) as response, output.open('wb') as target:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            target.write(chunk)


def run(base, control, output, require_device='cuda'):
    control, output = Path(control), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory.')
    status = get(base, '/api/v1/status')
    if not status['available'] or status['model']['device'] != require_device:
        raise ValueError(f'Required {require_device} model is not ready: {status}')
    output.mkdir(parents=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save('status.json', status)
    gallery_files = sorted((control / 'gallery').glob('*'))
    gallery = multipart(base, '/api/v1/galleries', {'name': f'E27 · контроль {len(gallery_files)}'},
                        [('csv', control / 'gallery.csv')] + [('images', p) for p in gallery_files])
    gallery = wait_for(base, '/api/v1/galleries/' + gallery['id'], {'ready'})
    save('gallery.json', gallery)
    run = multipart(base, '/api/v1/runs', {'gallery_id': gallery['id']},
                    [('csv', control / 'query.csv')] + [('images', p) for p in sorted((control / 'query').glob('*'))])
    run_path = '/api/v1/runs/' + run['id']
    # Labels are attached only to scoring, never sent to the neural network.
    multipart(base, run_path + '/evaluation', {}, [('ground_truth', control / 'ground_truth.csv')])
    run = wait_for(base, run_path, {'completed'})
    if run['failed'] or run['processed'] != run['total']:
        raise ValueError('The web run contains processing errors.')
    evaluation = wait_for(base, run_path + '/evaluation', {'completed'})
    save('run.json', get(base, run_path))
    save('evidence.json', get(base, run_path + '/evidence'))
    save('quality.json', evaluation)
    save_download(base, run_path + '/export', output / 'result.zip')
    save_download(base, run_path + '/evaluation/report', output / 'evaluation.json')
    summary = {'passed': True, 'scope': 'real HTTP upload and inference', 'device': require_device,
               'run_id': run['id'], 'query_count': run['total'], 'gallery_count': gallery['count'],
               'metrics': evaluation['report']['metrics'], 'official_speed_measurement': False}
    save('web-check.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8027')
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    run(args.base_url, args.control, args.output, args.device)

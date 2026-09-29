"""Select the frozen 77 queries and 324 gallery images without changing pixels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(dataset, output, control):
    dataset, output, control = map(lambda p: Path(p).resolve(), (dataset, output, control))
    images = dataset / 'images'
    if not images.is_dir():
        raise ValueError('Dataset must contain the original images/ directory.')
    if output == dataset or dataset in output.parents or output == control or control in output.parents:
        raise ValueError('Output must be outside the original dataset and control source.')
    if output.exists():
        raise ValueError('Use a new output directory; existing files are not overwritten.')
    manifest = json.loads((control / 'control-manifest.json').read_text(encoding='utf-8'))
    if set(manifest.get('csv_sha256', {})) != {'query.csv', 'gallery.csv', 'ground_truth.csv'}:
        raise ValueError('Unexpected control CSV manifest.')
    for name, expected in manifest['csv_sha256'].items():
        if sha256(control / name) != expected:
            raise ValueError(f'Control CSV changed: {name}')
    ids = {}
    for split, expected_count in (('query', manifest['query_count']), ('gallery', manifest['gallery_count'])):
        with (control / (split + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        ids[split] = [row['image_id'] for row in records]
        recorded = [r['image_id'] for r in manifest['images'] if r['split'] == split]
        if ids[split] != recorded or len(ids[split]) != expected_count or len(set(recorded)) != expected_count:
            raise ValueError('Control manifest count/order differs from CSV: ' + split)
    # All hashes are checked before copying; no relabeling, resize or recrop.
    selected = []
    for record in manifest['images']:
        filename = record['file_name']
        if Path(filename).name != filename or Path(filename).stem != record['image_id'] or record['split'] not in ('query', 'gallery'):
            raise ValueError('Invalid control image record.')
        image = images / filename
        if not image.is_file() or image.stat().st_size != record['bytes'] or sha256(image) != record['sha256']:
            raise ValueError(f'Original image missing or changed: {filename}')
        selected.append((image, record['split'], record['sha256']))
    for split in ('query', 'gallery'):
        (output / split).mkdir(parents=True)
    for image, split, expected in selected:
        destination = output / split / image.name
        shutil.copyfile(image, destination)
        if sha256(destination) != expected:
            raise ValueError('Copied control image changed during preparation: ' + image.name)
    for name in ('query.csv', 'gallery.csv', 'ground_truth.csv', 'control-manifest.json'):
        shutil.copyfile(control / name, output / name)
        if sha256(output / name) != sha256(control / name):
            raise ValueError('Control metadata changed during preparation: ' + name)
    result = {'status': 'verified', 'queries': manifest['query_count'], 'gallery': manifest['gallery_count'],
              'original_image_bytes': True, 'output': str(output),
              'scope': manifest['scope']}
    (output / 'preparation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, default=Path(__file__).resolve().parents[1] / 'control')
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output, args.control), ensure_ascii=False))

"""Frozen E27 inference and optional official scoring; no training or threshold fitting."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def components():
    from vehicle.server.provider import ProductionProvider
    from vehicle.server.batch import run_batch
    from vehicle.server.evaluation import evaluate_export
    return ProductionProvider, run_batch, evaluate_export


def export_results(images, query, gallery, output, *, ground_truth=None, device='cuda', batch_size=32):
    if device not in ('cpu', 'cuda') or type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Device must be cpu/cuda; batch size must be 1..32.')
    images, query, gallery, output = map(Path, (images, query, gallery, output))
    if not images.is_dir() or not query.is_file() or not gallery.is_file():
        raise ValueError('Original images directory and both bounding-box CSV files are required.')
    if ground_truth is not None and not Path(ground_truth).is_file():
        raise ValueError('Ground truth file is missing.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory; previous results are never overwritten.')
    factory, run_batch, evaluate = components()
    from vehicle.server.e27_contract import MODEL_VERSION
    provider = factory(device=device)
    try:
        if not provider.available:
            raise RuntimeError(provider.reason or 'The requested model/device is unavailable. No fallback is used.')
        if provider.model.get('version') != MODEL_VERSION:
            raise ValueError('This export requires the frozen E27 bundle.')
        proof = run_batch(provider, images, query, gallery, output, batch_size=batch_size)
    finally:
        provider.close()
    report = {'status': 'completed', 'model_version': proof['model']['version'], 'device': device,
              'output': str(output), 'query_count': len(proof['input_order']['query']),
              'gallery_count': len(proof['input_order']['gallery']),
              'quality_measured': False, 'official_gpu_speed_measured': False}
    if ground_truth is not None:
        # Labels enter only the official evaluator, after all predictions are frozen.
        quality = evaluate(output, query, gallery, Path(ground_truth))
        report.update(quality_measured=True, official_metrics=quality['official'])
    else:
        report['quality_note'] = 'No ground truth supplied: quality cannot be calculated.'
    (output / 'EXPORT-REPORT.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default=os.getenv('VEHICLE_DEVICE', 'cuda'))
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    os.environ.setdefault('VEHICLE_MODEL_BUNDLE', str(ROOT / 'models/e27'))
    try:
        result = export_results(args.images, args.query, args.gallery, args.output,
                                ground_truth=args.ground_truth, device=args.device, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()

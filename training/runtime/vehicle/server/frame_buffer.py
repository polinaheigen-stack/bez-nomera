"""Read one original frame ahead; the consumer alone owns model execution.

No resizing, colour conversion, EXIF transformation or tensor preparation occurs
here. A single pending future bounds decoded frames to current + one upcoming.
Use as a context manager so early exits also close the pending frame.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from PIL import Image


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_settings(ahead):
    if type(ahead) is not int or ahead not in (0, 1):
        raise ValueError('Frame prefetch must be 0 or 1.')
    return {'prefetch': ahead, 'decode_workers': ahead, 'max_open_frames': ahead + 1,
            'model_execution': 'consumer_only', 'pixel_transform': 'none',
            'source_sha256': _sha256(__file__)}


@dataclass
class PreparedFrame:
    row: dict
    image: object = None
    sha256: str = None
    prepare_ms: float = 0.
    error: Exception = None

    def require_image(self):
        if self.error is not None:
            raise self.error
        if self.image is None:
            raise RuntimeError('Prepared frame is closed.')
        return self.image

    def verify_unchanged(self):
        if self.sha256 is not None and _sha256(self.row['path']) != self.sha256:
            raise ValueError(f"Input image changed during inference: {self.row['image_id']}")

    def close(self):
        if self.image is not None:
            self.image.close()
            self.image = None


def prepare_frame(row, *, verify_hash=True):
    started = time.perf_counter()
    frame = PreparedFrame(row=row)
    try:
        if verify_hash:
            frame.sha256 = _sha256(row['path'])
        frame.image = Image.open(row['path'])
        image = frame.image
        if image.format not in ('JPEG', 'PNG') or getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f"Expected a single JPEG/PNG frame: {row['image_id']}")
        width, height = image.size
        x, y, w, h = row['bbox']
        if width * height > 40_000_000 or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(f"Invalid image dimensions or bbox: {row['image_id']}")
        image.load()
    except Exception as error:
        frame.close()
        # Yield failures in source order; web queries may continue after a bad image.
        frame.error = error.with_traceback(None)
    except BaseException:
        frame.close()
        raise
    frame.prepare_ms = (time.perf_counter() - started) * 1000
    return frame


class FrameBuffer:
    """Ordered, bounded iterator; a yielded PIL is valid until next() or exit.

    Cancellation cannot interrupt a file read already in progress. It prevents
    additional submissions and waits for that one read to finish during cleanup.
    """
    def __init__(self, items, *, ahead=1, cancelled=None, verify_hash=True):
        pipeline_settings(ahead)
        self.items = iter(items)
        self.ahead = ahead
        self.cancelled = cancelled or (lambda: False)
        self.verify_hash = verify_hash
        self.executor = None
        self.pending = None
        self.current = None
        self.closed = False
        self.entered = False

    def __enter__(self):
        if self.entered or self.closed:
            raise RuntimeError('FrameBuffer cannot be reused.')
        self.entered = True
        if self.ahead:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vehicle-frame-reader')
            try:
                self._submit_next()
            except BaseException:
                self.close()
                raise
        return self

    def _submit_next(self):
        if self.cancelled():
            return
        row = next(self.items, None)
        if row is not None:
            self.pending = self.executor.submit(prepare_frame, row, verify_hash=self.verify_hash)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.entered:
            raise RuntimeError('FrameBuffer must be used as a context manager.')
        if self.current is not None:
            self.current.close()
            self.current = None
        if self.closed or self.cancelled():
            raise StopIteration
        if self.ahead:
            if self.pending is None:
                raise StopIteration
            self.current = self.pending.result()
            self.pending = None
        else:
            row = next(self.items, None)
            if row is None:
                raise StopIteration
            self.current = prepare_frame(row, verify_hash=self.verify_hash)
        if self.cancelled():
            self.current.close()
            self.current = None
            raise StopIteration
        if self.ahead:
            self._submit_next()
        return self.current

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.current is not None:
            self.current.close()
            self.current = None
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        if self.pending is not None:
            pending, self.pending = self.pending, None
            if not pending.cancelled():
                # Unexpected worker failures still propagate through next(); cleanup
                # must not replace an existing exception from consumer inference.
                if pending.exception() is None:
                    pending.result().close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

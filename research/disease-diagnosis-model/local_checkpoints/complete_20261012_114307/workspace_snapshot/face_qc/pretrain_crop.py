"""Produce lossless native crops and RGB training previews, once per content hash."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
from pathlib import Path
import time
import warnings
from pretrain_release import OUT, atomic_json, digest, verify_code


def crop_record(row):
    from PIL import Image, ImageOps
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    h = row['image_id']
    result_path = OUT / 'crop_records' / h[:2] / (h + '.json')
    if result_path.exists():
        old = json.loads(result_path.read_text())
        assert old['original_sha256'] == row['original_sha256'] and old['crop_box'] == row['crop_box']
        return old
    result = dict(row, status='crop_error', exact_original_rgb_crop_verified=False, training_admitted=False)
    try:
        data = Path(row['image_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['original_sha256']:
            raise ValueError('Original changed since audit')
        with Image.open(io.BytesIO(data)) as source:
            exif = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
        assert image.size == (row['original_width'], row['original_height'])
        box = tuple(row['crop_box'])
        assert len(box) == 4 and all(type(v) is int for v in box)
        assert 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
        crop = image.crop(box)
        native = OUT / 'images_native' / h[:2] / (h + '.png')
        preview = OUT / 'images_768' / h[:2] / (h + '.jpg')
        native.parent.mkdir(parents=True, exist_ok=True)
        preview.parent.mkdir(parents=True, exist_ok=True)
        temp = native.with_suffix('.tmp')
        crop.save(temp, format='PNG', compress_level=1)
        with Image.open(temp) as saved:
            assert saved.mode == 'RGB' and saved.size == crop.size and saved.tobytes() == crop.tobytes()
        temp.replace(native)
        small = crop.copy()
        small.thumbnail((768, 768), Image.Resampling.LANCZOS)
        temp = preview.with_suffix('.tmp')
        small.save(temp, format='JPEG', quality=95, subsampling=0)
        temp.replace(preview)
        result.update(status='cropped', native_path=str(native), preview_path=str(preview),
                      native_sha256=digest(native), preview_sha256=digest(preview),
                      crop_width=crop.width, crop_height=crop.height, exif_orientation=exif,
                      exact_original_rgb_crop_verified=True, background_removed=False,
                      pixel_transform='EXIF transpose; RGB decode; rectangular crop; lossless PNG. Separate JPEG preview proportionally resized to max side 768.')
    except Exception as error:
        result.update(error_type=type(error).__name__, error_detail=str(error)[:300])
    atomic_json(result_path, result)
    return result


def run(sample, processes):
    verify_code()
    name = 'sample' if sample else 'all'
    lock = (OUT / ('crop_' + name + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [OUT / 'sample_input.jsonl'] if sample else [OUT / f'worker_{i}/input.jsonl' for i in range(11)]
    rows = []
    for path in paths:
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream)
    assert len(rows) == len({r['image_id'] for r in rows})
    started, counts, done = time.time(), {}, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as pool:
        # Bound the number of outstanding records; avoid holding all images in RAM.
        iterator = iter(rows)
        futures = {}
        for _ in range(processes * 2):
            row = next(iterator, None)
            if row is not None: futures[pool.submit(crop_record, row)] = row['image_id']
        while futures:
            finished, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                futures.pop(future)
                r = future.result()
                counts[r['status']] = counts.get(r['status'], 0) + 1
                done += 1
                row = next(iterator, None)
                if row is not None: futures[pool.submit(crop_record, row)] = row['image_id']
            if done % 100 < processes or done == len(rows):
                atomic_json(OUT / ('crop_' + name + '_status.json'), dict(state='running', processed=done, expected=len(rows), counts=counts, updated_at=time.time()))
                print(name, done, '/', len(rows), counts, 'seconds', round(time.time() - started), flush=True)
    atomic_json(OUT / ('crop_' + name + '_status.json'), dict(state='completed', processed=done, expected=len(rows), counts=counts, updated_at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--processes', type=int, default=16)
    args = parser.parse_args()
    run(args.sample, args.processes)

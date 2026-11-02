"""Consume complete CPU JSONL records while the upstream audit keeps running.

Seven disjoint consumers own CPU shard indices modulo seven. Original bytes,
identity/reservation flags and CPU measurements stay linked to each annotation.
No source deletion, cropping, or training admission is performed here.
"""
import argparse
import collections
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import warnings
import signal
import queue
import threading

import vlm_screen as baseline

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / 'face_qc/full_original_audit_v1'
OUT = ROOT / 'face_qc/qwen_stream_v2'
PROMPT = ROOT / 'face_qc/vlm_triage_prompt.txt'
WORKERS = 7


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def prepare(args):
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Existing stream preserved; use run to resume')
    upstream = json.loads((CPU / 'manifest.json').read_text())
    inputs = {str(CPU / f'input_{i:03d}.tsv'): file_hash(CPU / f'input_{i:03d}.tsv') for i in range(upstream['shards'])}
    manifest = {'scope': 'Every CPU-audited indexed original path; labels are provisional, not quality approval',
                'images': upstream['images'], 'shards': upstream['shards'], 'workers': WORKERS,
                'batch_size': args.batch_size, 'max_image_side': 1024, 'max_new_tokens': 240,
                'model_snapshot': str(baseline.MODEL), 'model_index_sha256': file_hash(baseline.MODEL / 'model.safetensors.index.json'),
                'cpu_manifest_sha256': file_hash(CPU / 'manifest.json'), 'input_sha256': inputs,
                'script_sha256': file_hash(__file__), 'helper_sha256': file_hash(baseline.__file__),
                'prompt_sha256': file_hash(PROMPT), 'assignment': 'CPU shard index modulo 7',
                'completeness_validated': False, 'training_admitted': False}
    atomic_json(OUT / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'input_sha256'}), flush=True)


class TailReader:
    def __init__(self, paths):
        self.paths = paths
        self.handles = {}
        self.cursor = 0

    def next_record(self):
        # Return complete JSON lines only. A partially appended line is retried.
        for _ in self.paths:
            path = self.paths[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.paths)
            if path not in self.handles:
                if not path.exists():
                    continue
                self.handles[path] = path.open('rb')
            f = self.handles[path]
            offset = f.tell()
            if os.fstat(f.fileno()).st_size < offset:
                raise RuntimeError('Upstream file truncated behind consumer')
            line = f.readline()
            if not line or not line.endswith(b'\n'):
                f.seek(offset)
                continue
            return json.loads(line)
        return None


def export_results(dest):
    fields = ['sample_id', 'status', 'triage', 'content_type', 'complete_face_visible', 'face_count', 'view',
              'orientation', 'certainty', 'reason', 'image_path', 'original_sha256', 'sources',
              'reservation_blocked', 'identity_review', 'identity_candidates', 'cpu_status',
              'cpu_proposed_crop_box', 'prior_assistant_view', 'cache_hit', 'training_admitted']
    temp = dest / 'results.tsv.tmp'
    with temp.open('w', newline='') as f, (dest / 'progress.jsonl').open() as source:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for line in source:
            d = json.loads(line)
            cpu = d['cpu_record']
            writer.writerow({**cpu, **d, **d.get('prediction', {}), 'cpu_status': cpu.get('status'),
                             'cpu_proposed_crop_box': cpu.get('proposed_crop_box', ''), 'training_admitted': False})
    temp.replace(dest / 'results.tsv')


def run(args):
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert file_hash(__file__) == manifest['script_sha256'], 'Worker changed'
    assert file_hash(baseline.__file__) == manifest['helper_sha256'], 'Helper changed'
    assert file_hash(PROMPT) == manifest['prompt_sha256'], 'Prompt changed'
    assert file_hash(CPU / 'manifest.json') == manifest['cpu_manifest_sha256'], 'Upstream changed'
    assert file_hash(baseline.MODEL / 'model.safetensors.index.json') == manifest['model_index_sha256']
    if not 0 <= args.worker < WORKERS:
        raise ValueError('Invalid worker')
    dest = OUT / f'worker_{args.worker}'
    dest.mkdir(exist_ok=True)
    lock = (dest / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {}
    paths = []
    for index in range(manifest['shards']):
        if index % WORKERS != args.worker:
            continue
        path = CPU / f'input_{index:03d}.tsv'
        assert file_hash(path) == manifest['input_sha256'][str(path)]
        with path.open() as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r['sample_id'] in expected:
                    raise RuntimeError('Duplicate source ID')
                expected[r['sample_id']] = r
        paths.append(CPU / f'shard_{index:03d}' / 'progress.jsonl')
    progress = dest / 'progress.jsonl'
    previous = baseline.load_progress(progress)
    done = set(previous)
    assert done <= expected.keys()
    statuses = collections.Counter(d['status'] for d in previous.values())
    routes = collections.Counter(d['triage'] for d in previous.values())
    cache = {d['original_sha256']: (d['prediction'], d['sample_id']) for d in previous.values() if d['status'] == 'ok'}
    del previous
    reader = TailReader(paths)
    prompt = PROMPT.read_text()
    model = processor = None
    batch_size = manifest['batch_size']
    newly_inferred = 0
    processing_seconds = 0.0
    last_activity = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
               'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
               'gpu_count': torch.cuda.device_count()}
    atomic_json(dest / 'runtime.json', runtime)

    def status(state):
        atomic_json(dest / 'status.json', {'state': state, 'worker': args.worker, 'expected': len(expected),
                    'processed': len(done), 'status_counts': dict(statuses), 'triage_counts': dict(routes),
                    'newly_inferred_this_session': newly_inferred,
                    'processing_seconds_this_session': round(processing_seconds, 3),
                    'inference_images_per_second_this_session': round(newly_inferred / processing_seconds, 3) if processing_seconds else None,
                    'updated_at_unix': time.time(), 'training_admitted': 0})

    def generate(images, prepared_inputs=None):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if prepared_inputs is None else None
        inputs = generated = None
        oom = False
        try:
            cpu_inputs = prepared_inputs if prepared_inputs is not None else processor(text=[text] * len(images), images=images, padding=True, return_tensors='pt')
            inputs = {key: value.to(model.device) for key, value in cpu_inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=manifest['max_new_tokens'], do_sample=False)
            return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            if len(images) == 1:
                raise
            oom = True
        finally:
            del inputs, generated
        if oom:
            torch.cuda.empty_cache()
            middle = len(images) // 2
            print(json.dumps({'event': 'split_oom_batch', 'size': len(images)}), flush=True)
            return generate(images[:middle]) + generate(images[middle:])

    status('starting')
    if len(done) < len(expected):
        if model is None:
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Each consumer requires exactly one allocated GPU')
            status('loading_model')
            print(json.dumps({'event': 'loading_model', 'worker': args.worker, **runtime}), flush=True)
            memory = {i: torch.cuda.get_device_properties(i).total_memory - 8 * 1024**3 for i in range(1)}
            model = Qwen3VLForConditionalGeneration.from_pretrained(baseline.MODEL, local_files_only=True,
                    dtype=torch.bfloat16, device_map={'': 0}, max_memory=memory, attn_implementation='sdpa').eval()
            processor = AutoProcessor.from_pretrained(baseline.MODEL, local_files_only=True, min_pixels=4096, max_pixels=1048576)
            processor.tokenizer.padding_side = 'left'
    prepared_queue = queue.Queue(maxsize=2)

    def prepare_batches():
        scheduled = set(done)
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if processor is not None else None
        while len(scheduled) < len(expected) and not stop_requested:
            records, images, ready = [], [], []
            pending_ids = set()
            for _ in range(10000):
                if len(records) >= batch_size:
                    break
                cpu = reader.next_record()
                if cpu is None:
                    break
                key = cpu['sample_id']
                if key in scheduled:
                    continue
                if key in pending_ids or key not in expected:
                    raise RuntimeError('Duplicate or unexpected upstream ID')
                for field in ['image_path', 'identity_candidates', 'reservation_blocked', 'identity_review', 'sources']:
                    if str(cpu[field]) != str(expected[key][field]):
                        raise RuntimeError('Upstream record does not match pinned input')
                pending_ids.add(key)
                scheduled.add(key)
                record = {'review_id': key, 'sample_id': key, 'cpu_record': cpu, 'training_admitted': False,
                          'cache_hit': False, 'completeness_validated': False,
                          'inference_run': OUT.name, 'nominal_batch_size': batch_size, 'execution_version': 'cpu_prefetch_v3'}
                records.append(record)
                if not cpu.get('original_sha256'):
                    record.update(status='upstream_unreadable', triage='manual_review', error_type=cpu.get('error_type', 'missing_original_hash'))
                    continue
                try:
                    data = Path(cpu['image_path']).read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    record['original_sha256'] = digest
                    if digest != cpu['original_sha256']:
                        record.update(status='original_changed', triage='manual_review')
                        continue
                    if digest in cache:
                        prediction, reused_id = cache[digest]
                        record.update(status='ok', prediction=prediction, triage=baseline.triage(prediction), cache_hit=True, reused_prediction_from=reused_id)
                        continue
                    with Image.open(io.BytesIO(data)) as source:
                        im = ImageOps.exif_transpose(source).convert('RGB')
                    im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    images.append(im)
                    ready.append(record)
                except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                    record.update(status='image_error', triage='manual_review', error_type=type(exc).__name__)
            if not records:
                yield None
                time.sleep(15)
                continue
            prepared_inputs = processor(text=[text] * len(images), images=images, padding=True, return_tensors='pt') if images else None
            yield records, images, ready, prepared_inputs

    def prefetch():
        try:
            for batch in prepare_batches():
                if stop_requested:
                    break
                prepared_queue.put(('batch', batch))
        except Exception as exc:
            prepared_queue.put(('error', exc))
        finally:
            prepared_queue.put(('end', None))

    if len(done) < len(expected):
        threading.Thread(target=prefetch, daemon=True, name='cpu_image_prefetch').start()
    with progress.open('a', buffering=1) as output:
        while len(done) < len(expected):
            kind, payload = prepared_queue.get()
            if kind == 'error':
                raise payload
            if kind == 'end':
                if len(done) != len(expected):
                    raise RuntimeError('Prefetch ended before complete input coverage')
                break
            if payload is None:
                records, images, ready, prepared_inputs = [], [], [], None
            else:
                records, images, ready, prepared_inputs = payload
            if not records:
                status('waiting_for_cpu')
                if time.monotonic() - last_activity > 7200:
                    raise RuntimeError('No new CPU records for two hours; inspect upstream')
                time.sleep(15)
                continue
            if images:
                start = time.monotonic()
                responses = generate(images, prepared_inputs)
                elapsed = time.monotonic() - start
                if len(responses) != len(ready):
                    raise RuntimeError('Batch output count mismatch')
                newly_inferred += len(ready)
                processing_seconds += elapsed
                for record, raw in zip(ready, responses):
                    record['raw_response'] = raw
                    record['batch_seconds_per_image'] = round(elapsed / len(ready), 4)
                    try:
                        prediction = baseline.parse_response(raw)
                        record.update(status='ok', prediction=prediction, triage=baseline.triage(prediction))
                        cache[record['original_sha256']] = (prediction, record['sample_id'])
                    except (ValueError, TypeError) as exc:
                        record.update(status='invalid_response', triage='manual_review', error_type=type(exc).__name__)
                print(json.dumps({'event': 'batch', 'worker': args.worker, 'images': len(images), 'seconds': round(elapsed, 2)}), flush=True)
            for record in records:
                # Preserve previous incomplete/nonface observations as holds.
                if record['cpu_record'].get('prior_assistant_view') in {'non_face_or_local_body_document', 'incomplete_face_original'}:
                    record['triage_before_prior_hold'] = record['triage']
                    record['triage'] = 'prior_visual_hold'
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                done.add(record['sample_id'])
                statuses[record['status']] += 1
                routes[record['triage']] += 1
            output.flush()
            os.fsync(output.fileno())
            last_activity = time.monotonic()
            status('running')
            if stop_requested:
                status('stopped_after_durable_batch')
                return
            if args.stop_after and len(done) >= args.stop_after:
                status('stopped_at_requested_limit')
                return
    export_results(dest)
    status('completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--batch-size', type=int, default=4)
    p = commands.add_parser('run')
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        run(args)

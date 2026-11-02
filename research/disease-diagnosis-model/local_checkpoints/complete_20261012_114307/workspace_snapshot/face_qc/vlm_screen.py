"""Local, original-image VLM triage. No image deletion or training admission."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/opt/craniofacial/user_home/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb')
INPUT = ROOT / 'multiview_review_v1/view_review.tsv'
PROMPT = ROOT / 'face_qc/vlm_triage_prompt.txt'
OUT = ROOT / 'face_qc/vlm_pilot_v1'
CONTENT = {'face_photo', 'partial_face_photo', 'local_body_photo', 'medical_procedure_photo', 'document_or_screen', 'other_nonface', 'uncertain'}
VIEWS = {'frontal', 'oblique', 'profile', 'strong_pitch', 'unresolved', 'not_applicable'}
PRIORITY = ['R001', 'R035', 'R048', 'R079', 'R161', 'R102', 'R179', 'R270', 'R022', 'R188']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def parse_response(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    d = json.loads(text)
    expected = {'content_type', 'complete_face_visible', 'face_count', 'view', 'orientation', 'certainty', 'reason'}
    if not isinstance(d, dict) or set(d) != expected:
        raise ValueError('Invalid schema')
    if d['content_type'] not in CONTENT or d['view'] not in VIEWS:
        raise ValueError('Invalid category')
    if type(d['complete_face_visible']) is not bool:
        raise ValueError('Invalid boolean')
    if d['face_count'] not in {'zero', 'one', 'multiple', 'uncertain'}:
        raise ValueError('Invalid face count')
    if d['orientation'] not in {'upright', 'rotated', 'uncertain', 'not_applicable'}:
        raise ValueError('Invalid orientation')
    if d['certainty'] not in {'clear', 'uncertain'} or not isinstance(d['reason'], str) or len(d['reason']) > 600:
        raise ValueError('Invalid explanation')
    return d


def triage(d):
    # Candidate separation only. Ambiguity or multiple people always stays in review.
    if d['certainty'] != 'clear' or d['face_count'] in {'multiple', 'uncertain'}:
        return 'manual_review'
    if (d['content_type'] in {'local_body_photo', 'medical_procedure_photo', 'document_or_screen', 'other_nonface'}
            and not d['complete_face_visible'] and d['face_count'] == 'zero'
            and d['view'] == 'not_applicable'):
        return 'nonface_candidate_hold'
    if (d['content_type'] == 'face_photo' and d['complete_face_visible']
            and d['face_count'] == 'one' and d['view'] in {'frontal', 'oblique', 'profile', 'strong_pitch'}
            and d['orientation'] == 'upright'):
        return 'face_candidate_quality_pending'
    return 'manual_review'


def load_progress(path):
    """Recover a truncated final append only; reject corrupt completed lines."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    done, offset = {}, 0
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                with path.open('r+b') as f:
                    f.truncate(offset)
                break
            raise
        if item['review_id'] in done:
            raise ValueError('Duplicate progress ID')
        done[item['review_id']] = item
        offset += len(line)
    # A complete JSON object can survive without the final newline.
    if done and path.read_bytes() and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:
            f.write(b'\n')
    return done


def summarize(rows, done, out):
    ordered = [done[r['review_id']] for r in rows if r['review_id'] in done]
    references = {r['review_id']: r for r in rows}
    fields = ['review_id', 'status', 'triage', 'content_type', 'complete_face_visible', 'face_count', 'view',
              'orientation', 'certainty', 'reason', 'original_sha256', 'width', 'height', 'elapsed_seconds',
              'prior_view_label', 'patient_passes_current_identity_reservation_filter', 'training_admitted']
    with (out / 'results.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for d in ordered:
            ref = references[d['review_id']]
            writer.writerow({**d, **d.get('prediction', {}), 'prior_view_label': ref['view_label'],
                             'patient_passes_current_identity_reservation_filter': ref['patient_passes_current_identity_reservation_filter']})
    counts = collections.Counter(d['triage'] for d in ordered)
    confusion = collections.Counter((references[d['review_id']]['view_label'], d.get('prediction', {}).get('view', 'error')) for d in ordered)
    prior_face_labels = {'frontal_candidate', 'oblique_candidate', 'profile_candidate', 'strong_pitch_candidate'}
    suspected = [d['review_id'] for d in ordered if references[d['review_id']]['view_label'] in prior_face_labels and d['triage'] == 'nonface_candidate_hold']
    nonface_ref = [d for d in ordered if references[d['review_id']]['view_label'] == 'non_face_or_local_body_document']
    summary = {'expected': len(rows), 'processed': len(ordered), 'complete': len(ordered) == len(rows),
               'triage_counts': dict(counts), 'status_counts': dict(collections.Counter(d['status'] for d in ordered)),
               'prior_nonface_reference_count_processed': len(nonface_ref),
               'prior_nonface_reference_triage': dict(collections.Counter(d['triage'] for d in nonface_ref)),
               'prior_face_sent_to_nonface_hold_ids': suspected,
               'prior_view_vs_model_view': [{'prior': a, 'model': b, 'count': n} for (a, b), n in sorted(confusion.items())],
               'reference_limit': 'Prior labels are assistant visual observations, not independent expert ground truth; agreement is not validated accuracy.',
               'training_admitted': 0}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def run(args):
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if torch.cuda.device_count() < 2:
        raise RuntimeError('Pilot requires two allocated CUDA GPUs')
    rows = read_rows(INPUT)
    if len({r['review_id'] for r in rows}) != len(rows):
        raise ValueError('Nonunique input IDs')
    order = {rid: i for i, rid in enumerate(PRIORITY)}
    rows.sort(key=lambda r: (order.get(r['review_id'], len(order)), r['review_id']))
    if args.limit:
        rows = rows[:args.limit]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.read_text()
    manifest = {'input_sha256': sha(INPUT.read_bytes()), 'prompt_sha256': sha(PROMPT.read_bytes()),
                'script_sha256': sha(Path(__file__).read_bytes()), 'model_snapshot': str(MODEL),
                'model_index_sha256': sha((MODEL / 'model.safetensors.index.json').read_bytes()),
                'model_config_sha256': sha((MODEL / 'config.json').read_bytes()),
                'limit': args.limit, 'max_image_side': 1024, 'max_new_tokens': 240,
                'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(),
                'dtype': 'bfloat16', 'attention': 'sdpa', 'input_mode': 'EXIF-transposed original RGB; aspect-preserving thumbnail; no segmentation',
                'training_admitted': False}
    mpath = out / 'manifest.json'
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise ValueError('Resume provenance differs; choose a new output directory')
    mpath.write_text(json.dumps(manifest, indent=2) + '\n')
    progress = out / 'progress.jsonl'
    done = load_progress(progress)
    if not set(done) <= {r['review_id'] for r in rows}:
        raise ValueError('Progress contains unknown IDs')
    pending = [r for r in rows if r['review_id'] not in done]
    if not pending:
        summarize(rows, done, out)
        return
    print(json.dumps({'event': 'loading_model', 'job_id': os.environ.get('SLURM_JOB_ID'), 'gpus': torch.cuda.device_count(), 'pending': len(pending)}), flush=True)
    memory = {i: torch.cuda.get_device_properties(i).total_memory - 8 * 1024**3 for i in range(torch.cuda.device_count())}
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                device_map='balanced', max_memory=memory, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, min_pixels=4096, max_pixels=1048576)
    print(json.dumps({'event': 'model_loaded', 'device_map': {k: str(v) for k, v in model.hf_device_map.items()}}), flush=True)
    with progress.open('a', buffering=1) as f:
        for row in pending:
            start = time.monotonic()
            result = {'review_id': row['review_id'], 'training_admitted': False,
                      'patient_passes_current_identity_reservation_filter': row['patient_passes_current_identity_reservation_filter']}
            try:
                raw_image = Path(row['original_image_path']).read_bytes()
                result['original_sha256'] = sha(raw_image)
                with Image.open(io.BytesIO(raw_image)) as source:
                    im = ImageOps.exif_transpose(source).convert('RGB')
                result.update(width=im.width, height=im.height)
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                result.update(status='image_error', error_type=type(exc).__name__, triage='manual_review')
            else:
                messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[im], return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=240, do_sample=False)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                result['raw_response'] = raw
                try:
                    prediction = parse_response(raw)
                    result.update(status='ok', prediction=prediction, triage=triage(prediction))
                except (ValueError, TypeError) as exc:
                    result.update(status='invalid_response', error_type=type(exc).__name__, triage='manual_review')
                del inputs, generated
            result['elapsed_seconds'] = round(time.monotonic() - start, 3)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            done[row['review_id']] = result
            print(json.dumps({'event': 'progress', 'processed': len(done), 'total': len(rows), 'review_id': row['review_id'], 'status': result['status'], 'triage': result['triage'], 'seconds': result['elapsed_seconds']}), flush=True)
            if len(done) % 10 == 0:
                summarize(rows, done, out)
    print(json.dumps(summarize(rows, done, out)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=0)
    run(parser.parse_args())

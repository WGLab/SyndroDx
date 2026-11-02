"""HPC-only face screening. No data or detailed errors are printed."""
import argparse
import ast
import concurrent.futures
from collections import deque
import csv
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import time
import traceback

for _name, _value in {
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1',
    'WANDB_DISABLED': 'true', 'TOKENIZERS_PARALLELISM': 'false',
}.items():
    os.environ[_name] = _value
os.umask(0o077)
RUN = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.heic', '.heif', '.gif'}
ENUMS = {
    'content': {'human_face_photo', 'other', 'uncertain'},
    'face_count': {'one', 'multiple', 'none', 'uncertain'},
    'completeness': {'complete', 'partial', 'uncertain'},
    'view': {'frontal', 'oblique', 'profile', 'other', 'uncertain'},
    'quality': {'usable', 'poor', 'uncertain'},
    'orientation': {'upright', 'rotated', 'uncertain'},
    'context': {'ordinary', 'procedure', 'heavily_occluded', 'manipulated', 'uncertain'},
    'prominence': {'adequate', 'too_small', 'uncertain'},
    'certainty': {'high', 'medium', 'low'},
}
PROMPT = '''Evaluate only whether this original photograph is suitable for full-face visual self-supervised pretraining. Do not identify the person or infer a disease, genetic finding, ethnicity, MRN or other personal information. Do not transcribe text.
Keep real single-person facial photographs with a complete, adequately visible, adequately sized face and usable sharpness/exposure. Accept frontal, oblique and true profile views. In a true profile only one eye may be visible: this is normal and must not be treated as a partial face. Natural head tilt is allowed. Neck/body cropping is allowed if the face remains complete. Ordinary portraits can have white backgrounds; do not assume a white background was removed.
Exclude drawings, scans, radiology, dolls, animals, body-only photographs, isolated ears/mouth/eyes, composite panels, multiple visible faces without a uniquely established target, procedural/operative scenes, heavily occluded or cut-off faces, tiny faces, severe blur/exposure failures, background-removed/visibly manipulated faces, and views that are not frontal/oblique/profile. Do not guess when uncertain. A photograph rotated 90/180 degrees needs later correction and is not accepted in this stage; distinguish rotated photograph from natural head tilt.
Return exactly one JSON object, without explanations or Markdown. Use these keys and enum values:
{"content":"human_face_photo|other|uncertain","face_count":"one|multiple|none|uncertain","completeness":"complete|partial|uncertain","view":"frontal|oblique|profile|other|uncertain","quality":"usable|poor|uncertain","orientation":"upright|rotated|uncertain","context":"ordinary|procedure|heavily_occluded|manipulated|uncertain","prominence":"adequate|too_small|uncertain","certainty":"high|medium|low","face_bbox":null}
For a confidently visible single complete face, optionally replace face_bbox with [left,top,right,bottom], four integers in the 0..1000 coordinate system of the displayed image. The box encloses the face including forehead and chin. Do not invent an unavailable box. These are visual screening labels, not expert or clinical ground truth.'''


def config():
    return json.loads((RUN / 'config.json').read_text())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha_file(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def references(value):
    value = (value or '').strip()
    if not value or value.lower() in {'nan', 'none', 'null'}:
        return []
    if value[:1] in {'[', '('}:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, (list, tuple)):
                    return [item for item in parsed if isinstance(item, str)]
            except (ValueError, SyntaxError, TypeError):
                pass
    return [value]


def canonical_path(value, settings):
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = [Path(settings['workspace']) / raw, Path(settings['metadata']).parent / raw]
    return next((path for path in candidates if path.is_file()), candidates[0])


def prepare():
    settings = config()
    target = RUN / 'input.sqlite'
    if (RUN / 'input_manifest.json').is_file():
        manifest = json.loads((RUN / 'input_manifest.json').read_text())
        if manifest['metadata_sha256'] != sha_file(Path(settings['metadata'])) or manifest['pipeline_sha256'] != sha_file(Path(__file__)):
            raise RuntimeError('Frozen input or implementation changed; use a new run directory')
        if manifest['input_sqlite_sha256'] != sha_file(target):
            raise RuntimeError('Input database checksum mismatch')
        print('LOCAL_INPUT_MANIFEST_ALREADY_READY', flush=True)
        return
    lock = (RUN / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    salt_path = RUN / 'identity_salt.bin'
    if not salt_path.is_file():
        salt_path.write_bytes(secrets.token_bytes(32))
    salt = salt_path.read_bytes()
    def token(value):
        return hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()
    before = Path(settings['metadata']).stat()
    metadata_sha = sha_file(Path(settings['metadata']))
    building = RUN / 'input.building.sqlite'
    if building.exists():
        building.unlink()
    database = sqlite3.connect(building)
    database.executescript('''
        CREATE TABLE aliases(alias_id TEXT PRIMARY KEY, row_key TEXT NOT NULL, patient_key TEXT,
            path TEXT, sha TEXT, status TEXT NOT NULL, source_group TEXT NOT NULL);
        CREATE TABLE contents(sha TEXT PRIMARY KEY, path TEXT NOT NULL, worker INTEGER NOT NULL);
        CREATE INDEX aliases_sha ON aliases(sha);
        CREATE INDEX aliases_path ON aliases(path);
    ''')
    unique_paths = {}
    input_rows = 0
    csv.field_size_limit(16 * 1024 * 1024)
    with Path(settings['metadata']).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if settings['image_column'] not in (reader.fieldnames or []):
            raise RuntimeError('Configured image-reference column unavailable')
        # Only the image reference and optional non-MRN grouping key are used.
        # Clinical fields and MRN columns are never copied into outputs/prompts.
        for ordinal, row in enumerate(reader):
            input_rows += 1
            row_key = token('row:' + str(ordinal))
            identity = (row.get(settings.get('patient_column')) or '').strip() if settings.get('patient_column') else ''
            patient = token('patient:' + identity) if identity else None
            paths = references(row.get(settings['image_column']))
            if not paths:
                database.execute('INSERT INTO aliases VALUES(?,?,?,?,?,?,?)',
                    (token('empty:' + str(ordinal)), row_key, patient, None, None, 'missing_reference', 'unspecified'))
            for index, value in enumerate(paths):
                path = canonical_path(value, settings)
                alias = token(f'alias:{ordinal}:{index}')
                database.execute('INSERT INTO aliases VALUES(?,?,?,?,?,?,?)',
                    (alias, row_key, patient, str(path), None, 'pending', 'unspecified'))
                unique_paths.setdefault(str(path), None)
    database.commit()
    def inspect(value):
        path = Path(value)
        try:
            first = path.stat()
            if not path.is_file():
                return value, None, 'missing_file'
            content_sha = sha_file(path)
            after = path.stat()
            if (first.st_size, first.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return value, None, 'source_changed'
            return value, content_sha, 'ready'
        except FileNotFoundError:
            return value, None, 'missing_file'
        except Exception as error:
            return value, None, 'read_error:' + type(error).__name__
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(os.environ.get('SLURM_CPUS_PER_TASK', '16')), 16)) as pool:
        for index, (path, content_sha, status) in enumerate(pool.map(inspect, unique_paths)):
            database.execute('UPDATE aliases SET sha=?,status=? WHERE path=?', (content_sha, status, path))
            if content_sha:
                worker = int(content_sha[:16], 16) % settings['workers']
                database.execute('INSERT OR IGNORE INTO contents VALUES(?,?,?)', (content_sha, path, worker))
            if index % 1000 == 0:
                database.commit()
                save_json(RUN / 'prepare_progress.json', {'state': 'preparing', 'processed_unique_paths': index + 1, 'updated_at': time.time()})
    database.commit()
    assert database.execute('PRAGMA quick_check').fetchall() == [('ok',)]
    after = Path(settings['metadata']).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Metadata changed during preparation')
    counts = dict(database.execute('SELECT status,count(*) FROM aliases GROUP BY status'))
    shards = dict(database.execute('SELECT worker,count(*) FROM contents GROUP BY worker'))
    database.close()
    building.replace(target)
    save_json(RUN / 'input_manifest.json', {'state': 'ready', 'metadata_sha256': metadata_sha,
        'input_sqlite_sha256': sha_file(target), 'pipeline_sha256': sha_file(Path(__file__)),
        'prompt_sha256': sha_file(RUN / 'prompt.txt'), 'input_rows': input_rows, 'alias_status_counts': counts,
        'unique_content_by_worker': shards, 'workers': settings['workers'],
        'patient_group_available': bool(settings.get('patient_column')),
        'privacy': 'Local processing only; no metadata in the model prompt; no data printed.'})
    print('LOCAL_INPUT_MANIFEST_READY', flush=True)


def parse_label(text):
    raw = text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    label = json.loads(raw)
    if not isinstance(label, dict) or set(label) != set(ENUMS) | {'face_bbox'}:
        raise ValueError('Invalid label keys')
    for key, choices in ENUMS.items():
        if not isinstance(label[key], str) or label[key] not in choices:
            raise ValueError('Invalid enum')
    box = label['face_bbox']
    if box is not None:
        if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int or not 0 <= x <= 1000 for x in box):
            label['face_bbox'] = None
        elif box[0] >= box[2] or box[1] >= box[3]:
            label['face_bbox'] = None
    return label


def eligible(label):
    return (label['content'] == 'human_face_photo' and label['face_count'] == 'one'
            and label['completeness'] == 'complete' and label['view'] in {'frontal', 'oblique', 'profile'}
            and label['quality'] == 'usable' and label['orientation'] == 'upright'
            and label['context'] == 'ordinary' and label['prominence'] == 'adequate'
            and label['certainty'] == 'high')


def worker(number):
    settings = config()
    manifest = json.loads((RUN / 'input_manifest.json').read_text())
    if manifest['pipeline_sha256'] != sha_file(Path(__file__)) or manifest['prompt_sha256'] != sha_file(RUN / 'prompt.txt'):
        raise RuntimeError('Frozen implementation mismatch')
    if not 0 <= number < settings['workers']:
        raise ValueError('Worker outside frozen assignment')
    locks = RUN / 'workers'; locks.mkdir(exist_ok=True)
    lock = (locks / f'worker_{number:02d}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = locks / f'worker_{number:02d}.status.json'
    source = sqlite3.connect((RUN / 'input.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)
    assignments = source.execute('SELECT sha,path FROM contents WHERE worker=? ORDER BY sha', (number,)).fetchall()
    source.close()
    result = sqlite3.connect(locks / f'worker_{number:02d}.sqlite')
    result.execute('CREATE TABLE IF NOT EXISTS labels(sha TEXT PRIMARY KEY,status TEXT,label TEXT,raw TEXT,width INTEGER,height INTEGER,updated_at REAL)')
    existing = {row[0] for row in result.execute('SELECT sha FROM labels')}
    pending = [item for item in assignments if item[0] not in existing]
    def progress(state):
        save_json(status_path, {'state': state, 'worker': number, 'expected': len(assignments),
                               'processed': result.execute('SELECT count(*) FROM labels').fetchone()[0],
                               'updated_at': time.time()})
    progress('loading_model' if pending else 'completed')
    if not pending:
        result.close(); return
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no CPU inference fallback')
    model = Qwen3VLForConditionalGeneration.from_pretrained(settings['cached_model'],
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
        local_files_only=True, low_cpu_mem_usage=True).eval()
    processor = AutoProcessor.from_pretrained(settings['cached_model'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    prompt = (RUN / 'prompt.txt').read_text()
    def decode(item):
        content_sha, filename = item
        try:
            if sha_file(Path(filename)) != content_sha:
                return item, None, 'source_changed', None
            with Image.open(filename) as original:
                photo = ImageOps.exif_transpose(original).convert('RGB')
            dimensions = photo.size
            photo.thumbnail((settings['max_image_edge'], settings['max_image_edge']), Image.Resampling.LANCZOS)
            return item, photo, 'ok', dimensions
        except Exception as error:
            return item, None, 'image_error:' + type(error).__name__, None
    def store(content_sha, status, label=None, raw=None, dimensions=None):
        result.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?)',
            (content_sha, status, json.dumps(label) if label else None, raw,
             dimensions[0] if dimensions else None, dimensions[1] if dimensions else None, time.time()))
        result.commit()
    batch_size = settings['batch_size']
    queue = []
    progress('running')
    def infer(items):
        photos = [item[1] for item in items]
        messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': photo}, {'type': 'text', 'text': prompt}]}] for photo in photos]
        text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        inputs = processor(text=text, images=photos, padding=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=384)
        outputs = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del inputs, generated
        return outputs
    def consume(items):
        nonlocal batch_size
        try:
            outputs = infer(items)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1:
                raise RuntimeError('Single-image GPU memory failure; preserve run for resume')
            batch_size = max(1, min(batch_size // 2, len(items) // 2))
            midpoint = len(items) // 2
            consume(items[:midpoint]); consume(items[midpoint:]); return
        if len(outputs) != len(items):
            raise RuntimeError('Batch response alignment failure')
        for item, raw in zip(items, outputs):
            try:
                label = parse_label(raw)
                store(item[0][0], 'ok', label, raw, item[3])
            except (ValueError, TypeError, json.JSONDecodeError):
                store(item[0][0], 'invalid_response', raw=raw, dimensions=item[3])
            item[1].close()
        progress('running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Bound decoded-image prefetch while generation occupies the GPU.
        # Executor.map would queue the whole archive and retain decoded images.
        iterator = iter(pending)
        futures = deque()
        for _ in range(max(4, batch_size * 2)):
            try:
                futures.append(pool.submit(decode, next(iterator)))
            except StopIteration:
                break
        while futures:
            item = futures.popleft().result()
            try:
                futures.append(pool.submit(decode, next(iterator)))
            except StopIteration:
                pass
            if item[2] != 'ok':
                store(item[0][0], item[2]); continue
            queue.append(item)
            if len(queue) >= batch_size:
                consume(queue); queue = []
        if queue:
            consume(queue)
    count = result.execute('SELECT count(*) FROM labels').fetchone()[0]
    if count != len(assignments):
        raise RuntimeError('Incomplete worker coverage')
    progress('completed')
    result.close()
    print('LOCAL_ANNOTATION_WORKER_COMPLETED', flush=True)


def finalize():
    settings = config()
    source = sqlite3.connect((RUN / 'input.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)
    expected = {row[0] for row in source.execute('SELECT sha FROM contents')}
    labels = {}
    for number in range(settings['workers']):
        status = json.loads((RUN / 'workers' / f'worker_{number:02d}.status.json').read_text())
        if status['state'] != 'completed':
            raise RuntimeError('Worker not completed')
        db = sqlite3.connect((RUN / 'workers' / f'worker_{number:02d}.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)
        assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        for sha, state, label, width, height in db.execute('SELECT sha,status,label,width,height FROM labels'):
            if sha in labels:
                raise RuntimeError('Duplicate content across workers')
            labels[sha] = (state, json.loads(label) if label else None, width, height)
        db.close()
    if set(labels) != expected:
        raise RuntimeError('Merged coverage mismatch')
    accepted = RUN / 'accepted_faces.tsv'; held = RUN / 'excluded_or_hold.tsv'
    columns = ['alias_id','row_key','patient_key','original_path','original_sha256','decision','reason','view','width','height','face_bbox']
    counts = {}; aliases = 0
    with accepted.open('w', newline='') as pass_file, held.open('w', newline='') as hold_file:
        writers = {True: csv.DictWriter(pass_file, fieldnames=columns, delimiter='\t'), False: csv.DictWriter(hold_file, fieldnames=columns, delimiter='\t')}
        for writer in writers.values(): writer.writeheader()
        for alias, row_key, patient, path, sha, upstream in source.execute('SELECT alias_id,row_key,patient_key,path,sha,status FROM aliases ORDER BY alias_id'):
            aliases += 1
            state, label, width, height = labels.get(sha, (upstream, None, None, None))
            passed = state == 'ok' and label is not None and eligible(label)
            decision = 'face_candidate' if passed else 'exclude_or_hold'
            reason = 'screening_passed' if passed else state if state != 'ok' else ','.join(key + ':' + str(label[key]) for key in ENUMS)
            counts[decision] = counts.get(decision, 0) + 1
            writers[passed].writerow({'alias_id':alias,'row_key':row_key,'patient_key':patient or '', 'original_path':path or '',
                'original_sha256':sha or '', 'decision':decision,'reason':reason,'view':label['view'] if label else '',
                'width':width or '', 'height':height or '', 'face_bbox':json.dumps(label['face_bbox']) if label else ''})
    original_count = source.execute('SELECT count(*) FROM aliases').fetchone()[0]
    source.close()
    assert aliases == original_count == sum(counts.values())
    summary = {'state':'annotation_complete','completed_at':time.time(),'unique_contents':len(expected),
        'alias_count':aliases,'decisions':counts,'complete_content_coverage':True,'duplicate_content_results':False,
        'accepted_faces_sha256':sha_file(accepted),'excluded_or_hold_sha256':sha_file(held),
        'not_training_ready':True,'next_stage':'Background-preserving crops, crop validation, deduplication and patient-isolated split.',
        'patient_group_field_available':bool(settings.get('patient_column')),
        'known_limits':['Automated labels are not independent expert ground truth.',
                       'MRN deletion does not anonymize facial photographs.',
                       'Patient grouping must be resolved before training if absent in this metadata.'],
        'source_files_deleted':False}
    save_json(RUN / 'summary.json',summary)
    save_json(RUN / 'completion.json',summary)
    mail = settings.get('notification_email')
    if mail and not (RUN / 'notification_receipt.json').exists():
        save_json(RUN / 'notification_receipt.json',{'state':'sending','attempted_at':time.time()})
        message = ('To: '+mail+'\nSubject: HPC face annotation completed\nContent-Type: text/plain; charset=utf-8\n\n'
                   'The local Qwen face annotation run has completed. The candidate lists and detailed validation remain on HPC. '
                   'Cropping and final training-data acceptance are the next stage.\nLocal run directory: '+str(RUN)+'\n')
        sent = subprocess.run(['/usr/sbin/sendmail','-t','-oi'],input=message,text=True,capture_output=True)
        save_json(RUN / 'notification_receipt.json',{'state':'accepted_by_local_mail_system' if sent.returncode==0 else 'delivery_failed_review_required','returncode':sent.returncode,'updated_at':time.time()})
    print('LOCAL_ANNOTATION_COMPLETE_CROPPING_NEXT', flush=True)


def selftest():
    base = {key: next(iter(choices)) for key, choices in ENUMS.items()}
    base.update(content='human_face_photo',face_count='one',completeness='complete',quality='usable',orientation='upright',context='ordinary',prominence='adequate',certainty='high',face_bbox=None)
    for view in ('frontal','profile','oblique'):
        base['view']=view; assert eligible(parse_label(json.dumps(base)))
    for key,value in [('face_count','multiple'),('completeness','partial'),('quality','poor'),('orientation','rotated'),('prominence','too_small'),('certainty','low'),('content','other'),('context','procedure'),('view','other')]:
        modified=dict(base);modified[key]=value;assert not eligible(parse_label(json.dumps(modified)))
    modified=dict(base);modified['face_count']='two'
    try:parse_label(json.dumps(modified))
    except ValueError:pass
    else:raise AssertionError('Unrecognized enum accepted')
    assert references('["/synthetic/one.jpg","/synthetic/two.jpg"]')==['/synthetic/one.jpg','/synthetic/two.jpg']
    print('SYNTHETIC_SCREENING_CHECKS_PASSED',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','worker','finalize','selftest'])
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='worker':worker(args.worker)
        elif args.stage=='finalize':finalize()
        else:selftest()
    except Exception:
        with (RUN/'private_error.log').open('a') as handle:traceback.print_exc(file=handle)
        print('LOCAL_PIPELINE_FAILED_DETAILS_RETAINED_ON_HPC',flush=True)
        raise SystemExit(1)

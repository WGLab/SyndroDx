"""Apply the audited expert reservation before training sample construction."""
import csv
from pathlib import Path
from audit_metadata_sources import mrnkey

DEFAULT_RESERVATION = Path(__file__).resolve().parent / 'expert_holdout_audit'


def filter_reserved(rows, reservation_dir=DEFAULT_RESERVATION):
    root = Path(reservation_dir)
    # Fail closed if reservation files are absent, including in copied projects.
    with (root / 'protected_mrns.tsv').open() as f:
        ids = {mrnkey(r['mrn']) for r in csv.DictReader(f, delimiter='\t')}
    with (root / 'blocked_paths.tsv').open() as f:
        paths = {str(Path(r['path'])) for r in csv.DictReader(f, delimiter='\t')}
    with (root / 'blocked_crop_hashes.tsv').open() as f:
        hashes = {r['sha256'] for r in csv.DictReader(f, delimiter='\t')}
    # The complete Ian cohort includes patients absent from the expert label TSV.
    # Keep this second reservation independent so rerunning the source audit
    # cannot accidentally release those patients.
    if root.resolve() == DEFAULT_RESERVATION.resolve():
        supplemental = root.parent / 'split_plan_v0'
        with (supplemental / 'ian_protected_mrns.tsv').open() as f:
            ids.update(mrnkey(r['mrn']) for r in csv.DictReader(f, delimiter='\t'))
        with (supplemental / 'ian_blocked_paths.tsv').open() as f:
            paths.update(str(Path(r['path'])) for r in csv.DictReader(f, delimiter='\t'))
        with (supplemental / 'ian_blocked_crop_hashes.tsv').open() as f:
            hashes.update(r['sha256'] for r in csv.DictReader(f, delimiter='\t'))
    if not ids:
        raise ValueError('Expert reservation is empty; rebuild and review the audit.')
    kept = []
    for r in rows:
        row_ids = {mrnkey(v) for k in ('mrn', 'MRN', 'original_mrn', 'latest_norm_mrn', 'identity_candidates')
                   for v in (r.get(k) or '').split('|') if v.strip()}
        row_paths = {str(Path(r[k])) for k in ('image_path', 'cropping_path') if r.get(k)}
        if not row_ids & ids and not row_paths & paths and not (r.get('crop_sha256') and r['crop_sha256'] in hashes):
            kept.append(r)
    return kept, len(rows) - len(kept)

"""Recover an existing pseudonym mapping by exact metadata path correspondence.

Runs only on HPC. Original identifiers remain in a mode0600 private artifact;
only aggregate evidence and checksums may be emitted to the caller.
"""
import collections
import csv
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent
CURRENT = Path('/controlled-data/chop_data_metadata/metadata_internal.tsv')
ORIGINAL = Path('/controlled-data/craniofacial_data/complete_full_craniofacial_metadata_updated_03152026.tsv')


def norm(value):
    text = (value or '').strip()
    if text.lower() in {'', 'nan', 'none', 'null', 'na'}:
        return ''
    if re.fullmatch(r'\d+\.0', text):
        text = text[:-2]
    return (text.lstrip('0') or '0') if text.isdigit() else text


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    os.umask(0o077)
    private = WORK / 'private'
    private.mkdir(exist_ok=True)
    private.chmod(0o700)
    bypath = collections.defaultdict(set)
    pseudonyms = set()
    with CURRENT.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            pseudo = (row.get('pseudo_mrn') or row.get('psedudo_mrn') or '').strip()
            if pseudo:
                bypath[os.path.normpath(row['image_path'])].add(pseudo)
                pseudonyms.add(pseudo)
    forward, reverse = collections.defaultdict(set), collections.defaultdict(set)
    evidence = collections.Counter()
    with ORIGINAL.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            choices = bypath.get(os.path.normpath(row['image_path']), set())
            if not choices:
                continue
            evidence['exact_image_path_rows'] += 1
            if len(choices) != 1:
                evidence['ambiguous_path_rows'] += 1
                continue
            identifier = norm(row.get('norm_mrn'))
            if not identifier:
                evidence['missing_original_normalized_identifier_rows'] += 1
                continue
            pseudo = next(iter(choices))
            forward[identifier].add(pseudo)
            reverse[pseudo].add(identifier)
    good = {p: next(iter(v)) for p, v in reverse.items()
            if len(v) == 1 and len(forward[next(iter(v))]) == 1}
    if set(good) != pseudonyms:
        raise RuntimeError('incomplete_or_ambiguous_mapping_recovery')
    target = private / 'mrn_mapping_RECONSTRUCTED_PRIVATE.csv'
    if target.exists():
        with target.open() as f:
            previous = {r['psedudo_mrn']: r['mrn'] for r in csv.DictReader(f)}
        if previous != good:
            raise RuntimeError('existing_recovery_disagrees_do_not_overwrite')
    else:
        temporary = target.with_suffix('.tmp')
        with temporary.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mrn', 'psedudo_mrn'])
            writer.writeheader()
            for pseudo in sorted(good):
                writer.writerow(dict(mrn=good[pseudo], psedudo_mrn=pseudo))
        temporary.chmod(0o600)
        temporary.replace(target)
    receipt = dict(state='reconstructed_not_the_missing_original_file',
                   private_mapping_path=str(target), reconstructed_pairs=len(good),
                   all_existing_pseudonyms_preserved=True, real_to_multiple_pseudonyms=0,
                   pseudonym_to_multiple_real_identifiers=0, evidence=dict(evidence),
                   evidence_type='Exact original image path -> original norm_mrn -> existing pseudo_mrn',
                   patient_identity_clinically_adjudicated=False,
                   sources={str(p): sha(p) for p in [CURRENT, ORIGINAL]},
                   mapping_sha256=sha(target), mapping_permissions='0600; private directory0700',
                   original_mapping_overwritten=False, mapping_contents_exported=False,
                   updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (WORK / 'mapping_recovery_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['state', 'reconstructed_pairs', 'all_existing_pseudonyms_preserved', 'evidence', 'mapping_contents_exported']}))


if __name__ == '__main__':
    main()

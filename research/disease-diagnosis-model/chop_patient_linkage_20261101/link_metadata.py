"""Local patient-level linkage. No images, MRNs, notes or secrets are logged.

The reconstructed mapping preserves existing pseudonyms. Clinical identifiers
are used in memory only; output tables use the existing salted patient keys.
Unknown assertions and unverified clinical provenance are never promoted to gold.
"""
import collections
import csv
import datetime
import gzip
import hashlib
import hmac
import json
import os
import re
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
V1 = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
RUN = V1.parents[1] / 'face_ssl_annotation/run_20261012_baed142d'
MED = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
RAW = MED / 'Clarity_20260704/clarity_raw_exports'
META = Path('/controlled-data/chop_data_metadata/metadata_internal.tsv')
ARCHIVE = Path('/controlled-data/craniofacial_data/arcus_export')
os.umask(0o077)
csv.field_size_limit(100 * 1024 * 1024)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def norm(value):
    s = str(value).strip() if value is not None else ''
    if s.lower() in {'', 'nan', 'none', 'null', 'na', 'n/a'}:
        return ''
    if re.fullmatch(r'\d+\.0', s):
        s = s[:-2]
    return (s.lstrip('0') or '0') if s.isdigit() else s


def table(path):
    with Path(path).open('rb') as f:
        magic = f.read(2)
    opener = gzip.open(path, 'rt', newline='') if magic == b'\x1f\x8b' else Path(path).open(newline='')
    with opener as f:
        line = f.readline()
        f.seek(0)
        yield from csv.DictReader(f, delimiter='\t' if line.count('\t') > line.count(',') else ',')


def parquet_rows(path, columns, patids=None):
    f = pq.ParquetFile(path)
    wanted = None
    if patids is not None:
        typ = f.schema_arrow.field('pat_id').type
        vals = list(patids)
        if pa.types.is_integer(typ):
            vals = [int(x) for x in vals if re.fullmatch(r'\d+', x)]
        wanted = pa.array(vals, type=typ)
    for b in f.iter_batches(columns=columns, batch_size=65536):
        if wanted is not None:
            b = b.filter(pc.is_in(b.column(columns.index('pat_id')), value_set=wanted))
        yield from b.to_pylist()


def main():
    private = WORK / 'private'
    private.mkdir(exist_ok=True)
    private.chmod(0o700)
    salt = (RUN / 'direct_identity_salt.bin').read_bytes()

    def token(kind, value):
        return hmac.new(salt, (kind + ':' + str(value)).encode(), hashlib.sha256).hexdigest()

    save(WORK / 'linkage_status.json', dict(state='linking_patient_identity_and_guards', updated_at_utc=now()))
    mapping = PROJECT / 'mrn_mapping_PRIVATE.csv'
    forward, reverse = {}, {}
    for r in table(mapping):
        m, p = norm(r['mrn']), (r.get('pseudo_mrn') or r.get('psedudo_mrn') or '').strip()
        if not m or not p:
            raise RuntimeError('missing_mapping_identifier')
        if m in forward and forward[m] != p or p in reverse and reverse[p] != m:
            raise RuntimeError('non_bijective_mapping')
        forward[m], reverse[p] = p, m
    people = {p: dict(patient_key=token('patient', p), pseudo_mrn=p, subsets=set(), image_rows=0,
                      reserved=False, reserve_reasons=set(), role='unassigned', group_id='',
                      identity_conflict=False) for p in reverse}
    bykey = {r['patient_key']: r for r in people.values()}
    for r in table(META):
        p = (r.get('pseudo_mrn') or r.get('psedudo_mrn') or '').strip()
        if p not in people:
            continue
        people[p]['image_rows'] += 1
        people[p]['subsets'].add(r.get('subset') or 'unassigned')
    guards = set()
    for path in [PROJECT / 'expert_holdout_audit/protected_mrns.tsv', PROJECT / 'split_plan_v0/ian_protected_mrns.tsv']:
        guards.update(norm(r.get('mrn')) for r in table(path))
    plan = load(V1 / 'dataset_release/patient_identity_plan.private.json')['patients']
    for p, row in people.items():
        info = plan.get(row['patient_key'], {})
        row['group_id'] = info.get('group_id') or row['patient_key']
        row['role'] = info.get('split') or 'unassigned'
        if reverse[p] in guards:
            row['reserve_reasons'].add('expert_or_Ian_guard')
        if '1' in row['subsets']:
            row['reserve_reasons'].add('current_subset1')
        if info.get('split') in ['protected', 'holdout', 'reserved']:
            row['reserve_reasons'].add('existing_identity_plan_reserved')
    protected_groups = {r['group_id'] for r in people.values() if r['reserve_reasons']}
    for row in people.values():
        row['reserved'] = row['group_id'] in protected_groups
        if row['reserved']:
            row['reserve_reasons'].add('patient_group_propagation')
            row['role'] = 'protected'

    db = sqlite3.connect(private / 'linked_metadata.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS stage(name TEXT PRIMARY KEY, completed_at TEXT);
        CREATE TABLE IF NOT EXISTS demographics(patient_key TEXT PRIMARY KEY, birth_year INTEGER, sex TEXT, ethnicity TEXT, bridge_state TEXT);
        CREATE TABLE IF NOT EXISTS images(photo_id TEXT PRIMARY KEY,patient_key TEXT,group_id TEXT,split TEXT,view TEXT,crop_path TEXT,crop_sha256 TEXT,source TEXT);
        CREATE INDEX IF NOT EXISTS images_patient ON images(patient_key);
        CREATE TABLE IF NOT EXISTS hpo(patient_key TEXT,code TEXT,assertion TEXT,source TEXT,source_rows INTEGER,event_date TEXT,context_present INTEGER,
            PRIMARY KEY(patient_key,code,assertion,source,event_date,context_present));
        CREATE TABLE IF NOT EXISTS notes(note_key TEXT,patient_key TEXT,source_row INTEGER,date_text TEXT,text_sha256 TEXT,text_characters INTEGER,source_state TEXT,
            PRIMARY KEY(note_key,patient_key,text_sha256));
        CREATE INDEX IF NOT EXISTS notes_patient ON notes(patient_key);
        CREATE TABLE IF NOT EXISTS variants(variant_key TEXT,patient_key TEXT,gene_symbol TEXT,hgvs TEXT,clinical_significance_code TEXT,assessment_code TEXT,
            PRIMARY KEY(variant_key,patient_key));
        CREATE INDEX IF NOT EXISTS variants_patient ON variants(patient_key);
    ''')

    def done(stage):
        return bool(db.execute('SELECT 1 FROM stage WHERE name=?', (stage,)).fetchone())

    def finish(stage):
        db.execute('INSERT OR REPLACE INTO stage VALUES(?,?)', (stage, now()))
        db.commit()

    releases = load(V1 / 'dataset_release/release.json')
    if not done('images'):
        for split, item in releases['manifests'].items():
            if split == 'reference_train':
                continue
            for r in table(item['path']):
                if r['patient_key'] not in bykey:
                    continue
                person = bykey[r['patient_key']]
                if person['reserved'] and split == 'train':
                    person['reserve_reasons'].add('prior_training_exposure_requires_audit')
                elif not person['reserved']:
                    if person['role'] not in ['unassigned', split]:
                        person['identity_conflict'] = True
                    person['role'] = split
                db.execute('INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?)',
                           (r['photo_id'], r['patient_key'], r['group_id'], split, r['view'], r['crop_path'], r['crop_sha256'], r['source']))
        finish('images')

    save(WORK / 'linkage_status.json', dict(state='linking_structured_patient_tables', patients=len(people), updated_at_utc=now()))
    pid_to_keys = collections.defaultdict(set)
    key_to_pids = collections.defaultdict(set)
    demographic_values = {}
    for r in parquet_rows(RAW / 'patient.parquet', ['pat_id', 'pat_mrn_id', 'birth_date', 'sex_c', 'ethnic_group_c']):
        pseudo = forward.get(norm(r['pat_mrn_id']))
        if not pseudo:
            continue
        key = people[pseudo]['patient_key']
        pid = norm(r['pat_id'])
        if not pid:
            continue
        pid_to_keys[pid].add(key)
        key_to_pids[key].add(pid)
        demographic_values[key] = r
    good_pid = {pid: next(iter(keys)) for pid, keys in pid_to_keys.items()
                if len(keys) == 1 and len(key_to_pids[next(iter(keys))]) == 1}
    ambiguous_keys = {key for key, pids in key_to_pids.items() if len(pids) > 1}
    ambiguous_keys |= {key for keys in pid_to_keys.values() if len(keys) > 1 for key in keys}
    for key in ambiguous_keys:
        bykey[key]['identity_conflict'] = True
    sex_names = {norm(r['internal_id']): r['name'] for r in parquet_rows(RAW / 'zc_sex.parquet', ['internal_id', 'name'])}
    ethnicity_names = {norm(r['ethnic_group_c']): r['name'] for r in parquet_rows(RAW / 'zc_ethnic_group.parquet', ['ethnic_group_c', 'name'])}
    for key, r in demographic_values.items():
        year = getattr(r['birth_date'], 'year', None)
        if year is None:
            match = re.match(r'^(\d{4})', str(r['birth_date']))
            year = int(match[1]) if match else None
        db.execute('INSERT OR REPLACE INTO demographics VALUES(?,?,?,?,?)',
                   (key, year, sex_names.get(norm(r['sex_c']), 'unknown'), ethnicity_names.get(norm(r['ethnic_group_c']), 'unknown'),
                    'ambiguous' if key in ambiguous_keys else 'one_to_one'))
    finish('demographics')

    # Independent Arcus identifiers can differ from Clarity identifiers. Keep namespaces separate.
    arcus_pid = collections.defaultdict(set)
    for file in sorted((ARCHIVE / 'arcus_2026_06_29_full_ctakes').glob('patient-*.csv')):
        for r in table(file):
            pseudo = forward.get(norm(r.get('pat_mrn_id')))
            if pseudo and norm(r.get('pat_id')):
                arcus_pid[norm(r['pat_id'])].add(people[pseudo]['patient_key'])
    good_arcus = {pid: next(iter(keys)) for pid, keys in arcus_pid.items() if len(keys) == 1}
    ontology = load(PROJECT / 'arcface1000_hpo_alignment_20261028/private/ontology.json')['terms']
    valid_hpo = set(ontology)
    if not done('structured_HPO'):
        db.execute("DELETE FROM hpo WHERE source='Arcus_clinician_HPO'")
        for file in sorted((ARCHIVE / 'arcus_2026_06_29').glob('clinician_hpo_element*')):
            for r in table(file):
                key = good_arcus.get(norm(r.get('pat_id')))
                codes = set(re.findall(r'HP:\d{7}', r.get('hpo_code', ''))) & valid_hpo
                if not key or not codes:
                    continue
                context = int(bool((r.get('notes') or '').strip()))
                date = (r.get('creation_date') or '').strip()
                # The table has no explicit presence/absence field. Do not invent polarity.
                for code in codes:
                    db.execute('INSERT INTO hpo VALUES(?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET source_rows=source_rows+1',
                               (key, code, 'recorded_polarity_unresolved', 'Arcus_clinician_HPO', 1, date, context))
        finish('structured_HPO')
    for filename, source in [('ctakes_hpo.csv', 'ctakes_candidate'), ('phenominal_hpo.csv', 'phenominal_candidate')]:
        if done(source):
            continue
        db.execute('DELETE FROM hpo WHERE source=?', (source,))
        save(WORK / 'linkage_status.json', dict(state='linking_HPO_candidates', source=source, updated_at_utc=now()))
        for i, r in enumerate(table(MED / 'CHOP_Images/final_results' / filename)):
            pseudo = forward.get(norm(r.get('pat_mrn_id')))
            codes = set(re.findall(r'HP:\d{7}', r.get('HPOId', ''))) & valid_hpo
            if not pseudo or not codes or 'pubmind' in (r.get('method') or '').lower():
                continue
            key = people[pseudo]['patient_key']
            for code in codes:
                db.execute('INSERT INTO hpo VALUES(?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET source_rows=source_rows+1',
                           (key, code, 'unknown', source, 1, '', 0))
            if i and i % 100000 == 0:
                db.commit()
        finish(source)

    if not done('variants'):
        save(WORK / 'linkage_status.json', dict(state='linking_variant_records', updated_at_utc=now()))
        genes = {norm(r['gene_c']): r['gene_symbol'] for r in parquet_rows(RAW / 'zc_gene.parquet', ['gene_c', 'gene_symbol'])}
        cols = ['pat_id', 'variant_id', 'gene_c', 'hgvs_name', 'clinical_signif_c', 'assessment_c']
        for r in parquet_rows(RAW / 'variant.parquet', cols, good_pid):
            key = good_pid.get(norm(r['pat_id']))
            if key:
                db.execute('INSERT OR REPLACE INTO variants VALUES(?,?,?,?,?,?)',
                           (token('variant', norm(r['variant_id'])), key, genes.get(norm(r['gene_c']), ''), str(r['hgvs_name'] or ''),
                            str(r['clinical_signif_c'] or ''), str(r['assessment_c'] or '')))
        finish('variants')

    if not done('notes'):
        save(WORK / 'linkage_status.json', dict(state='indexing_notes_without_copying_clinical_text', updated_at_utc=now()))
        source = MED / 'CHOP_Images/all_clarity_notes_merged.csv'
        for i, r in enumerate(table(source)):
            pseudo = forward.get(norm(r.get('MRN')))
            if not pseudo or not (r.get('TEXT') or '').strip():
                continue
            text = r['TEXT']
            db.execute('INSERT OR IGNORE INTO notes VALUES(?,?,?,?,?,?,?)',
                       (token('note', r.get('NOTE_ID') or 'row:' + str(i)), people[pseudo]['patient_key'], i,
                        (r.get('DATE') or '').strip(), hashlib.sha256(text.encode()).hexdigest(), len(text),
                        'selection_and_text_provenance_pending_no_training'))
            if i and i % 25000 == 0:
                db.commit()
                save(WORK / 'linkage_status.json', dict(state='indexing_notes_without_copying_clinical_text', source_rows_scanned=i,
                                                       updated_at_utc=now()))
        finish('notes')

    # Available data and admitted supervision remain distinct. No false negatives from absence.
    hpo_counts = collections.defaultdict(dict)
    for key, source, count in db.execute('SELECT patient_key,source,COUNT(*) FROM hpo GROUP BY patient_key,source'):
        hpo_counts[key][source] = count
    image_counts = dict(db.execute('SELECT patient_key,COUNT(*) FROM images GROUP BY patient_key'))
    note_counts = dict(db.execute('SELECT patient_key,COUNT(*) FROM notes GROUP BY patient_key'))
    variant_counts = dict(db.execute('SELECT patient_key,COUNT(*) FROM variants GROUP BY patient_key'))
    rows = []
    for pseudo, person in sorted(people.items()):
        key = person['patient_key']
        counts = hpo_counts.get(key, {})
        image_count = image_counts.get(key, 0)
        note_count = note_counts.get(key, 0)
        variant_count = variant_counts.get(key, 0)
        rows.append(dict(patient_key=key, pseudo_mrn=pseudo, group_id=person['group_id'],
                         subsets='|'.join(sorted(person['subsets'])), split=person['role'], protected=person['reserved'],
                         reservation_reasons='|'.join(sorted(person['reserve_reasons'])), identity_conflict=person['identity_conflict'],
                         metadata_image_rows=person['image_rows'], released_QC_face_images=image_count,
                         structured_HPO_records=counts.get('Arcus_clinician_HPO', 0), ctakes_HPO_candidates=counts.get('ctakes_candidate', 0),
                         phenominal_HPO_candidates=counts.get('phenominal_candidate', 0), note_records=note_count, variant_records=variant_count,
                         can_build_image_HPO_review_pair=bool(image_count and counts and not person['reserved'] and not person['identity_conflict']),
                         confirmed_HPO_training_admitted=False, clinical_diagnosis_adjudicated=False))
    with (private / 'patient_master.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    subsets = []
    for subset in ['1', '2', '3', '4', 'unassigned', 'ALL']:
        group = [r for r in rows if subset == 'ALL' or subset in r['subsets'].split('|')]
        subsets.append(dict(subset=subset, patients=len(group), protected_patients=sum(r['protected'] for r in group),
            identity_conflict_patients=sum(r['identity_conflict'] for r in group),
            patients_with_released_QC_faces=sum(bool(r['released_QC_face_images']) for r in group),
            patients_with_structured_HPO=sum(bool(r['structured_HPO_records']) for r in group),
            patients_with_any_HPO_candidates=sum(bool(r['structured_HPO_records'] + r['ctakes_HPO_candidates'] + r['phenominal_HPO_candidates']) for r in group),
            patients_with_notes=sum(bool(r['note_records']) for r in group), patients_with_variants=sum(bool(r['variant_records']) for r in group),
            nonreserved_QC_face_HPO_review_candidates=sum(r['can_build_image_HPO_review_pair'] for r in group)))
    save(WORK / 'coverage.json', dict(state='linked_with_assertion_review_queues', subsets=subsets,
        real_MRN_columns_in_patient_master=False, raw_notes_copied=False, private_mapping_contents_exported=False,
        images_read=False, split_policy='Preserve existing visual patient/group roles; propagate expert/Ian protection; new/unassigned patients remain unassigned',
        source_restrictions=['PubMind and derivatives excluded', 'Final cTAKES/Phenominal tables lack negation/experiencer/time; candidates only',
                             'Clinician HPO table lacks an explicit polarity field; recorded association retained for review, not invented present/absent labels',
                             'Merged note selection and text lineage pending; index only, no training admission',
                             'Genetic record is not automatically a pathogenic molecular diagnosis',
                             'Birth year retained privately; no age at photograph invented from unverified photo dates'],
        private_patient_master=str(private / 'patient_master.tsv'), private_database=str(private / 'linked_metadata.sqlite'), updated_at_utc=now()))
    db.close()
    save(WORK / 'linkage_status.json', dict(state='complete_technical_linkage_clinical_assertions_pending', updated_at_utc=now()))
    print(json.dumps(subsets[-1]))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK / 'linkage_failure.json', dict(error_type=type(error).__name__, updated_at_utc=now()))
        raise

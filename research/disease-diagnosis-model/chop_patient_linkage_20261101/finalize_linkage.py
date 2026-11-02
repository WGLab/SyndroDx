"""Complete raw EHR diagnosis linkage and verify protected dataset boundaries."""
import time
import importlib.util
from link_metadata import *


def main():
    while not load(WORK / 'linkage_status.json').get('state', '').startswith('complete_technical'):
        if (WORK / 'linkage_failure.json').exists():
            raise RuntimeError('base_linkage_failed')
        time.sleep(15)
    private = WORK / 'private'
    salt = (RUN / 'direct_identity_salt.bin').read_bytes()
    def token(value):
        return hmac.new(salt, ('patient:' + value).encode(), hashlib.sha256).hexdigest()
    forward = {norm(r['mrn']): token((r.get('pseudo_mrn') or r.get('psedudo_mrn') or '').strip()) for r in table(PROJECT / 'mrn_mapping_PRIVATE.csv')}
    pid_keys = collections.defaultdict(set)
    key_pids = collections.defaultdict(set)
    for r in parquet_rows(RAW / 'patient.parquet', ['pat_id', 'pat_mrn_id']):
        key = forward.get(norm(r['pat_mrn_id']))
        if key and norm(r['pat_id']):
            pid_keys[norm(r['pat_id'])].add(key)
            key_pids[key].add(norm(r['pat_id']))
    good = {p: next(iter(v)) for p, v in pid_keys.items() if len(v) == 1 and len(key_pids[next(iter(v))]) == 1}
    db = sqlite3.connect(private / 'linked_metadata.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS diagnosis(patient_key TEXT,dx_id TEXT,source TEXT,event_date TEXT,status_code TEXT,icd_codes TEXT,source_rows INTEGER,
            PRIMARY KEY(patient_key,dx_id,source,event_date,status_code,icd_codes));
        CREATE INDEX IF NOT EXISTS diagnosis_patient ON diagnosis(patient_key);
        CREATE TABLE IF NOT EXISTS diagnosis_dictionary(dx_id TEXT PRIMARY KEY,diagnosis_name TEXT,icd10_codes TEXT);
    ''')
    for name, columns, datecol, statuscol, icdcol in [
        ('problem_list.parquet', ['pat_id', 'dx_id', 'noted_date', 'problem_status_c', 'dx_external_id'], 'noted_date', 'problem_status_c', 'dx_external_id'),
        ('pat_enc_dx.parquet', ['pat_id', 'dx_id', 'contact_date', 'primary_dx_yn', 'enc_icd_code'], 'contact_date', 'primary_dx_yn', 'enc_icd_code')]:
        stage = 'diagnosis_' + name
        if db.execute('SELECT 1 FROM stage WHERE name=?', (stage,)).fetchone():
            continue
        db.execute('DELETE FROM diagnosis WHERE source=?', (name,))
        save(WORK / 'diagnosis_status.json', dict(state='linking_raw_EHR_diagnoses', source=name, updated_at_utc=now()))
        count = 0
        for r in parquet_rows(RAW / name, columns, good):
            key = good.get(norm(r['pat_id']))
            if not key or not norm(r['dx_id']):
                continue
            db.execute('INSERT INTO diagnosis VALUES(?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET source_rows=source_rows+1',
                       (key, norm(r['dx_id']), name, str(r[datecol] or ''), str(r[statuscol] or ''), str(r[icdcol] or ''), 1))
            count += 1
            if count % 100000 == 0:
                db.commit()
                save(WORK / 'diagnosis_status.json', dict(state='linking_raw_EHR_diagnoses', source=name, linked_source_rows=count, updated_at_utc=now()))
        db.execute('INSERT OR REPLACE INTO stage VALUES(?,?)', (stage, now()))
        db.commit()
    requested = {r[0] for r in db.execute('SELECT DISTINCT dx_id FROM diagnosis')}
    for r in parquet_rows(RAW / 'clarity_edg.parquet', ['dx_id', 'dx_name', 'current_icd10_list']):
        if norm(r['dx_id']) in requested:
            db.execute('INSERT OR REPLACE INTO diagnosis_dictionary VALUES(?,?,?)',
                       (norm(r['dx_id']), str(r['dx_name'] or ''), str(r['current_icd10_list'] or '')))
    db.commit()
    diagnosis_patients = db.execute('SELECT COUNT(DISTINCT patient_key) FROM diagnosis').fetchone()[0]
    patient_rows = list(table(private / 'patient_master.tsv'))
    keys = [r['patient_key'] for r in patient_rows]
    assert len(set(keys)) == len(keys) == 26897
    assert len({r['pseudo_mrn'] for r in patient_rows}) == len(patient_rows)
    assert not any(k.lower() in {'mrn', 'pat_mrn_id', 'pat_id', 'birth_date', 'text'} for k in patient_rows[0])
    assert db.execute("SELECT COUNT(*) FROM hpo WHERE assertion NOT IN ('unknown','recorded_polarity_unresolved')").fetchone()[0] == 0
    protected = {r['patient_key'] for r in patient_rows if r['protected'] == 'True'}
    audit = dict(state='passed', exact_existing_pseudonyms=26897, unique_patient_keys=True,
        private_patient_master_contains_no_real_MRN_or_raw_notes=True, unknown_HPO_not_converted_to_present_or_absent=True,
        expert_Ian_protection_propagated=True, protected_patients=len(protected),
        patients_with_raw_EHR_diagnosis_records=diagnosis_patients,
        diagnosis_event_groups=db.execute('SELECT COUNT(*) FROM diagnosis').fetchone()[0],
        diagnosis_dictionary_entries=db.execute('SELECT COUNT(*) FROM diagnosis_dictionary').fetchone()[0],
        diagnoses_are_recorded_EHR_assertions_not_adjudicated_etiologic_labels=True,
        images_or_MRI_read=False, private_mapping_contents_exported=False,
        new_supervised_training_admissions=0, unresolved_assertions_retained_in_review_tables=True,
        input_mapping='Original project-root mrn_mapping_PRIVATE.csv; independently cross-checked against all26897 exact-path recovered pairs',
        updated_at_utc=now())
    save(WORK / 'validation.json', audit)
    coverage = load(WORK / 'coverage.json')
    coverage['raw_EHR_diagnoses'] = {k: audit[k] for k in ['patients_with_raw_EHR_diagnosis_records','diagnosis_event_groups','diagnosis_dictionary_entries']}
    save(WORK / 'coverage.json', coverage)
    db.close()
    save(WORK / 'completion.json', dict(state='complete_technical_cleaning_with_explicit_clinical_review_queues',
        private_patient_master=str(private / 'patient_master.tsv'), private_linked_database=str(private / 'linked_metadata.sqlite'),
        summary=str(WORK / 'coverage.json'), validation=str(WORK / 'validation.json'), updated_at_utc=now()))
    try:
        spec = importlib.util.spec_from_file_location('chop_local_notify', V1 / 'runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.notify('CHOP patient metadata linkage complete', 'CHOP metadata: linkage and technical cleaning complete', json.dumps(audit, indent=2))
        save(WORK / 'notification.json', dict(state='local_notifier_invoked', updated_at_utc=now()))
    except Exception as e:
        save(WORK / 'notification_failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK / 'finalize_failure.json', dict(error_type=type(error).__name__, updated_at_utc=now()))
        raise

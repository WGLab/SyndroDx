"""Recover local clinician HPO evidence for train-only review, never print records."""
from link_metadata import *


def main():
    private = WORK / 'private'
    release = private / 'review_release'
    eligible = {r['patient_key'] for r in table(release / 'existing_train_visual_references.tsv')}
    salt = (RUN / 'direct_identity_salt.bin').read_bytes()
    forward = {}
    for r in table(PROJECT / 'mrn_mapping_PRIVATE.csv'):
        pseudo = (r.get('pseudo_mrn') or r.get('psedudo_mrn') or '').strip()
        if not pseudo:
            raise RuntimeError('missing_mapping_pseudonym')
        key = hmac.new(salt, ('patient:' + pseudo).encode(), hashlib.sha256).hexdigest()
        if key in eligible:
            forward[norm(r['mrn'])] = key
    bridge = collections.defaultdict(set)
    for file in sorted((ARCHIVE / 'arcus_2026_06_29_full_ctakes').glob('patient-*.csv')):
        for r in table(file):
            key = forward.get(norm(r.get('pat_mrn_id')))
            if key and norm(r.get('pat_id')):
                bridge[norm(r['pat_id'])].add(key)
    good = {p: next(iter(keys)) for p, keys in bridge.items() if len(keys) == 1}
    db = sqlite3.connect(f'file:{private / "linked_metadata.sqlite"}?mode=ro', uri=True)
    wanted = {(key, code, date) for key, code, date in db.execute("SELECT patient_key,code,event_date FROM hpo WHERE source='Arcus_clinician_HPO'") if key in eligible}
    db.close()
    counts = collections.Counter()
    patients, matched = set(), set()
    source_receipts = []
    destination = release / 'clinician_HPO_evidence.private.jsonl'
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w') as out:
        for source in sorted((ARCHIVE / 'arcus_2026_06_29').glob('clinician_hpo_element*')):
            source_receipts.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            for index, r in enumerate(table(source)):
                key = good.get(norm(r.get('pat_id')))
                if not key:
                    continue
                date = (r.get('creation_date') or '').strip()
                codes = sorted(c for c in set(re.findall(r'HP:\d{7}', r.get('hpo_code', ''))) if (key, c, date) in wanted)
                if not codes:
                    continue
                context = (r.get('notes') or '').strip()
                item = dict(patient_key=key, codes=codes, event_date=date, source=str(source), source_data_row_zero_based=index,
                            clinician_notes_local_only=context, source_hpo_version=r.get('hpo_version') or '',
                            assertion='recorded_polarity_unresolved', training_admitted=False,
                            review_required=['assertion_and_experiencer', 'clinical_time', 'source_provenance'])
                out.write(json.dumps(item) + '\n')
                counts['source_records'] += 1
                counts['records_with_context' if context else 'records_without_context'] += 1
                patients.add(key)
                matched.update((key, c, date) for c in codes)
    temporary.replace(destination)
    audit = dict(state='clinician_evidence_prepared_for_local_review', patients=len(patients), counts=dict(counts),
                 target_patient_HPO_date_groups=len(wanted), matched_patient_HPO_date_groups=len(matched),
                 unmatched_patient_HPO_date_groups=len(wanted - matched), sources=source_receipts,
                 training_admissions=0, data_location='HPC only; patient rows and evidence not emitted', updated_at_utc=now())
    save(WORK / 'clinician_evidence_audit.json', audit)
    print(json.dumps(audit))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(WORK / 'clinician_evidence_failure.json', dict(error_type=type(error).__name__, updated_at_utc=now()))
        raise

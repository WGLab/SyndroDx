"""Release source-recorded patient associations, never fabricate HPO polarity."""
import csv
import collections
import numpy as np
from common import *


def table(path):
    with Path(path).open() as f:
        yield from csv.DictReader(f, delimiter='\t')


def main():
    os.umask(0o077)
    source = PROJECT / 'chop_patient_linkage_20261101'
    release = source / 'private/review_release'
    audit = load(source / 'review_release_audit.json')
    assert audit['emitted_train_protected_overlap'] == 0
    assert audit['emitted_train_identity_or_group_conflicts'] == 0
    evidence = load(source / 'clinician_evidence_audit.json')
    assert evidence['unmatched_patient_HPO_date_groups'] == 0
    for item in evidence['sources']:
        assert digest(item['path']) == item['sha256']
    patients = {r['patient_key']: r for r in table(release / 'patient_disposition.tsv')}
    term_ids = load(WORK / 'private/term_index.json')['term_ids']
    terms = {v: i + 1 for i, v in enumerate(term_ids)}
    groups = collections.defaultdict(set)
    group_patients = collections.defaultdict(set)
    for line in (release / 'clinician_HPO_evidence.private.jsonl').open():
        r = json.loads(line)
        p = patients[r['patient_key']]
        assert p['disposition'] == 'train_existing_split' and p['split'] == 'train' and not p['reasons']
        assert r['assertion'] == 'recorded_polarity_unresolved' and not r['training_admitted']
        groups[p['group_id']].update(c for c in r['codes'] if c in terms)
        group_patients[p['group_id']].add(r['patient_key'])
    candidates = [r for r in table(release / 'existing_train_visual_references.tsv') if r['group_id'] in groups]
    # Keep exactly the visual population already covered by the GMDB overlap audit.
    stage = PROJECT / 'transface_arcface_identity_cpt_20261026'
    spec = load(stage / 'protocol.json')['train_manifest']
    assert digest(spec['path']) == spec['sha256']
    audited_hashes = {r['crop_sha256'] for r in table(spec['path'])}
    assert load(MATCH / 'audit.json')['state'] == 'passed'
    excluded = {r['group_id'] for r in candidates if r['crop_sha256'] not in audited_hashes}
    rows = [r for r in candidates if r['group_id'] not in excluded]
    keys = sorted({r['group_id'] for r in rows if groups[r['group_id']]})
    index = {k: i for i, k in enumerate(keys)}
    ids, states = np.zeros((len(keys), 128), np.int64), np.zeros((len(keys), 128), np.int64)
    truncated = 0
    for k, i in index.items():
        codes = sorted(groups[k])
        truncated += len(codes) > 128
        values = [terms[c] for c in codes[:128]]
        ids[i, :len(values)] = values
        states[i, :len(values)] = 2  # Explicitly recorded/unverified, never assumed present.
    images = []
    seen = set()
    for r in rows:
        if r['group_id'] in index and r['crop_sha256'] not in seen:
            seen.add(r['crop_sha256'])
            images.append(dict(patient_index=index[r['group_id']], crop_path=r['crop_path'], crop_sha256=r['crop_sha256']))
    assert len(keys) >= 50 and np.all(ids.any(axis=1))
    save(WORK / 'private/chop_images.json', dict(images=images))
    save(WORK / 'private/chop_groups.json', dict(groups=keys))
    np.savez(WORK / 'private/chop_terms.npz', ids=ids, states=states)
    old = PROJECT / 'hpo_fusion_staged_20261031'
    original = load(old / 'data_receipt.json')
    for n in ['gmdb.npz', 'public_cases.npz', 'profiles.npz', 'term_index.json']:
        assert digest(WORK / 'private' / n) == original['files'][n]
    receipt = dict(state='weak_patient_association_release', patient_groups=len(keys), images=len(images),
        patients_before_group_merging=evidence['patients'], excluded_groups_outside_prior_visual_overlap_audit=len(excluded),
        truncated_groups=truncated, confirmed_positive_HPO_labels_created=0, disease_labels_created=0,
        assertion_state_2='clinician-recorded concept with unresolved polarity',
        purpose='patient-level cross-modal association experiment only, not positive phenotype prediction',
        metadata_sources='direct Arcus clinician HPO only; no NLP/PubMind/merged note data admitted',
        source_files={str(f):digest(f) for f in [release / 'clinician_HPO_evidence.private.jsonl',release / 'patient_disposition.tsv',release / 'existing_train_visual_references.tsv']},
        known_protected_overlap=0, GMDB_overlap_audit_reused=True, BCH_identity_overlap_unresolved=True,
        unresolved_limitations=['No same-visit pairing claim', 'Unknown family and cross-institution aliases are not certified'],
        counts=original['counts'], files={n:digest(WORK / 'private' / n) for n in ['gmdb.npz','public_cases.npz','profiles.npz','term_index.json','chop_terms.npz','chop_images.json','chop_groups.json']}, updated_at_utc=now())
    save(WORK / 'data_receipt.json', receipt)
    print(json.dumps({k:receipt[k] for k in ['state','patient_groups','images','confirmed_positive_HPO_labels_created','disease_labels_created']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(WORK / 'preparation_failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))
        raise

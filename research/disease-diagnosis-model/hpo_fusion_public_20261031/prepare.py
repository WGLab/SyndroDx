"""Prepare patient-isolated arrays locally; emit aggregate receipts only."""
import collections
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from common import *

MAX_TERMS = 128


def hpos(value):
    return sorted(set(re.findall(r'HP:\d{7}', str(value))))


def disease(value):
    s = str(value).upper().replace('ORPHANET', 'ORPHA')
    hit = re.fullmatch(r'(OMIM|ORPHA)[:_](\d+)', s)
    return hit.group(1) + ':' + hit.group(2) if hit else None


def pmids(value):
    return set(re.findall(r'(?<!\d)\d{6,9}(?!\d)', str(value)))


def signature(pos, neg=()):
    return hashlib.sha256(json.dumps([sorted(set(pos)), sorted(set(neg))]).encode()).hexdigest()


def pack(positive, negative, lookup):
    assert not set(positive) & set(negative)
    items = [(lookup[x], 0) for x in sorted(set(positive)) if x in lookup]
    items += [(lookup[x], 1) for x in sorted(set(negative)) if x in lookup]
    ids = np.zeros(MAX_TERMS, np.int64)
    states = np.zeros(MAX_TERMS, np.int64)
    for i, (term, state) in enumerate(items[:MAX_TERMS]):
        ids[i], states[i] = term, state
    return ids, states, max(0, len(items) - MAX_TERMS)


def main():
    os.umask(0o077)
    if load(WORK / 'data_receipt.json').get('state') == 'complete':
        return
    selection = load(WORK / 'visual_selection.json')
    assert digest(selection['path']) == selection['sha256']
    case = selection['case']
    receipt = load(MATCH / 'inference' / (case + '.json'))
    feature_path = MATCH / 'private' / (case + '_features.npy')
    assert digest(feature_path) == receipt['sha256']
    original_path = MATCH / 'private/init__transface_chop_cpt_features.npy'
    assert digest(original_path) == load(MATCH / 'inference/init__transface_chop_cpt.json')['sha256']
    feature = np.load(feature_path, allow_pickle=False)
    original = np.load(original_path, allow_pickle=False)
    source_rows = [json.loads(line) for line in (SHARED / 'inputs.private.jsonl').open()]
    lookup = {r['image_id']: i for i, r in enumerate(source_rows)}
    assert len(lookup) == len(source_rows) == len(feature) == len(original)
    assert np.isfinite(feature).all() and np.isfinite(original).all()
    image_rows = load(PROJECT / 'gmdb_supervised_transface_20261026/private/records.private.json')['rows']
    by_index = {r['array_index']: r for r in image_rows}
    ontology = load(OLD / 'private/ontology.json')['terms']
    term_ids = sorted(ontology)
    terms = {h: i + 1 for i, h in enumerate(term_ids)}
    # Reuse previously audited genuine HPO pairs; apply the latest common quarantine too.
    previous = load(OLD / 'data_receipt.json')
    assert digest(OLD / 'private/patients.json') == previous['patient_file_sha256']
    blocked = set(load(MATCH / 'private/blocked_patients.json')['subjects'])
    train_subjects, val_subjects, test_subjects = set(), set(), set()
    for name, target in [('gmdb_train_images', train_subjects), ('gmdb_val_images', val_subjects),
                         ('gmdb_frequent_test_images', test_subjects), ('gmdb_rare_test_images', test_subjects)]:
        q = SHARED / 'split_snapshot' / (name + '.csv')
        with q.open() as f:
            target.update(r['subject'] for r in csv.DictReader(f))
    train_subjects -= test_subjects | blocked
    val_subjects -= test_subjects | blocked
    assert not train_subjects & val_subjects
    paired, vectors, raw_vectors, ids, states = [], [], [], [], []
    truncated = 0
    for p in load(OLD / 'private/patients.json')['patients']:
        allowed = train_subjects if p['split'] == 'train' else val_subjects
        if p['subject'] not in allowed:
            continue
        indices = [lookup[by_index[i]['image_id']] for i in p['image_indices']]
        assert all(source_rows[i]['subject'] == p['subject'] for i in indices)
        if not indices:
            continue
        vector = feature[indices].mean(0)
        raw = original[indices].mean(0)
        vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
        raw_vectors.append(raw / max(np.linalg.norm(raw), 1e-12))
        ii, ss, lost = pack(p['all_present_hpo'], p['absent_hpo'], terms)
        ids.append(ii)
        states.append(ss)
        truncated += bool(lost)
        paired.append(p)
    classes = sorted({p['label'] for p in paired if p['split'] == 'train'})
    class_map = {v: i for i, v in enumerate(classes)}
    train = np.array([i for i, p in enumerate(paired) if p['split'] == 'train'])
    validation = np.array([i for i, p in enumerate(paired) if p['split'] == 'validation' and p['label'] in class_map])
    np.savez(WORK / 'private/gmdb.npz', features=np.stack(vectors), original_features=np.stack(raw_vectors),
             ids=np.stack(ids), states=np.stack(states), labels=np.array([class_map.get(p['label'], -1) for p in paired]),
             train=train, validation=validation)
    save(WORK / 'private/patient_index.json', dict(patients=[dict(subject=p['subject'], label=p['label'], split=p['split']) for p in paired]))

    metadata = Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata')
    with (metadata / 'gmdb_syndromes_v1.1.0.tsv').open() as f:
        syndromes = list(csv.DictReader(f, delimiter='\t'))
    syndrome_omim = {}
    for s in syndromes:
        values = sorted(set(re.findall(r'\d{6}', s['OMIM'])))
        if len(values) == 1:
            syndrome_omim[s['syndrome_id']] = 'OMIM:' + values[0]
    class_diseases = {i: syndrome_omim[c] for c, i in class_map.items() if c in syndrome_omim}

    # Public test cohorts are used only as exclusion inventories, never for gradient updates.
    rarebench_signatures, rarebench_positive_signatures = set(), set()
    benchmark_count = 0
    with zipfile.ZipFile(WORK / 'public/rarebench_data.zip') as z:
        for name in z.namelist():
            if not name.endswith('.jsonl'):
                continue
            for line in z.read(name).splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                positive = hpos(r['Phenotype'])
                rarebench_signatures.add(signature(positive))
                rarebench_positive_signatures.add(tuple(positive))
                benchmark_count += 1

    blocked_publications, all_gmdb_signatures = set(), set()
    with (metadata / 'image_metadata_v1.1.0.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            blocked_publications |= pmids(r.get('pmid', ''))
            positive = hpos(r.get('present_features', ''))
            if positive:
                all_gmdb_signatures.add(tuple(positive))

    # Rare GMDB disease labels are excluded from public case supervision too.
    rare_labels = set()
    with (SHARED / 'split_snapshot/gmdb_rare_test_images.csv').open() as f:
        rare_labels.update(r['label'] for r in csv.DictReader(f))
    prohibited_diseases = {syndrome_omim[x] for x in rare_labels if x in syndrome_omim}
    public = []
    exclusions = collections.Counter()
    seen = set()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)

    with zipfile.ZipFile(WORK / 'public/phenopackets_0.1.27.zip') as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.json'):
                continue
            d = json.loads(z.read(name))
            diagnoses = {disease(v.get('term', {}).get('id', '')) for v in d.get('diseases', [])} - {None}
            if len(diagnoses) != 1:
                exclusions['non_unique_or_unsupported_diagnosis'] += 1
                continue
            label = next(iter(diagnoses))
            if label in prohibited_diseases:
                exclusions['GMDB_rare_disease_supervision_excluded'] += 1
                continue
            positive, negative = set(), set()
            for v in d.get('phenotypicFeatures', []):
                h = v.get('type', {}).get('id')
                if h in terms:
                    (negative if v.get('excluded', False) else positive).add(h)
            if len(positive) < 3 or positive & negative:
                exclusions['insufficient_or_conflicting_HPO'] += 1
                continue
            refs = d.get('metaData', {}).get('externalReferences', [])
            refs_pmids = set()
            for r in refs:
                value = r.get('id', '') + ' ' + r.get('reference', '')
                for token in re.findall(r'(?:PMID[: /]|pubmed[.]ncbi[.]nlm[.]nih[.]gov/)(\d+)', value, flags=re.I):
                    refs_pmids.add(token)
            if refs_pmids & blocked_publications:
                exclusions['GMDB_publication_overlap'] += 1
                continue
            sig = tuple(sorted(positive))
            if sig in rarebench_positive_signatures or sig in all_gmdb_signatures:
                exclusions['GMDB_or_RareBench_exact_positive_HPO_signature'] += 1
                continue
            key = (label, signature(positive, negative))
            if key in seen:
                exclusions['duplicate_diagnosis_and_HPO'] += 1
                continue
            seen.add(key)
            groups = sorted('PMID:' + s for s in refs_pmids) or ['cohort:' + '/'.join(name.split('/')[:-1])]
            for g in groups:
                union(groups[0], g)
            public.append(dict(label=label, positive=sorted(positive), negative=sorted(negative), group=groups[0]))
    for p in public:
        p['group'] = find(p['group'])
        bucket = int(hashlib.sha256(('hpo-public-v1:' + p['group']).encode()).hexdigest()[:8], 16) % 5
        p['split'] = 'validation' if bucket == 0 else 'train'
    assert not {p['group'] for p in public if p['split'] == 'train'} & {p['group'] for p in public if p['split'] == 'validation'}
    public_labels = sorted({p['label'] for p in public if p['split'] == 'train'})
    public_map = {d: i for i, d in enumerate(public_labels)}
    public_ids, public_states = [], []
    for p in public:
        ii, ss, _ = pack(p['positive'], p['negative'], terms)
        public_ids.append(ii)
        public_states.append(ss)
    if len(public_labels) < 10 or sum(p['split'] == 'train' for p in public) < 100:
        raise RuntimeError('insufficient_audited_public_cases')
    np.savez(WORK / 'private/public_cases.npz', ids=np.stack(public_ids), states=np.stack(public_states),
             labels=np.array([public_map.get(p['label'], -1) for p in public]),
             train=np.array([i for i, p in enumerate(public) if p['split'] == 'train']),
             validation=np.array([i for i, p in enumerate(public) if p['split'] == 'validation' and p['label'] in public_map]))
    save(WORK / 'private/public_case_index.json', dict(cases=public, labels=public_labels))

    profiles = collections.defaultdict(set)
    with (PROJECT / 'cranioface_qwen_alignment_20261027/public/phenotype_associations.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            label = disease(r['disease_id'])
            if label and r['hpo_id'] in terms and str(r.get('qualifier', '')).upper() != 'NOT':
                profiles[label].add(r['hpo_id'])
    orphanet_file = WORK / 'public/orphadata_phenotypes.xml'
    orphanet_loaded = False
    if orphanet_file.exists():
        tree = ET.parse(orphanet_file)
        for disorder in tree.findall('.//Disorder'):
            code = disorder.findtext('OrphaCode')
            if not code:
                continue
            for assoc in disorder.findall('.//HPODisorderAssociation'):
                frequency = (assoc.findtext('./HPOFrequency/Name') or '').lower()
                if frequency.startswith('excluded'):
                    continue
                h = assoc.findtext('./HPO/HPOId')
                if h in terms:
                    profiles['ORPHA:' + code].add(h)
        orphanet_loaded = True
    label_order = classes + public_labels
    profile_ids, profile_states, available = [], [], []
    for i, lab in enumerate(label_order):
        label = class_diseases.get(i) if i < len(classes) else lab
        positive = profiles.get(label, set())
        ii, ss, _ = pack(positive, [], terms)
        profile_ids.append(ii)
        profile_states.append(ss)
        available.append(bool(positive))
    np.savez(WORK / 'private/profiles.npz', ids=np.stack(profile_ids), states=np.stack(profile_states), available=np.array(available))
    save(WORK / 'private/term_index.json', dict(term_ids=term_ids))
    texts = ['[No phenotype]'] + [ontology[h]['name'] + '. Parent concepts: ' + '; '.join(ontology[p]['name'] for p in ontology[h]['parents'] if p in ontology) for h in term_ids]
    save(WORK / 'private/term_texts.json', dict(texts=texts))
    save(WORK / 'data_receipt.json', dict(state='complete', counts=dict(
        GMDB_train_patients=len(train), GMDB_validation_patients=len(validation), GMDB_classes=len(classes),
        public_train_cases=sum(p['split'] == 'train' for p in public),
        public_validation_cases=sum(p['split'] == 'validation' for p in public),
        public_closed_set_validation_cases=sum(p['split'] == 'validation' and p['label'] in public_map for p in public),
        public_training_diseases=len(public_labels), reserved_RareBench_cases=benchmark_count,
        GMDB_classes_with_disease_profile=sum(available[:len(classes)]),
        public_training_diseases_with_profile=sum(available[len(classes):]),
        HPO_terms=len(term_ids), truncated_GMDB_patients=truncated),
        public_exclusions=dict(exclusions), public_PMID_groups_disjoint=True,
        shared_GMDB_patient_split=True, CHOP_expert_training=False, BCH_training=False,
        test_case_gradient_updates=False, private_MRN_mapping_read=False,
        exact_HPO_matches_are_conservative_flags_not_identity_proof=True,
        public_case_identity_alias_audit_complete=False,
        orphanet_loaded=orphanet_loaded,
        features='frozen native512 patient-mean features from matched pinned inference',
        files={name: digest(WORK / 'private' / name) for name in ['gmdb.npz', 'public_cases.npz', 'profiles.npz', 'term_index.json', 'term_texts.json']},
        updated_at_utc=now()))
    print(json.dumps(load(WORK / 'data_receipt.json')['counts']))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Conservative, reproducible MONDO crosswalks for group and raw image labels.

Group candidates are never inherited by images. Outputs annotate source labels,
not confirmed patient diagnoses. Existing labels, splits and patients are untouched.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from ontology import Ontology, normstr

HERE = Path(__file__).resolve().parent

# Whole-label aliases only, taken from the existing taxonomy and observed labels.
# No substring inference, no choice of the first candidate, no default NF1.
ALIASES = {
    'MONDO:0007041': ['aperts', 'aperts syn', 'aperte'],
    'MONDO:0007405': ['crouzon', 'crouzons'],
    'MONDO:0007043': ['pfeiffer', 'pfeiffer syn', 'pfeiffers'],
    'MONDO:0007042': ['saethre chotzen'],
    'MONDO:0011274': ['muenke syn'],
    'MONDO:0002457': ['treacher collins'],
    'MONDO:0015397': ['hfm', 'goldenhar', 'goldenhar syn', 'goldenhars syn',
                      'bifac microsomia'],
    'MONDO:0018923': ['22q 11.2 del', '22q11.2 del', '22q11.2ds'],
    'MONDO:0007534': ['beckwith wiedeman', 'beckwith weideman'],
    'MONDO:0007710': ['parry romberg', 'perry rombergs', 'perry romberg'],
    'MONDO:0008006': ['moebius', 'mobius', 'moebius syn'],
    'MONDO:0021061': ['nf'],
    'MONDO:0010570': ['cfnd', 'cfn dysplasia'],
    'MONDO:0002378': ['dermoid'],
    'MONDO:0016044': ['uclp', 'uclcp', 'lclp', 'lclcp', 'bclp', 'clp'],
    'MONDO:0004747': ['cl', 'bcl', 'bilat cleft lip', 'unilateral cleft lip'],
    'MONDO:0016064': ['cp'],
}

# Even an EXACT ontology synonym can be less specific than the target label.
# These require review rather than automatic assignment.
BLOCKED = {
    'MONDO:0009869': 'Target specifies isolated Pierre-Robin; source must establish that qualifier.',
    'MONDO:0044792': 'Target specifies large congenital nevus; source must establish size and type.',
    'MONDO:0008583': 'Target specifies inherited torticollis; source must establish inheritance.',
    'MONDO:0008094': 'Target specifies familial multiple nevi; port-wine stain alone is insufficient.',
    'MONDO:0009331': 'Target specifies isolated hemihyperplasia; source must establish isolated status.',
    'MONDO:0007340': 'Target specifies cleidocranial dysplasia type 1; subtype needs review.',
    'MONDO:0007233': 'Target specifies second branchial cleft; source must establish branchial origin.',
}

# Explicit synostosis may map to the broader disease, retaining its suture HPO.
# Shape alone (scaphocephaly, metopic ridge) is deliberately excluded.
SUTURE_ALIASES = {
    'HP:0004442': ['sagittal syn', 'sagittal synostosis', 'sag syn', 'sagittal craniosynostosis'],
    'HP:0011330': ['metopic syn', 'met synostosis', 'metopic synostosis', 'metopic craniosynostosis'],
    'HP:0004440': ['bicoronal synostosis', 'coronal synostosis', 'bicoronal syn',
                   'unicoronal craniosyn', 'coronal syn', 'coronal craniosynostosis'],
    'HP:0004443': ['lambdoid syn', 'lambdoid synostosis'],
    '': ['multi suture synostosis', 'pancraniosynostosis', 'pansynostosis',
         'squamosal syn', 'mercedes synostosis'],
}
SUTURE_LOOKUP = {normstr(s): hp for hp, names in SUTURE_ALIASES.items() for s in names}

# These aliases contain laterality that the MONDO target does not preserve.
BROAD_ALIASES = {normstr(s) for s in [
    'uclp', 'uclcp', 'lclp', 'lclcp', 'bclp', 'bcl',
    'bilat cleft lip', 'unilateral cleft lip',
]}


def read_tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def exact_index(terms, prefix):
    index = collections.defaultdict(set)
    for t in terms.values():
        if t.obsolete or not t.id.startswith(prefix) or 'not_a_disease' in t.subsets:
            continue
        for name in [t.name] + [s for s, scope in t.synonyms if scope == 'EXACT']:
            if name:
                index[normstr(name)].add(t.id)
    return index


def normalize_raw(raw, mondo_index, hpo_index, aliases):
    key = normstr(raw)
    hp = ';'.join(sorted(hpo_index.get(key, ())))
    candidates = set(mondo_index.get(key, ()))
    alias = aliases.get(key)
    if alias:
        candidates.add(alias)
    base = dict(mondo_id='', candidate_mondo_ids='', hpo_ids=hp,
                mapping_relation='', evidence='', notes='')
    if key in SUTURE_LOOKUP:
        return dict(base, status='mapped', mondo_id='MONDO:0015469',
                    hpo_ids=SUTURE_LOOKUP[key] or hp, mapping_relation='skos:broadMatch',
                    evidence='curated_whole_label_alias',
                    notes='Suture subtype retained in raw label/HPO; MONDO target is broader.')
    if key == 'nevus':
        return dict(base, status='needs_review', candidate_mondo_ids='MONDO:0005073',
                    notes='Unspecified nevus; confirm melanocytic rather than another nevus type.')
    if len(candidates) == 1:
        mid = next(iter(candidates))
        if mid in BLOCKED:
            return dict(base, status='needs_review', candidate_mondo_ids=mid,
                        notes=BLOCKED[mid])
        return dict(base, status='mapped', mondo_id=mid,
                    mapping_relation='skos:broadMatch' if key in BROAD_ALIASES else 'skos:exactMatch',
                    evidence='curated_whole_label_alias' if alias else 'ontology_label_or_exact_synonym')
    if candidates:
        return dict(base, status='needs_review', candidate_mondo_ids=';'.join(sorted(candidates)),
                    notes='Multiple MONDO matches; no candidate selected.')
    return dict(base, status='hpo_only' if hp else 'unresolved',
                evidence='ontology_label_or_exact_synonym' if hp else '',
                notes='' if hp else 'Needs original-label curation; group candidates are not inherited.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=HERE)
    ap.add_argument('--out-dir', type=Path, default=HERE / 'mondo_normalization')
    args = ap.parse_args()
    o = Ontology()
    mi = exact_index(o.mondo, 'MONDO:')
    hi = exact_index(o.hp, 'HP:')
    assert o.mondo['MONDO:0015469'].name == 'craniosynostosis'
    for hid in filter(None, SUTURE_ALIASES):
        assert hid in o.hp and not o.hp[hid].obsolete, hid
    aliases = {}
    for mid, names in ALIASES.items():
        assert mid in o.mondo and not o.mondo[mid].obsolete, mid
        for name in names:
            key = normstr(name)
            assert key not in aliases or aliases[key] == mid, name
            aliases[key] = mid
    policies = read_tsv(HERE / 'mondo_group_policy.tsv')
    patients = read_tsv(args.data_dir / 'patients.tsv')
    images = read_tsv(args.data_dir / 'labels.tsv')
    counts = collections.Counter(r['primary_group'] for r in patients if r['primary_group'])
    assert set(counts) == {r['group'] for r in policies}, 'Group policy coverage differs from patient labels'
    assert len(policies) == len(counts), 'Duplicate group policies'
    by_group = {r['group']: r for r in policies}
    for r in policies:
        for field, terms in [('mondo_id', o.mondo), ('candidate_mondo_ids', o.mondo), ('hpo_ids', o.hp)]:
            for tid in filter(None, r[field].split(';')):
                assert tid in terms and not terms[tid].obsolete, (r['group'], tid)
        assert (r['status'] == 'mapped') == bool(r['mondo_id']), r
        r['mondo_name'] = o.name_of(r['mondo_id']) if r['mondo_id'] else ''
        r['candidate_mondo_names'] = ';'.join(o.name_of(i) for i in r['candidate_mondo_ids'].split(';') if i)
        r['primary_patient_count'] = counts[r['group']]

    # Retain all raw DX labels, including rare groups removed from image taxonomy.
    raw_counts = collections.Counter(r['raw_label'] for r in images if r['bucket'] == 'DX')
    raw_patients = collections.defaultdict(set)
    raw_groups = collections.defaultdict(set)
    for r in images:
        if r['bucket'] == 'DX':
            if r['mrn']:
                raw_patients[r['raw_label']].add(r['mrn'])
            if r['dx_group']:
                raw_groups[r['raw_label']].add(r['dx_group'])
    raw_rows = []
    for raw, n in sorted(raw_counts.items(), key=lambda x: (-x[1], x[0])):
        result = normalize_raw(raw, mi, hi, aliases)
        candidates = set(filter(None, result['candidate_mondo_ids'].split(';')))
        if result['status'] == 'unresolved':
            for group in raw_groups[raw]:
                p = by_group[group]
                candidates.update(filter(None, (p['mondo_id'] + ';' + p['candidate_mondo_ids']).split(';')))
            result['candidate_mondo_ids'] = ';'.join(sorted(candidates))
        raw_rows.append(dict(raw_label=raw, image_count=n,
                             patient_count=len(raw_patients[raw]),
                             source_groups=';'.join(sorted(raw_groups[raw])),
                             **result, mondo_name=o.name_of(result['mondo_id']) or ''))
    lookup = {r['raw_label']: r for r in raw_rows}
    concept_images = collections.Counter()
    concept_patients = collections.defaultdict(set)
    concept_labels = collections.defaultdict(set)
    for r in images:
        if r['bucket'] != 'DX':
            continue
        mid = lookup[r['raw_label']]['mondo_id']
        if mid:
            concept_images[mid] += 1
            if r['mrn']:
                concept_patients[mid].add(r['mrn'])
            concept_labels[mid].add(r['raw_label'])
    concepts = [dict(mondo_id=mid, mondo_name=o.name_of(mid), image_count=n,
                     patient_count=len(concept_patients[mid]),
                     raw_labels=' | '.join(sorted(concept_labels[mid])))
                for mid, n in sorted(concept_images.items(), key=lambda x: (-x[1], x[0]))]
    sidecar = []
    for idx, r in enumerate(images):
        mapped = lookup[r['raw_label']] if r['bucket'] == 'DX' else {}
        sidecar.append(dict(source_row=idx, img_id=r['img_id'],
                            mondo_id=mapped.get('mondo_id', ''),
                            mondo_name=mapped.get('mondo_name', ''),
                            status=mapped.get('status', 'not_diagnostic'),
                            mapping_relation=mapped.get('mapping_relation', ''),
                            hpo_ids=mapped.get('hpo_ids', '')))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / 'group_mondo_map.tsv', policies, list(policies[0]))
    write_tsv(args.out_dir / 'raw_label_mondo_map.tsv', raw_rows, list(raw_rows[0]))
    write_tsv(args.out_dir / 'raw_label_review.tsv',
              [r for r in raw_rows if r['status'] != 'mapped'], list(raw_rows[0]))
    write_tsv(args.out_dir / 'image_mondo_annotations.tsv', sidecar, list(sidecar[0]))
    write_tsv(args.out_dir / 'mondo_concept_counts.tsv', concepts,
              ['mondo_id', 'mondo_name', 'image_count', 'patient_count', 'raw_labels'])
    provenance = {}
    for path in [HERE / 'ontology/mondo.obo', HERE / 'ontology/hp.obo',
                 HERE / 'mondo_group_policy.tsv', Path(__file__).resolve(),
                 args.data_dir / 'labels.tsv', args.data_dir / 'patients.tsv']:
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        item = dict(sha256=digest.hexdigest())
        if path.suffix == '.obo':
            with open(path) as f:
                for line in f:
                    if line.startswith('data-version:'):
                        item['release'] = line.strip().partition(': ')[2]
                        break
        provenance[str(path)] = item
    summary = dict(
        group_status_counts=dict(collections.Counter(r['status'] for r in policies)),
        unique_group_mondo_ids=len({r['mondo_id'] for r in policies if r['mondo_id']}),
        raw_label_status_counts=dict(collections.Counter(r['status'] for r in raw_rows)),
        image_status_counts=dict(collections.Counter(r['status'] for r in sidecar)),
        unique_raw_mondo_ids=len({r['mondo_id'] for r in raw_rows if r['mondo_id']}),
        provenance=provenance,
        scope='Source-label normalization only, not confirmed patient diagnoses. '
              'Image sidecar joins by zero-based source_row for the SHA256-pinned labels.tsv. '
              'HPO synonym matches are auxiliary candidates, not a curated phenotype gold standard. '
              'Counts of MONDO IDs are not a finalized mutually exclusive disease taxonomy.')
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'provenance'}, indent=2))


if __name__ == '__main__':
    main()

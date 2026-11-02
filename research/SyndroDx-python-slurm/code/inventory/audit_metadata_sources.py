#!/usr/bin/env python3
"""Read-only metadata audit. Reports contain aggregate counts, not patient IDs."""
import collections
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path('/controlled-data/craniofacial_data')
OUT = Path(__file__).resolve().parent / 'metadata_audit'
FRONT = 'a photo of the full human face from the front'
FILES = {
    'frontal': 'craniofacial_metadata_frontal_updated_123025_deduplicated.csv',
    'full_dec': 'complete_full_craniofacial_metadata_updated_123025_deduplicated.csv',
    'full_mar13': 'complete_full_craniofacial_metadata_031326.tsv',
    'full_mar15': 'complete_full_craniofacial_metadata_updated_03152026.tsv',
    'merged_mrns': 'chop_merged_mrns.csv',
}
MISSING = {'', 'nan', 'none', 'null', 'na', 'n/a'}


def clean(x):
    x = (x or '').strip()
    return '' if x.lower() in MISSING else x


def mrnkey(x):
    x = clean(x)
    if re.fullmatch(r'\d+\.0', x):
        x = x[:-2]
    return (x.lstrip('0') or '0') if x.isdigit() else x


def load(name):
    path = ROOT / name
    by_path = collections.defaultdict(list)
    counts = collections.Counter()
    missing = collections.Counter()
    ids = set()
    frontal_ids = set()
    frontal_nonempty = collections.Counter()
    with path.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fields = reader.fieldnames
        for r in reader:
            for k in fields:
                if not clean(r[k]):
                    missing[k] += 1
            image = clean(r.get('image_path'))
            organ = clean(r.get('EVA02_Organ'))
            mrn = mrnkey(r.get('norm_mrn') or r.get('mrn'))
            rec = dict(mrn=mrn, organ=organ, crop=clean(r.get('cropping_path')),
                       dx=clean(r.get('disease_or_phenotypes')), icd=clean(r.get('ICD10')),
                       date=clean(r.get('date_taken')),
                       duplicate=clean(r.get('duplication_remove')),
                       mrn_fields={k: mrnkey(r.get(k)) for k in ['norm_mrn','mrn','MRN','chart_mrn'] if k in fields})
            by_path[image].append(rec)
            counts[organ] += 1
            if mrn:
                ids.add(mrn)
            if organ == FRONT:
                if mrn:
                    frontal_ids.add(mrn)
                for key in ['mrn','crop','dx','icd','date']:
                    if rec[key]:
                        frontal_nonempty[key] += 1
    total = sum(counts.values())
    dups = collections.Counter(r['duplicate'] for rs in by_path.values() for r in rs)
    return by_path, dict(file=str(path), size_bytes=path.stat().st_size,
                         modified_time_ns=path.stat().st_mtime_ns, delimiter='tab', rows=total,
                         columns=fields, missing_counts=dict(missing),
                         distinct_image_paths=len(by_path) - int('' in by_path),
                         repeated_path_extra_rows=sum(len(v)-1 for v in by_path.values()),
                         unique_mrns=len(ids), frontal_unique_mrns=len(frontal_ids),
                         organ_counts=dict(counts), frontal_nonempty_fields=dict(frontal_nonempty),
                         duplication_remove_counts=dict(dups)), ids


def compare(left, right):
    common = set(left) & set(right) - {''}
    stats = collections.Counter()
    transitions = collections.Counter()
    for path in common:
        a, b = left[path], right[path]
        if len(a) != 1 or len(b) != 1:
            stats['nonunique_join_paths'] += 1
            continue
        a, b = a[0], b[0]
        for key in ['mrn', 'organ', 'crop', 'dx', 'icd', 'date']:
            if a[key] and b[key]:
                stats[key + '_both_present'] += 1
                if a[key] != b[key]:
                    stats[key + ('_string_disagree' if key == 'date' else '_disagree')] += 1
            elif a[key] and not b[key]:
                stats[key + '_only_left'] += 1
            elif b[key] and not a[key]:
                stats[key + '_only_right'] += 1
        if a['organ'] != b['organ']:
            transitions[a['organ'] + ' -> ' + b['organ']] += 1
    return dict(common_paths=len(common), only_left_paths=len(set(left)-set(right)),
                only_right_paths=len(set(right)-set(left)), fields=dict(stats),
                organ_transitions=dict(transitions))


def sample_files(records, field, n=100):
    values = sorted({r[field] for rs in records.values() for r in rs if r[field]}) if field != 'image_path' else sorted(p for p in records if p)
    picks = random.Random(20260928).sample(values, min(n,len(values)))
    statuses = collections.Counter()
    for value in picks:
        try:
            statuses['exists' if Path(value).is_file() else 'missing'] += 1
        except OSError:
            statuses['os_error'] += 1
    return dict(sample_size=len(picks), total_unique_paths=len(values), counts=dict(statuses))


def main():
    csv.field_size_limit(10**8)
    OUT.mkdir(exist_ok=True)
    data, report, idsets = {}, {}, {}
    for key, name in FILES.items():
        data[key], report[key], idsets[key] = load(name)
        print(key, report[key]['rows'], flush=True)
    comparisons = {}
    for a,b in [('frontal','full_dec'),('full_dec','full_mar13'),
                ('full_mar13','full_mar15'),('frontal','full_mar15'),
                ('frontal','merged_mrns')]:
        comparisons[a + '_vs_' + b] = compare(data[a], data[b])
    front = data['frontal']
    latest = data['full_mar15']
    eligible = {p for p,rs in latest.items() if len(rs)==1 and rs[0]['organ']==FRONT}
    common_front = set(front) & eligible
    gate = dict(frontal_file_paths=len(front), latest_predicted_frontal_paths=len(eligible),
                both_predicted_frontal_paths=len(common_front),
                frontal_file_absent_from_latest=len(set(front)-set(latest)),
                latest_frontal_absent_from_frontal_file=len(eligible-set(front)))
    common_field_counts=collections.Counter()
    for p in common_front:
        a,b=front[p][0],latest[p][0]
        if a['mrn'] and b['mrn'] and a['mrn'] != b['mrn']:
            common_field_counts['identity_conflict']+=1
        elif a['mrn'] and b['mrn']:
            common_field_counts['identity_agree']+=1
        else:
            common_field_counts['identity_incomplete']+=1
        if a['dx']:common_field_counts['old_diagnosis_present']+=1
        if a['icd']:common_field_counts['old_icd_present']+=1
        if a['crop']:common_field_counts['old_crop_present']+=1
    gate['both_frontal_fields']=dict(common_field_counts)
    internal=collections.Counter()
    for rs in latest.values():
        for r in rs:
            ids={x for x in r['mrn_fields'].values() if x}
            internal['multiple_mrn_fields_disagree' if len(ids)>1 else 'no_internal_mrn_disagreement']+=1
    supplemental={}
    for name,field in [('expert_chop_disease_mondo_labels.tsv','MRN'),
                        ('all_chops_icd10.tsv','pat_mrn_id'),
                        ('all_chops_snomed_ct.tsv','pat_mrn_id')]:
        n=0;ids=set();nonempty=collections.Counter()
        with (ROOT/name).open(newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                n+=1
                if mrnkey(r.get(field)):ids.add(mrnkey(r[field]))
                for k in ['mondo','Diagnosis','code','snomed_concept_id']:
                    if clean(r.get(k)):nonempty[k]+=1
        supplemental[name]=dict(rows=n,unique_mrns=len(ids),
                                shared_frontal_mrns=len(ids & idsets['frontal']),
                                nonempty_fields=dict(nonempty))
    existing={}
    with (OUT.parent/'labels.tsv').open(newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            existing[r['image_path']]=r
    current = dict(rows_by_unique_path=len(existing),
                   absent_from_frontal_metadata=len(set(existing)-set(front)),
                   frontal_metadata_not_in_current=len(set(front)-set(existing)))
    for field,oldkey in [('crop','cropping_path'),('dx','raw_label'),('mrn','mrn')]:
        current[field+'_disagree']=sum(front[p][0][field] != (mrnkey(r[oldkey]) if field=='mrn' else clean(r[oldkey])) for p,r in existing.items() if p in front)
    output=dict(sources=report, comparisons=comparisons, frontal_gate=gate,
                latest_internal_identity=dict(internal), supplemental_sources=supplemental,
                current_labels_vs_frontal=current,
                path_existence_samples={k:sample_files(front,k) for k in ['image_path','crop']},
                notes=['MRNs are compared after whitespace/NA cleanup, numeric .0 removal and leading-zero normalization; no identities are changed.',
                       'Frontal means exact EVA02_Organ label, not visual verification.',
                       'Date disagreements compare raw strings, not calendar dates. Formats differ across versions; do not interpret these counts as event-date conflicts.',
                       'Existence checks are deterministic 100-path samples, not exhaustive image decoding.',
                       'Supplemental sources are audited for schema and patient overlap only, not label correctness.'])
    (OUT/'source_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='sources'},indent=2),flush=True)


if __name__=='__main__':
    main()

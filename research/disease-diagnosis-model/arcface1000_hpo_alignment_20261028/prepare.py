"""Freeze genuine patient HPO pairs. No test labels enter model selection."""
import collections
import csv
import re
import os
from pathlib import Path
from common import *

def table(path):
    with Path(path).open(newline='') as f:
        first=f.readline(); f.seek(0)
        return list(csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ','))

def ontology(path):
    terms={}; aliases={}
    for block in Path(path).read_text().split('[Term]')[1:]:
        fields=collections.defaultdict(list)
        for line in block.splitlines():
            if ': ' in line:
                k,v=line.split(': ',1); fields[k].append(v)
        if not fields['id'] or fields.get('is_obsolete')==['true']: continue
        key=fields['id'][0]
        terms[key]=dict(name=fields['name'][0],parents=[v.split()[0] for v in fields['is_a']])
        aliases.update({v:key for v in fields['alt_id']})
    return terms,aliases

def descendants(terms, roots):
    seen=set(roots)
    while True:
        extra={k for k,v in terms.items() if set(v['parents'])&seen}-seen
        if not extra: return seen
        seen.update(extra)

def hp_ids(value, terms, aliases):
    raw=set(re.findall(r'HP[:_]\d{7}',value or ''))
    return {aliases.get(x.replace('_',':'),x.replace('_',':')) for x in raw if aliases.get(x.replace('_',':'),x.replace('_',':')) in terms}

def main():
    os.umask(0o077)
    if load(WORK/'data_receipt.json').get('state')=='complete': return
    terms,aliases=ontology(HPO)
    # Morphological craniofacial branches; this is semantic association, not localization supervision.
    roots=['HP:0000271','HP:0000377','HP:0000366','HP:0000492','HP:0000929','HP:0000315']
    if not set(roots)<=terms.keys(): raise RuntimeError('HPO_root_missing')
    facial=descendants(terms,roots)
    metadata_path=METADATA/'image_metadata_v1.1.0.tsv'
    meta=table(metadata_path); byimage={r['image_id']:r for r in meta}
    if len(byimage)!=len(meta): raise RuntimeError('duplicate_GMDB_image_metadata')
    reserved=set()
    for r in meta:
        text=' '.join(r.get(k,'') for k in ['username','patient_group','email','author'])
        if re.search(r'chop|philadelphia|bch|boston|childrens[.]harvard|tch[.]harvard|childrenshospital[.]org',text,re.I):
            reserved.add(r['patient_id'])
    blocked_hashes=set(); protected_files=[]
    for folder,prefix in [(PROJECT/'expert_holdout_audit',''),(PROJECT/'split_plan_v0','ian_')]:
        p=folder/(prefix+'blocked_crop_hashes.tsv');protected_files.append(p)
        blocked_hashes.update(r['sha256'] for r in table(p))
    source=load(GMDB_STAGE/'private/records.private.json')['rows']
    previous=load(GMDB_STAGE/'data_receipt.json')
    for name in ['private/records.private.json','private/images.uint8.npy']:
        if digest(GMDB_STAGE/name)!=previous['files'][name]: raise RuntimeError('pinned_GMDB_data_changed')
    protocol=load(SHARED/'protocol.json')
    splits={}
    for name in ['gmdb_train_images','gmdb_val_images','gmdb_frequent_test_images','gmdb_rare_test_images']:
        p=SHARED/'split_snapshot'/(name+'.csv')
        if digest(p)!=protocol['split_sha256'][name]: raise RuntimeError('official_split_changed')
        splits[name]=table(p)
    train_ids={r['subject'] for r in splits['gmdb_train_images']}
    val_ids={r['subject'] for r in splits['gmdb_val_images']}
    test_ids={r['subject'] for name,rows in splits.items() if 'test' in name for r in rows}
    if train_ids&val_ids: raise RuntimeError('GMDB_train_development_patient_overlap')
    # Preserve every official test patient; quarantine its training/development occurrences.
    split_overlap=dict(train_test=len(train_ids&test_ids),validation_test=len(val_ids&test_ids))
    reserved.update(test_ids)
    with (SHARED/'inputs.private.jsonl').open() as f:
        all_inputs=[__import__('json').loads(line) for line in f]
    test_hashes={r['crop_sha256'] for r in all_inputs if r['subject'] in test_ids}
    # Whole-patient quarantine, including every view of known institutional/duplicate cases.
    for r in source:
        if r['crop_sha256'] in blocked_hashes|test_hashes: reserved.add(r['subject'])
    counts=collections.Counter(); eligible=[]
    for r in source:
        m=byimage.get(r['image_id'])
        if not m or m['patient_id']!=r['subject']: raise RuntimeError('image_patient_pair_mismatch')
        if r['subject'] in reserved: counts['reserved_institution_or_duplicate_images']+=1; continue
        positive=hp_ids(m['present_features'],terms,aliases)
        negative=hp_ids(m['absent_features'],terms,aliases)
        if positive&negative: counts['contradictory_HPO_images']+=1; continue
        facial_positive=positive&facial
        if not facial_positive: counts['no_recorded_craniofacial_HPO_images']+=1; continue
        eligible.append(dict(array_index=r['array_index'],subject=r['subject'],split=r['split'],label=r['label'],
            image_id=r['image_id'],crop_sha256=r['crop_sha256'],hpo=sorted(facial_positive),
            all_present_hpo=sorted(positive),absent_hpo=sorted(negative)))
    groups=collections.defaultdict(list)
    for r in eligible: groups[r['subject']].append(r)
    patients=[]
    for subject,rows in sorted(groups.items()):
        if len({r['split'] for r in rows})!=1: raise RuntimeError('patient_has_two_splits')
        if len({r['label'] for r in rows})!=1: counts['conflicting_diagnosis_patients']+=1;continue
        positive=set().union(*(set(r['all_present_hpo']) for r in rows))
        absent=set().union(*(set(r['absent_hpo']) for r in rows))
        if positive&absent: counts['cross_record_HPO_conflict_patients']+=1;continue
        hp=sorted(set().union(*(set(r['hpo']) for r in rows)))
        patients.append(dict(subject=subject,split=rows[0]['split'],label=rows[0]['label'],
            image_indices=[r['array_index'] for r in rows],hpo=hp,
            text='Observed craniofacial findings: '+ '; '.join(terms[h]['name'] for h in hp)+'.',
            all_present_hpo=sorted(positive),absent_hpo=sorted(absent)))
    if not patients: raise RuntimeError('no_patient_HPO_pairs')
    summary={s:dict(patients=sum(p['split']==s for p in patients),
        images=sum(len(p['image_indices']) for p in patients if p['split']==s),
        diseases=len({p['label'] for p in patients if p['split']==s})) for s in ['train','validation']}
    if summary['train']['patients']<100 or summary['validation']['patients']<20: raise RuntimeError('insufficient_paired_development_cohort')
    save(WORK/'private/patients.json',dict(patients=patients))
    save(WORK/'private/ontology.json',dict(terms=terms,facial_roots=roots))
    save(WORK/'data_receipt.json',dict(state='complete',source='GMDB v1.1.0 patient present_features',
        counts=summary,exclusions=dict(counts),reserved_GMDB_patients=len(reserved),
        patient_split_disjoint=True,official_split_reused_with_test_patient_exclusion=True,
        original_split_patient_overlaps=split_overlap,CHOP_internal_training=False,BCH_training=False,
        GMDB_CHOP_BCH_tagged_patients_excluded=True,known_expert_crop_hashes_excluded=True,
        clinical_test_labels_used=False,private_MRN_mapping_used=False,PubMind_used=False,
        disease_profile_HPO_used_as_patient_observation=False,
        external_BCH_identity_crosswalk='not_available; independent cross-source audit pending',
        internal_HPO='NLP candidate sources withheld pending provenance and assertion adjudication',
        patient_file_sha256=digest(WORK/'private/patients.json'),metadata_sha256=digest(metadata_path),HPO_sha256=digest(HPO),
        protected_controls_sha256={str(p):digest(p) for p in protected_files},
        image_array_sha256=previous['files']['private/images.uint8.npy'],updated_at_utc=now()))
    print('Paired HPO release complete; private records remain local.')

if __name__=='__main__': main()

"""Local-only CHOP pair/provenance audit and public terminology inventory.

No original notes, identities or individual phenotype records are printed.
Candidate pairs remain quarantined until polarity and contemporaneity are established.
"""
import collections,csv,gzip,importlib.util,re,sys,os
from runtime import *
V1=PYTHON.parents[2]
RUN=Path('/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d')
CF=Path('/controlled-data/craniofacial_data')
MED=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
def rows(path):
    path=Path(path)
    with path.open('rb') as f:gz=f.read(2)==b'\x1f\x8b'
    with (gzip.open(path,'rt',newline='') if gz else path.open(newline='')) as f:
        first=f.readline();f.seek(0)
        yield from csv.DictReader(f,delimiter='\t' if first.count('\t')>first.count(',') else ',')
def norm(x):
    x=re.sub(r'\.0$','',str(x or '').strip());return x.lstrip('0') or '0' if x.isdigit() else x
def main():
    os.umask(0o077);csv.field_size_limit(10**8)
    sys.path.insert(0,str(RUN))
    from annotation_pipeline import canonical_path
    config=load(RUN/'config.json')
    def canonical(s):return str(canonical_path(s,config))
    source=PROJECT/'data_release_audit/chop_patient_consistent_candidates.tsv'
    paths=collections.defaultdict(set)
    for r in rows(source):
        for k in ['image_path','cropping_path','crop_path']:
            if r.get(k):paths[canonical(r[k])].add(norm(r['mrn']))
    reserved=set()
    for path in [PROJECT/'expert_holdout_audit/protected_mrns.tsv',PROJECT/'split_plan_v0/ian_protected_mrns.tsv']:
        if path.exists():
            for r in rows(path):reserved.update(norm(v) for k,v in r.items() if 'mrn' in k.lower() and v)
    for r in rows(PROJECT/'four_subset_inventory/patient_subsets.tsv'):
        if r['subset']=='1':reserved.add(norm(r['mrn']))
    release=load(V1/'dataset_release/release.json')
    training={r['photo_id']:r for r in rows(release['manifests']['train']['path'])}
    linked=collections.defaultdict(list);group_mrn=collections.defaultdict(set);counts=collections.Counter()
    for r in rows(RUN/'cropped_candidates.tsv'):
        if r['photo_id'] not in training:continue
        ids=paths.get(canonical(r['original_path']),set())
        if len(ids)!=1:counts['missing_or_ambiguous_old_identity_images']+=1;continue
        m=next(iter(ids));t=training[r['photo_id']]
        group_mrn[t['group_id']].add(m)
        linked[m].append(dict(group_id=t['group_id'],patient_key=t['patient_key'],crop_path=t['crop_path'],crop_sha256=t['crop_sha256']))
    ambiguous=set().union(*(v for v in group_mrn.values() if len(v)>1)) if group_mrn else set()
    targets=set(linked)-reserved-ambiguous
    evidence={norm(r['mrn']):r for r in rows(PROJECT/'data_release_audit/patient_manifest.tsv')}
    hpo_candidate={m for m in targets if any(int(evidence.get(m,{}).get(k) or 0) for k in ['ctakes_hpo_candidate_rows','phenominal_hpo_candidate_rows'])}
    save(WORK/'source_audit_progress.json',dict(state='linking_clinician_patient_table',eligible_training_patients=len(targets),NLP_HPO_candidate_patients=len(hpo_candidate),updated_at_utc=now()))
    bridge=collections.defaultdict(set);reverse=collections.defaultdict(set)
    for path in sorted((CF/'arcus_export/arcus_2026_06_29_full_ctakes').glob('patient-*.csv')):
        for r in rows(path):
            m=norm(r['pat_mrn_id']);pid=norm(r['pat_id'])
            # Track all target links; quarantine non-bijective identities.
            if m in targets:bridge[pid].add(m);reverse[m].add(pid)
    good={p:next(iter(v)) for p,v in bridge.items() if len(v)==1 and len(reverse[next(iter(v))])==1}
    ontology=load(PREVIOUS/'private/ontology.json')['terms'];observations=collections.defaultdict(list)
    flags=collections.Counter();keys=collections.Counter()
    hp_files=sorted((CF/'arcus_export/arcus_2026_06_29').glob('clinician_hpo_element*'))
    for path in hp_files:
        for r in rows(path):
            m=good.get(norm(r['pat_id']))
            if m is None:continue
            terms=set(re.findall(r'HP:\d{7}',r.get('hpo_code','')))&ontology.keys()
            if not terms:counts['unmapped_clinician_HPO_rows']+=1;continue
            code=r.get('code_json','')
            try:
                parsed=__import__('json').loads(code)
                if isinstance(parsed,dict):keys.update(parsed.keys())
            except Exception:pass
            flags['has_free_text_note']+=bool(r.get('notes','').strip())
            flags['has_creation_date']+=bool(r.get('creation_date','').strip())
            observations[m].append(dict(hpo=sorted(terms),creation_date=r.get('creation_date'),source=str(path),
                source_type='clinician_structured_HPO',assertion_state='unverified',code_json=r.get('code_json'),
                notes_present=bool(r.get('notes','').strip())))
    private=[]
    for m,obs in observations.items():
        private.append(dict(patient_groups=sorted({r['group_id'] for r in linked[m]}),images=linked[m],observations=obs,
            admitted=False,reason='assertion polarity and image-time relationship not established'))
    save(WORK/'private/chop_clinician_candidates.json',dict(patients=private))
    summary=dict(state='audit_complete',linked_current_train_patients=len(targets),linked_current_train_images=sum(len(linked[m]) for m in targets),
        reserved_patients_excluded=len(set(linked)&reserved),ambiguous_identity_patients_excluded=len(ambiguous),
        NLP_candidate_patients=len(hpo_candidate),clinician_HPO_paired_patients=len(observations),
        clinician_HPO_paired_images=sum(len(linked[m]) for m in observations),clinician_HPO_rows=sum(len(v) for v in observations.values()),
        clinician_HPO_terms=len({h for v in observations.values() for r in v for h in r['hpo']}),
        record_field_availability=dict(flags),code_json_schema_keys=sorted(keys),counts=dict(counts),
        CHOP_pairs_admitted_to_first_run=0,expert_test_admitted=False,private_MRN_mapping_read=False,
        next_requirement='resolve structured assertion semantics and temporal pairing; retain weak NLP labels as separate ablation',
        public_case_resources='Phenopacket Store and RareBench held out from training pending overlap/provenance review',updated_at_utc=now())
    save(WORK/'source_audit.json',summary)
    # Public disease-level associations are knowledge, never patient-specific labels.
    hpo=MED/'HPO/phenotype.hpoa';index=WORK/'public/phenotype_associations.jsonl';index.parent.mkdir(exist_ok=True)
    count=0;diseases=set()
    with hpo.open() as f,index.open('w') as out:
        lines=(s for s in f if not s.startswith('#'))
        reader=csv.DictReader(lines,delimiter='\t')
        for r in reader:
            disease=r.get('database_id',r.get('DatabaseID',''));term=r.get('hpo_id',r.get('HPO_ID',''))
            if not disease or not term:continue
            value=dict(disease_id=disease,disease_name=r.get('disease_name',''),hpo_id=term,
                qualifier=r.get('qualifier',''),reference=r.get('reference',''),evidence=r.get('evidence',''),
                source='HPO phenotype.hpoa',unit='disease_knowledge_not_patient')
            out.write(__import__('json').dumps(value)+'\n');count+=1;diseases.add(disease)
    save(WORK/'public_resource_receipt.json',dict(state='indexed',HPO_associations=count,HPO_diseases=len(diseases),
        source_sha256=digest(hpo),index_sha256=digest(index),patient_training_rows_added=0,
        use='version-pinned knowledge reference; no disease-to-patient phenotype substitution',updated_at_utc=now()))
    print('CHOP source audit and public knowledge index completed; records remain local.')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Read upstream CHOP resources and report aggregate linkage, never clinical text."""
import collections
import csv
import json
import random
from pathlib import Path
from audit_metadata_sources import FRONT, clean, mrnkey

HERE=Path(__file__).resolve().parent
ROOT=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images')
OUT=HERE/'metadata_audit'


def read(path):
    with path.open(newline='') as f:
        first=f.readline();sep='\t' if first.count('\t')>first.count(',') else ',';f.seek(0)
        yield from csv.DictReader(f,delimiter=sep)


def main():
    csv.field_size_limit(10**8)
    cf=set();gf=set();paths=set();cfrows={}
    for r in read(HERE/'labels.tsv'):
        if mrnkey(r['mrn']):cf.add(mrnkey(r['mrn']))
        paths.add(r['image_path']);cfrows[r['image_path']]=r
    for r in read(Path('/controlled-data/craniofacial_data/genetics_photos_metdata_top10dx.csv')):
        if r['EVA02_Organ']==FRONT:gf.add(mrnkey(r['pat_mrn_id']))
    cohorts={'craniofacial_frontal_old_mrn':cf,'genetics_frontal':gf,'union':cf|gf}
    def overlaps(ids):return {k:len(v&ids) for k,v in cohorts.items()}
    specs=[
        ('variant_info.csv','PAT_MRN_ID',['HGVS_NAME','DNA_CHANGE','GENE','ISCN_NAME','CLINICAL_SIGNIF_C','GENE_ID']),
        ('final_results/arcus_genetics_photos_faith.csv','MRN',['file_path','modified_file_path','DOB','APPOINTMENT_DATE']),
        ('final_results/ctakes_hpo.csv','pat_mrn_id',['HPOId','HPOName','method']),
        ('final_results/phenominal_hpo.csv','pat_mrn_id',['HPOId','HPOName','method']),
        ('photos_labels/craniofacial_labels.csv','PAT_MRN_ID',['ALL_NAMES']),
        ('photos_labels/genetics_labels.csv','PAT_MRN_ID',['ICD10']),
        ('notes20260220/arcus_notes.tsv','mrn',['note_id','final_score','rank_within_patient']),
    ]
    result={'excluded_sources': json.loads((HERE/'excluded_data_sources.json').read_text())}
    for name,mfield,fields in specs:
        path=Path(name) if name.startswith('/') else ROOT/name
        n=0;ids=set();nonempty=collections.Counter();fieldpats=collections.defaultdict(set)
        pc=collections.Counter();methods=collections.Counter();filepaths=set();modified=set()
        path_mrns=collections.defaultdict(set)
        for r in read(path):
            n+=1;m=mrnkey(r.get(mfield))
            if m:ids.add(m);pc[m]+=1
            for k in fields:
                if clean(r.get(k)):
                    nonempty[k]+=1
                    if m:fieldpats[k].add(m)
            if 'method' in r:methods[clean(r['method'])]+=1
            if name.endswith('arcus_genetics_photos_faith.csv'):
                if clean(r.get('file_path')):
                    filepaths.add(r['file_path']);path_mrns[r['file_path']].add(m)
                if clean(r.get('modified_file_path')):modified.add(r['modified_file_path'])
        rec=dict(path=str(path),rows=n,unique_mrns=len(ids),overlap=overlaps(ids),
                 nonempty_fields=dict(nonempty),
                 field_patient_coverage={k:dict(unique_mrns=len(v),overlap=overlaps(v)) for k,v in fieldpats.items()},
                 methods=dict(methods),patients_with_multiple_rows=sum(v>1 for v in pc.values()))
        if filepaths:
            rec['distinct_file_paths']=len(filepaths);rec['distinct_modified_paths']=len(modified)
            rec['paths_referencing_known_full_genetics_archive']=sum('/Genetics-Patient-Photos/' in p for p in modified)
            rec['path_sample']={}
            for label,ps in [('file_path',filepaths),('modified_file_path',modified)]:
                c=collections.Counter()
                for p in random.Random(20260928).sample(sorted(ps),min(50,len(ps))):
                    pp=Path(p);c['file' if pp.is_file() else 'directory' if pp.is_dir() else 'missing']+=1
                rec['path_sample'][label]=dict(c)
            rec['identity_join_check']=dict(distinct_source_paths=len(path_mrns),
                source_paths_with_multiple_mrns=sum(len(v)>1 for v in path_mrns.values()))
            rec['relative_path_samples_resolved_under_craniofacial_data']={}
            for label,ps in [('file_path',filepaths),('modified_file_path',modified)]:
                c=collections.Counter()
                for p in random.Random(20260928).sample(sorted(ps),min(50,len(ps))):
                    pp=Path('/controlled-data/craniofacial_data')/p
                    c['file' if pp.is_file() else 'directory' if pp.is_dir() else 'missing']+=1
                rec['relative_path_samples_resolved_under_craniofacial_data'][label]=dict(c)
            rec['path_note']='Both path columns are relative. Raw exists() checks alone are not availability evidence; resolved samples under /controlled-data/craniofacial_data were also checked.'
        result[name]=rec
        print(name,n,'patients',len(ids),'overlap',overlaps(ids),flush=True)
    # Older image table can retain demographic fields omitted from current labels.
    latest_full={r['image_path']:r['EVA02_Organ'] for r in read(Path('/controlled-data/craniofacial_data/complete_full_craniofacial_metadata_updated_03152026.tsv'))}
    oldfront=collections.Counter();oldfront_other=collections.Counter()
    oldstats=collections.Counter();shared=collections.Counter();organ=collections.Counter();extra_paths=set();old_ids=set()
    for r in read(ROOT/'final_results/craniofacial_metadata_cropping.csv'):
        oldstats['rows']+=1;p=clean(r['image_path']);organ[clean(r['Organ'])]+=1
        if r['Organ']=='frontal view of human face':
            oldfront['old_predicted_frontal_rows']+=1
            if p in paths:oldfront['already_in_current_frontal']+=1
            elif p in latest_full:oldfront_other[latest_full[p]]+=1
            else:oldfront['not_current_and_absent_from_latest_full']+=1
        if p in paths:
            oldstats['rows_shared_with_current_frontal']+=1
            for k in ['DOB','Age','Race','Ethnicity','Gender','date','MRN_ID','cropping_path']:
                if clean(r.get(k)):shared[k]+=1
        else:extra_paths.add(p)
        if mrnkey(r.get('MRN_ID')):old_ids.add(mrnkey(r['MRN_ID']))
    result['older_cropping_metadata']=dict(counts=dict(oldstats),shared_nonempty_fields=dict(shared),
                                         organ_counts=dict(organ),extra_paths=len(extra_paths),
                                         note='Additional paths are not automatically frontal; demographic values are extracted candidates, not verified attributes.')
    result['old_frontal_candidate_comparison']=dict(oldfront,not_current_but_latest_other_view=dict(oldfront_other))
    newer=list(read(ROOT/'final_results/genetics_photos_top10_metadata_04082026.tsv'))
    gp={r['image_path'] for r in newer if r['EVA02_Organ']==FRONT}
    result['april_genetics_metadata']=dict(rows=len(newer),frontal_unique_paths=len(gp),
        frontal_patients=len({mrnkey(r['pat_mrn_id']) for r in newer if r['EVA02_Organ']==FRONT}),
        frontal_rows_with_crop_path=sum(bool(clean(r['cropping_path'])) for r in newer if r['EVA02_Organ']==FRONT),
        note='Contains explicit cropping_path; intended frontal-only subset still needs view and crop QC.')
    result['scope_notes']=[
        'All overlaps use available MRNs, not adjudicated identities. No note text or patient identifiers are output.',
        'PubMind and all derived labels and selected subsets are excluded by user instruction; see DATA_SCOPE.md.',
        'Coded variant significance fields require source code dictionaries; missing significance is not pathogenicity.',
        'The 9.2 GB Clarity notes table is audited separately; its cohort selection lineage still requires verification.',
        'No additional images are admitted to the dataset by this audit.']
    OUT.mkdir(exist_ok=True)
    (OUT/'upstream_resource_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

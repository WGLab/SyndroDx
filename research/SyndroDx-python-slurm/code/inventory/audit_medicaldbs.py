#!/usr/bin/env python3
"""Read-only broad-source audit; save aggregates, never identifiers or clinical text.
Run with an environment containing pyarrow. No source notebooks/pickles are executed.
"""
import csv,json,re,os,collections
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from audit_metadata_sources import FRONT,mrnkey
ROOT=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
HERE=Path(__file__).resolve().parent
OUT=HERE/'metadata_audit/medicaldbs_audit.json'
csv.field_size_limit(10**8)
def rows(p):
    with p.open(newline='') as f:
        line=f.readline();f.seek(0)
        yield from csv.DictReader(f,delimiter='\t' if line.count('\t')>line.count(',') else ',')
def key(x):return mrnkey(str(x)) if x is not None else ''
def present(x):return x is not None and str(x).strip().lower() not in ('','nan','none','null','[]')
def main():
    cf={key(r['mrn']) for r in rows(HERE/'labels.tsv')}-{''}
    gf={key(r['pat_mrn_id']) for r in rows(Path('/controlled-data/craniofacial_data/genetics_photos_metdata_top10dx.csv')) if r['EVA02_Organ']==FRONT}-{''}
    cohorts={'craniofacial_frontal':cf,'genetics_frontal_existing_subset':gf,'union':cf|gf}
    target=cf|gf
    def overlap(ids):return {k:len(v&ids) for k,v in cohorts.items()}
    report={'scope':'All disease types; no top10 filter; provisional old-frontal MRN identities; PubMind and derived cohorts excluded.', 'cohort_patients':{k:len(v) for k,v in cohorts.items()},'tables':{}}
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    for rel,mc,dc in [
        ('CHOP_Diagnosis/expert_chop_disease_labels_combined_20260511.tsv','MRN','mondo'),
        ('CHOP_Diagnosis/expert_labels_manual_fix_20260511.tsv','MRN','mondo'),
        ('Arcus_old/genetic_testing_results.csv','MRN','Diagnosis Name'),
        ('Arcus_old/complete_processing_analysis.csv','MRN','mondo'),
        ('Clarity/all_mrns_chop_images_clarity_edg_dx.csv','MRN','DX_ID')]:
        ids=set();labels=set();linkedlabels=set();linked=set();n=0;fieldids=collections.defaultdict(set);dist=collections.defaultdict(set)
        for r in rows(ROOT/rel):
            n+=1;m=key(r.get(mc));ids.add(m)
            ds=re.findall(r'MONDO:\d+',(r.get(dc) or '')) if dc=='mondo' else [(r.get(dc) or '')]
            ds=[d for d in ds if present(d)];labels.update(ds)
            if m in target:
                linked.add(m);linkedlabels.update(ds)
                for d in ds:dist[d].add(m)
                for k,v in r.items():
                    if present(v):fieldids[k].add(m)
        ids.discard('')
        rec={'rows':n,'unique_mrns':len(ids),'overlap':overlap(ids),'distinct_raw_label_or_mondo_values':len(labels),'linked_distinct_values':len(linkedlabels),'linked_field_nonempty_patients':{k:len(v) for k,v in fieldids.items()},'linked_labels_by_minimum_patients':{str(k):sum(len(v)>=k for v in dist.values()) for k in [1,5,10,20,50,100]},'status':'candidate evidence, not adjudicated disease truth'}
        report['tables'][rel]=rec;print(rel,rec,flush=True);save()
    raw=ROOT/'Clarity_20260704/clarity_raw_exports'
    bridge=collections.defaultdict(set);mrn_patids=collections.defaultdict(set)
    for b in pq.ParquetFile(raw/'patient.parquet').iter_batches(columns=['pat_id','pat_mrn_id']):
        for p,m in zip(b.column(0).to_pylist(),b.column(1).to_pylist()):
            m=key(m)
            if m in target and p is not None:bridge[str(p)].add(m);mrn_patids[m].add(str(p))
    good={p:next(iter(ms)) for p,ms in bridge.items() if len(ms)==1 and len(mrn_patids[next(iter(ms))])==1}
    report['patient_bridge']={'matched':overlap(set(mrn_patids)), 'target_mrns_with_multiple_pat_ids':sum(len(v)>1 for v in mrn_patids.values()),'pat_ids_with_multiple_target_mrns':sum(len(v)>1 for v in bridge.values()),'unambiguous':overlap(set(good.values()))};print('BRIDGE',report['patient_bridge'],flush=True);save()
    for name,cols in [('variant.parquet',['pat_id','variant_id','gene_c','hgvs_name','clinical_signif_c','assessment_c']),('problem_list.parquet',['pat_id','dx_id','problem_status_c']),('pat_enc_dx.parquet',['pat_id','dx_id','enc_icd_code']),('expert_notes.parquet',['MRN','NOTE_ID']),('expert_full_notes_filtered.parquet',['MRN','NOTE_ID'])]:
        f=pq.ParquetFile(raw/name);ids=set();n=0;vals=collections.defaultdict(set);fieldids=collections.defaultdict(set);freq=collections.defaultdict(collections.Counter)
        ismrn=cols[0]=='MRN'; schema=f.schema_arrow
        matchset=None if ismrn else pa.array(list(good),type=schema.field('pat_id').type)
        for b in f.iter_batches(columns=cols,batch_size=131072):
            if not ismrn:b=b.filter(pc.is_in(b.column(0),value_set=matchset))
            for r in b.to_pylist():
                m=key(r['MRN']) if ismrn else good[str(r['pat_id'])]
                if m not in target:continue
                ids.add(m);n+=1
                for k in cols[1:]:
                    if present(r[k]):vals[k].add(str(r[k]));fieldids[k].add(m)
                    if k in ['clinical_signif_c','assessment_c','problem_status_c']:freq[k][str(r[k])]+=1
        rec={'total_rows':f.metadata.num_rows,'linked_rows':n,'overlap':overlap(ids),'linked_distinct_values':{k:len(v) for k,v in vals.items()},'nonempty_field_patients':{k:overlap(v) for k,v in fieldids.items()},'coded_value_counts_uninterpreted':{k:dict(v) for k,v in freq.items()},'status':'raw candidate' if 'expert' not in name else 'selection provenance pending; filtered derivative not admitted'}
        report['tables']['Clarity_20260704/clarity_raw_exports/'+name]=rec;print(name,rec,flush=True);save()
    for rel,patient,label in [('GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/image_metadata_v1.1.0.tsv','patient_id','internal_syndrome_id'),('CFAD/CFAD_Database.csv',None,'Anomaly')]:
        rr=list(rows(ROOT/rel));ids={r[patient] for r in rr} if patient else set();fields=['image_type','person_type'] if patient else ['Pose','Preproc','Etiology']
        report['tables'][rel]={'rows':len(rr),'unique_patient_ids':len(ids) if patient else None,'distinct_label_strings':len({r[label] for r in rr if present(r[label])}),'categories':{k:dict(collections.Counter(r[k] for r in rr)) for k in fields}}
        print(rel,report['tables'][rel],flush=True);save()
    inv={}
    # Count files, not patient-level filenames. Walk image resources only, without following symlinks.
    for name in ['CFAD','GMDB','GMDB_v1.1.0','FairFace','Caltech_WebFaces','Clinical-Genetics-Training-Data','RareBench','phenopacket_store','public_images']:
        c=collections.Counter();size=0
        for base,dirs,files in os.walk(ROOT/name):
            dirs[:]=[d for d in dirs if not d.startswith('.')]
            for file in files:
                p=Path(base)/file;c[p.suffix.lower()]+=1;size+=p.stat().st_size
        inv[name]={'file_extensions':dict(c),'total_bytes':size,'note':'Physical file counts include copies/augmentations; not unique usable patients or frontal images.'}
    report['file_inventory']=inv;save();print('DONE',flush=True)
if __name__=='__main__':main()

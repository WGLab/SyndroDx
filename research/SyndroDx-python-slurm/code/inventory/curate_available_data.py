#!/usr/bin/env python3
"""Inventory frontal candidates, verify files, quarantine identity conflicts.
No raw data, existing labels or model splits are altered. Requires Pillow.
Outputs remain local; row-level manifests include identifiers for audit/review.
"""
import csv,json,hashlib,io,os,stat,collections,warnings,concurrent.futures,time
from pathlib import Path
from PIL import Image,ImageFile
from audit_metadata_sources import FRONT,clean,mrnkey
from audit_validation_evidence import xlsx
HERE=Path(__file__).resolve().parent
OUT=HERE/'data_release_audit'
CF=Path('/controlled-data/craniofacial_data')
MED=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')
CACHE=OUT/'file_checks.tsv'
csv.field_size_limit(10**8)
ImageFile.LOAD_TRUNCATED_IMAGES=False
warnings.simplefilter('error',Image.DecompressionBombWarning)
INPUTS={}
def read(p):
    p=Path(p);st=p.stat();INPUTS[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    with p.open(newline='') as f:
        line=f.readline();f.seek(0)
        yield from csv.DictReader(f,delimiter='\t' if line.count('\t')>line.count(',') else ',')
def write_tsv(path,rr,fields=None):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
def main():
    OUT.mkdir(exist_ok=True)
    latest={r['image_path']:r for r in read(CF/'complete_full_craniofacial_metadata_updated_03152026.tsv')}
    dec={r['image_path']:r for r in read(CF/'complete_full_craniofacial_metadata_updated_123025_deduplicated.csv') if r['EVA02_Organ']==FRONT}
    previous={r['image_path']:r for r in read(HERE/'labels.tsv')}
    rows=[]
    def add_cf(r,src):
        path=r['image_path'];new=latest.get(path,{});old=mrnkey(r.get('norm_mrn'));norm=mrnkey(new.get('norm_mrn'))
        # Preserve every nonempty identity field. Do not silently select one version.
        fields={old,norm}|{mrnkey(new.get(k)) for k in ['MRN','chart_mrn']};fields.discard('')
        conflict=len(fields)>1
        m=next(iter(fields)) if len(fields)==1 else ''
        ident='conflict' if conflict else 'missing' if not m else 'single_available_version' if not new else 'stable_or_nonconflicting_fields'
        prev=previous.get(path,{})
        rows.append(dict(source=src,image_path=path,cropping_path=r.get('cropping_path',''),mrn=m,original_mrn=old,latest_norm_mrn=norm,identity_candidates='|'.join(sorted(fields)),identity_status=ident,view_status='predicted_frontal' if not new or new.get('EVA02_Organ')==FRONT else 'view_disagreement',raw_label=r.get('disease_or_phenotypes',''),raw_label_status='folder_extraction_unverified',original_split=prev.get('split',''),crossfolder_tier=prev.get('tier',''),patient_groups=prev.get('patient_groups',''),expert_record=False,official_group='',normalization_status='pending',visual_qc='pending'))
    seen=set()
    for r in read(CF/'craniofacial_metadata_frontal_updated_123025_deduplicated.csv'):
        add_cf(r,'CHOP_craniofacial');seen.add(r['image_path'])
    for path,r in dec.items():
        if path not in seen:add_cf(r,'CHOP_craniofacial_full_only');seen.add(path)
    print('CF candidates',len(rows),flush=True)
    # Check all historical genetics assignments, including rows removed by deduplication.
    gm=collections.defaultdict(set)
    for r in read(CF/'genetics_photos_metdata_top10dx.csv'):
        m=mrnkey(r['pat_mrn_id'])
        if m:gm[r['image_path']].add(m)
    gp=MED/'CHOP_Images/final_results/genetics_photos_top10_metadata_04082026.tsv'
    for r in read(gp):
        if r['EVA02_Organ']!=FRONT:continue
        fields=gm[r['image_path']]|{mrnkey(r['pat_mrn_id'])};fields.discard('');m=next(iter(fields)) if len(fields)==1 else ''
        rows.append(dict(source='CHOP_genetics',image_path=r['image_path'],cropping_path=r.get('cropping_path',''),mrn=m,original_mrn=mrnkey(r['pat_mrn_id']),latest_norm_mrn='',identity_candidates='|'.join(sorted(fields)),identity_status='conflict' if len(fields)>1 else 'missing' if not m else 'stable_or_nonconflicting_fields',view_status='predicted_frontal',raw_label=r.get('unified_syndrome',''),raw_label_status='upstream_disease_lineage_pending',original_split='',crossfolder_tier='',patient_groups='',expert_record=False,official_group='',normalization_status='pending',visual_qc='pending'))
    # An expert-record link is a review candidate, not automatic clinical confirmation.
    ep=MED/'CHOP_Images/examples/Ian_verified.xlsx';expert=xlsx(ep);es=ep.stat();INPUTS[str(ep)]={'bytes':es.st_size,'mtime_ns':es.st_mtime_ns}
    eids={mrnkey(r['pat_mrn_id']) for r in expert}
    for r in rows:r['expert_record']=r['mrn'] in eids if r['mrn'] else False
    # External sources remain separate; never add their IDs to CHOP patient counts.
    gbase=MED/'GMDB_v1.1.0'
    maps={}
    for typ in ['gmdb_images_v1.1.0/gmdb_images','gmdb_crops_v1.1.0/gmdb_crops']:
        maps[typ]={p.name:str(p) for p in (gbase/typ).iterdir() if p.is_file()}
    for r in read(gbase/'gmdb_metadata_v1.1.0/gmdb_metadata/image_metadata_v1.1.0.tsv'):
        if r['image_type']!='Frontal face':continue
        paths=[]
        for typ,mp in maps.items():paths.append(mp.get(r['filename']) or mp.get(r['image_id']+'.jpg') or mp.get(r['image_id']+'_aligned.jpg') or '')
        rows.append(dict(source='GMDB_v1.1.0',image_path=paths[0],cropping_path=paths[1],mrn='GMDB:'+r['patient_id'],original_mrn='',latest_norm_mrn='',identity_candidates='',identity_status='external_id_not_reconciled',view_status='metadata_frontal',raw_label=r['internal_syndrome_name'],raw_label_status='source_metadata_pending_review',original_split='',crossfolder_tier='',patient_groups='',expert_record=False,official_group=r.get('patient_group',''),normalization_status='pending',visual_qc='pending'))
    for r in read(MED/'CFAD/CFAD_Database.csv'):
        if r['Pose']!='Front':continue
        p=str(MED/'CFAD/CFAD'/r['Folder']/r['Filename'])
        rows.append(dict(source='CFAD',image_path=p,cropping_path=p,mrn='CFAD_group:'+r['Folder'],original_mrn='',latest_norm_mrn='',identity_candidates='',identity_status='image_group_not_verified_patient',view_status='metadata_frontal',raw_label=r.get('Anomaly',''),raw_label_status='appearance_label_not_genetic_diagnosis',original_split='',crossfolder_tier='',patient_groups='',expert_record=False,official_group=r.get('Preproc',''),normalization_status='not_disease_ground_truth',visual_qc='pending'))
    del latest,dec,previous
    write_tsv(OUT/'candidate_inventory.tsv',rows)
    (OUT/'input_versions.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
    checks={}
    if CACHE.exists():
        with CACHE.open() as f:
            for r in csv.DictReader(f,delimiter='\t'):checks[r['path']]=r
    tasks={}
    for r in rows:
        if r['image_path']:tasks.setdefault(r['image_path'],False)
        if r['cropping_path']:tasks[r['cropping_path']]=True
    fields=['path','size','mtime_ns','mode','status','width','height','format','sha256','error_type']
    def check(task):
        p,decode=task
        rec=dict(path=p,size='',mtime_ns='',mode='decode_reduced_512' if decode else 'stat',status='',width='',height='',format='',sha256='',error_type='')
        try:
            st=Path(p).stat();rec.update(size=str(st.st_size),mtime_ns=str(st.st_mtime_ns))
            old=checks.get(p)
            if old and all(old.get(k)==rec[k] for k in ['size','mtime_ns','mode']):return old
            if not stat.S_ISREG(st.st_mode):rec['status']='not_file';return rec
            if not decode:rec['status']='exists';return rec
            data=Path(p).read_bytes();rec['sha256']=hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                rec.update(width=str(im.width),height=str(im.height),format=im.format or '')
                im.draft("RGB",(512,512))
                im.load()
            rec['status']='decoded'
        except FileNotFoundError:rec['status']='missing'
        except PermissionError:rec.update(status='unreadable',error_type='PermissionError')
        except Exception as e:rec.update(status='decode_or_io_error',error_type=type(e).__name__)
        return rec
    append=CACHE.exists() and CACHE.stat().st_size>0
    start=time.time();todo=list(tasks.items());print('Checking paths',len(todo),'decode',sum(tasks.values()),'cached',len(checks),flush=True)
    with CACHE.open('a',newline='') as f,concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t')
        if not append:w.writeheader()
        for offset in range(0,len(todo),1024):
            for rec in pool.map(check,todo[offset:offset+1024]):
                checks[rec['path']]=rec;w.writerow(rec)
            f.flush()
            if offset%2048==0:print('Checked',min(offset+1024,len(todo)),'/',len(todo),'seconds',round(time.time()-start),flush=True)
    # Exact crop duplicates: stronger cross-patient identity conflict than filename duplication.
    hashes=collections.defaultdict(list)
    for i,r in enumerate(rows):
        oc=checks.get(r['image_path'],{});cc=checks.get(r['cropping_path'],{})
        r.update(original_file_status=oc.get('status','no_path'),crop_file_status=cc.get('status','no_path'),crop_width=cc.get('width',''),crop_height=cc.get('height',''),crop_sha256=cc.get('sha256',''),exact_duplicate_group='',duplicate_identity_conflict=False,technical_candidate=False,eligible_identity_candidate=False,review_reasons='')
        if cc.get('status')=='decoded':hashes[cc['sha256']].append(i)
    cross_source_groups=0
    for h,idx in hashes.items():
        if len(idx)<2:continue
        ms={rows[i]['mrn'] for i in idx if rows[i]['mrn']};conflict=len(ms)>1
        if len({rows[i]['source'] for i in idx})>1:cross_source_groups+=1
        for i in idx:rows[i]['exact_duplicate_group']=h;rows[i]['duplicate_identity_conflict']=conflict
    for r in rows:
        reasons=[]
        r['technical_candidate']=r['crop_file_status']=='decoded' and r['original_file_status'] in ['exists','decoded']
        if not r['technical_candidate']:reasons.append('file_or_decode_issue')
        if r['identity_status'] in ['conflict','missing']:reasons.append('identity_'+r['identity_status'])
        if r['duplicate_identity_conflict']:reasons.append('exact_crop_shared_by_different_ids')
        if r['view_status']=='view_disagreement':reasons.append('view_disagreement')
        r['eligible_identity_candidate']=r['technical_candidate'] and not reasons and r['source'].startswith('CHOP')
        r['review_reasons']='|'.join(reasons or ['visual_qc_and_diagnosis_review_pending'])
    write_tsv(OUT/'image_manifest.tsv',rows)
    summary={'scope':'All diseases, frontal candidates only. PubMind excluded. No automated clinical or visual confirmation.','by_source':{},'cross_source_exact_crop_duplicate_groups':cross_source_groups,'excluded_by_design':['PubMind and derivatives','side/top/body/document/radiology views','videos and 3D files','unresolved full genetics image archive','synthetic/augmented GMDB copies as new patients'],'limitations':['Full file read, SHA256 and Pillow decoding (JPEG draft to 512 target, original dimensions retained) validate crop readability, not frontal view or diagnosis. Full-resolution pixel decoding was not performed for every JPEG. Originals checked for existence only unless also used as a crop.','Exact byte hashes do not detect recompression, near-duplicates or family overlap.','Current identity rules quarantine disagreements without adjudicating them.','Expert-file linkage does not establish review scope or test independence.','CFAD identifiers are image groups, not verified unique people.','External subject categories, official splits and licensing still need review.']}
    for src in sorted({r['source'] for r in rows}):
        rr=[r for r in rows if r['source']==src];valid=[r for r in rr if r['eligible_identity_candidate']];tech=[r for r in rr if r['technical_candidate']]
        summary['by_source'][src]={'candidate_rows':len(rr),'technical_rows':len(tech),'technical_exact_unique_crops':len({r['crop_sha256'] for r in tech}),'technical_distinct_ids':len({r['mrn'] for r in tech if r['mrn']}),'identity_status_rows':dict(collections.Counter(r['identity_status'] for r in rr)),'crop_status_rows':dict(collections.Counter(r['crop_file_status'] for r in rr)),'original_status_rows':dict(collections.Counter(r['original_file_status'] for r in rr)),'duplicate_identity_conflict_rows':sum(r['duplicate_identity_conflict'] for r in rr),'identity_screened_candidate_rows':len(valid),'identity_screened_unique_crops':len({r['crop_sha256'] for r in valid}),'identity_screened_patients':len({r['mrn'] for r in valid}),'expert_linked_screened_patients':len({r['mrn'] for r in valid if r['expert_record']})}
    chop=[r for r in rows if r['source'].startswith('CHOP') and r['eligible_identity_candidate']]
    summary['CHOP_union']={'identity_screened_rows':len(chop),'exact_unique_crops':len({r['crop_sha256'] for r in chop}),'patients':len({r['mrn'] for r in chop}),'expert_record_patients':len({r['mrn'] for r in chop if r['expert_record']})}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

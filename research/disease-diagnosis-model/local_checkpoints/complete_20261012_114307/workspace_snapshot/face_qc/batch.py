"""Resumable, CPU screening only; never releases images to training."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from face_qc.pilot import ROOT, MODEL, read, write, process, mp, cv2
from expert_reservation import filter_reserved

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(out, shards):
    if out.exists(): raise SystemExit('Run directory already exists; preserve it.')
    selected={(r['mrn'],r['crop_sha256']) for r in read(ROOT/'four_subset_inventory/image_subsets.tsv') if r['subset']=='4'}
    rows=[r for r in read(ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv') if (r['mrn'],r['crop_sha256']) in selected]
    rows,blocked=filter_reserved(rows)
    assert len({(r['mrn'],r['crop_sha256']) for r in rows})==len(rows)
    assert not any('pubmind' in r['source'].lower() for r in rows)
    out.mkdir(parents=True)
    records=[dict(sample_id=f's4_{i:06d}',sampling='subset4_all',mrn=r['mrn'],source=r['source'],image_path=r['image_path'],upstream_crop_sha256=r['crop_sha256']) for i,r in enumerate(sorted(rows,key=lambda r:(r['mrn'],r['crop_sha256'])))]
    for i in range(shards): write(out/f'input_{i:03d}.tsv',records[i::shards])
    inputs=['four_subset_inventory/image_subsets.tsv','data_release_audit/chop_patient_consistent_candidates.tsv','face_qc/pilot.py','face_qc/batch.py','face_qc/requirements.lock.txt']
    manifest=dict(images=len(records),patients=len({r['mrn'] for r in records}),shards=shards,blocked=blocked,model_sha256=digest(MODEL),input_sha256={p:digest(ROOT/p) for p in inputs},status='screening_only_no_training_admission')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
def run(out,index):
    manifest=json.loads((out/'manifest.json').read_text())
    for p in ['face_qc/pilot.py','face_qc/batch.py']:
        assert digest(ROOT/p)==manifest['input_sha256'][p], 'Code changed since preparation'
    assert digest(MODEL)==manifest['model_sha256']
    dest=out/f'shard_{index:03d}';dest.mkdir(exist_ok=True)
    for d in ['crops','oriented']:(dest/d).mkdir(exist_ok=True)
    progress=dest/'progress.jsonl';done={}
    if progress.exists():
        for line in progress.read_text().splitlines():
            r=json.loads(line);done[r['sample_id']]=r
    rows=read(out/f'input_{index:03d}.tsv');cv2.setNumThreads(1)
    opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,output_facial_transformation_matrixes=True,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
        for i,rec in enumerate(rows):
            if rec['sample_id'] in done:continue
            r=process(rec,det,dest)
            # Retain bounded previews per shard; numerical QC and original path for all.
            if i>=12:
                for key in ['crop_preview_path','oriented_preview_path']:
                    p=r.pop(key,None)
                    if p:Path(p).unlink()
            with progress.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
            done[r['sample_id']]=r
            if (i+1)%100==0:print(f'{index}: {i+1}/{len(rows)}',flush=True)
    results=[done[r['sample_id']] for r in rows];write(dest/'results.tsv',results)
    summary=dict(images=len(results),decisions=dict(Counter(r['decision'] for r in results)),errors=sum(r['reasons'].startswith('processing_error') for r in results),training_admitted=False)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
def merge(out):
    m=json.loads((out/'manifest.json').read_text());rows=[]
    for i in range(m['shards']):
        rr=read(out/f'shard_{i:03d}/results.tsv'); expected=read(out/f'input_{i:03d}.tsv')
        assert {r['sample_id'] for r in rr}=={r['sample_id'] for r in expected}
        rows.extend(rr)
    assert len(rows)==m['images']==len({r['sample_id'] for r in rows})
    assert all(r['training_admitted']=='False' for r in rows)
    write(out/'results.tsv',rows)
    report=dict(images=len(rows),patients=len({r['mrn'] for r in rows}),decisions=dict(Counter(r['decision'] for r in rows)),reasons=dict(Counter(x for r in rows for x in r['reasons'].split('|') if x)),training_admitted=False,status='uncalibrated_screening_requires_review')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','merge']);ap.add_argument('--out',type=Path,default=ROOT/'face_qc/subset4_screen_v1');ap.add_argument('--shards',type=int,default=32);ap.add_argument('--index',type=int,default=0);a=ap.parse_args()
    if a.action=='prepare':prepare(a.out,a.shards)
    elif a.action=='run':run(a.out,a.index)
    else:merge(a.out)

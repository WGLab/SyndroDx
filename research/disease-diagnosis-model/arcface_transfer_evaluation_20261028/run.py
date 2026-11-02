"""Locked checkpoint selection, local leakage checks, frozen retrieval/few-shot."""
import sys,os,time,csv,collections,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MATCH=ROOT.parent/'transface_objective_comparison_20261028'
sys.path.insert(0,str(MATCH))
from matched_runtime import load,save,digest,now,PROJECT,ID,V1,PYTHON,SLURM,environment,free_jobs,notify
SHARED=PROJECT/'gmdb_checkpoint_eval_20261026'
import numpy as np

def prepare():
    if (ROOT/'selection.json').exists():return
    candidates=[]
    for folder in sorted((MATCH/'arcface_lr5e7/evaluations').glob('step_*')):
        r=load(folder/'results.json')
        if r and (folder/'encoder.pt').exists() and not r.get('severe_collapse'):
            candidates.append((r['gmdb_development']['retrieval']['top1'],-r['step'],folder,r))
    if not candidates:raise RuntimeError('no_evaluated_checkpoint')
    _,_,folder,report=max(candidates,key=lambda v:v[:2])
    import shutil
    shutil.copy2(folder/'encoder.pt',ROOT/'selected_encoder.pt')
    save(ROOT/'selection.json',dict(step=report['step'],path=str(ROOT/'selected_encoder.pt'),sha256=digest(ROOT/'selected_encoder.pt'),
        selected_by='highest available GMDB development macro retrieval Top1; earlier step breaks ties',
        development_score=report['gmdb_development']['retrieval'],candidate_steps=[v[3]['step'] for v in candidates],
        selection_locked_before_test=True,updated_at_utc=now()))
    save(ROOT/'input_pins.json',dict(inputs=digest(SHARED/'inputs.private.jsonl'),splits={p.stem:digest(p) for p in (SHARED/'split_snapshot').glob('*.csv')}))

def records():
    import json
    assert digest(SHARED/'inputs.private.jsonl')==load(ROOT/'input_pins.json')['inputs']
    return [json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()]

def worker(kind):
    import torch,cv2
    from model import backbone
    from torch.utils.data import Dataset,DataLoader
    rows=records();torch.set_num_threads(4);cv2.setNumThreads(1)
    model=backbone().cuda().eval().requires_grad_(False)
    if kind=='selected':
        s=load(ROOT/'selection.json');assert digest(s['path'])==s['sha256']
        model.load_state_dict(torch.load(s['path'],map_location='cpu',weights_only=False)['backbone'],strict=True)
    class Images(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('image_integrity_changed')
            im=cv2.imread(r['crop_path']);
            if im is None:raise RuntimeError('local_decode_failure')
            x=cv2.cvtColor(cv2.resize(im,(112,112)),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()
            return torch.from_numpy(x).float().div_(127.5).sub_(1)
    values=[]
    with torch.inference_mode():
        for x in DataLoader(Images(),batch_size=256,num_workers=2,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):f=model(x.cuda(non_blocking=True))
            f=torch.nn.functional.normalize(f.float(),dim=-1)
            if not torch.isfinite(f).all():raise RuntimeError('nonfinite_features')
            values.append(f.cpu().numpy())
    np.save(ROOT/'private'/f'{kind}.npy',np.concatenate(values))
    save(ROOT/f'{kind}_ready.json',dict(state='complete',updated_at_utc=now()))

def evaluate():
    from data import verify_manifest
    from score import fewshot
    sys.path.insert(0,str(V1))
    from evaluation_core import normalized,retrieval,macro_score,paired_interval
    rows=records();byimage={r['image_id']:i for i,r in enumerate(rows)}
    assert len(byimage)==len(rows)
    pins=load(ROOT/'input_pins.json');tables={}
    for name in ['gmdb_train_images','gmdb_frequent_gallery_images','gmdb_frequent_test_images','gmdb_rare_gallery_images','gmdb_rare_test_images']:
        p=SHARED/'split_snapshot'/(name+'.csv');assert digest(p)==pins['splits'][name]
        with p.open() as f:tables[name]=list(csv.DictReader(f))
    train=verify_manifest(load(ID/'protocol.json')['train_manifest'])
    clinical_hashes={r.get(k) for r in train for k in ['crop_sha256','pixel_sha256','source_sha256'] if r.get(k)}
    blocked={r['subject'] for r in rows if r['crop_sha256'] in clinical_hashes}
    # Quarantine whole patients for duplicate file bytes across different public patient IDs.
    hashes=collections.defaultdict(set)
    for r in rows:hashes[r['crop_sha256']].add(r['subject'])
    ambiguous=set().union(*(v for v in hashes.values() if len(v)>1)) if hashes else set()
    blocked|=ambiguous
    def clean(rs):return [r for r in rs if r['subject'] not in blocked]
    tests=clean(tables['gmdb_frequent_test_images']+tables['gmdb_rare_test_images'])
    testsubjects={r['subject'] for r in tests}
    support=clean(tables['gmdb_train_images']);overlap={r['subject'] for r in support}&testsubjects
    support=[r for r in support if r['subject'] not in testsubjects]
    assert not {r['subject'] for r in support}&testsubjects
    def aggregate(rs,features):
        groups=collections.defaultdict(list);labels={}
        for r in rs:
            if r['subject'] in labels and labels[r['subject']]!=r['label']:raise RuntimeError('conflicting_patient_labels')
            labels[r['subject']]=r['label'];groups[r['subject']].append(byimage[r['image_id']])
        ids=sorted(groups)
        return ids,normalized(np.stack([features[sorted(set(groups[i]))].mean(0) for i in ids])),[labels[i] for i in ids]
    result={};hitstore={}
    for kind in ['original','selected']:
        features=np.load(ROOT/'private'/f'{kind}.npy',allow_pickle=False);assert features.shape==(len(rows),512)
        out={};hitstore[kind]={}
        for case in ['frequent','rare']:
            folds=[]
            for fold in ([None] if case=='frequent' else range(10)):
                g=clean(tables[f'gmdb_{case}_gallery_images']);q=clean(tables[f'gmdb_{case}_test_images'])
                if fold is not None:g=[r for r in g if int(r['split'])==fold];q=[r for r in q if int(r['split'])==fold]
                # A query patient can never supply its own gallery image.
                gids,gf,gy=aggregate(g,features);qids,qf,qy=aggregate(q,features)
                h,c=retrieval(qf,gf,qids,gids,[{v} for v in qy],[{v} for v in gy])
                folds.append(dict(query_patients=len(qids),gallery_patients=len(gids),covered_patients=int(c.sum()),**{f'top{k}':macro_score(h[k],[{v} for v in qy]) for k in h}))
            out[case]=dict(folds=len(folds),**{f'top{k}':float(np.mean([v[f'top{k}'] for v in folds])) for k in [1,5,10]},cohort=folds)
        _,x,y=aggregate(support,features);_,q,qy=aggregate(tests,features)
        out['fewshot']=fewshot(x,q,y,qy);result[kind]=out
    audit=dict(patient_level_aggregation=True,self_patient_retrieval_excluded=True,
        support_test_patient_overlap_after_filter=0,training_test_overlap_patients_quarantined=len(overlap),
        public_duplicate_patient_ids_quarantined=len(ambiguous),all_quarantined_patients=len(blocked),
        cross_source_patient_identity_crosswalk_complete=False,near_duplicate_cross_source_audit_complete=False,
        limitation='Exact-hash and known patient controls verified; unresolved cross-source identities/near-duplicates prevent a comprehensive no-leakage claim.')
    save(ROOT/'audit.json',audit)
    report=dict(state='complete',selection=load(ROOT/'selection.json'),models=result,audit=audit,
        protocol='patient-level cosine retrieval and fixed-support prototype few-shot, Top1/5/10; 20 identical support draws per k',
        official_image_level_numbers_directly_comparable=False,exploratory_repeated_test=True,updated_at_utc=now())
    save(ROOT/'results.json',report);notify('ArcFace retrieval and few-shot complete',__import__('json').dumps(report,indent=2))
    save(ROOT/'notification.json',dict(state='notifier_invoked',updated_at_utc=now()))

def main():
    os.umask(0o077);(ROOT/'private').mkdir(exist_ok=True);prepare()
    for kind in ['original','selected']:
        if load(ROOT/f'{kind}_ready.json').get('state')=='complete':continue
        while True:
            pool=free_jobs()
            if pool:break
            save(ROOT/'status.json',dict(state='waiting_for_free_gpu',updated_at_utc=now()));time.sleep(20)
        j=pool[0];save(ROOT/'status.json',dict(state='extracting',model=kind,job=j['id'],updated_at_utc=now()))
        cmd=[str(SLURM/'srun'),'--jobid='+j['id'],'--job-name=arcface-transfer-'+kind,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(ROOT/'run.py'),'worker',kind]
        with (ROOT/'logs'/f'{kind}.private.log').open('ab') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=f,check=True)
    evaluate();save(ROOT/'status.json',dict(state='complete',updated_at_utc=now()))

if __name__=='__main__':
    try:
        if len(sys.argv)>1:worker(sys.argv[2])
        else:main()
    except Exception as e:
        save(ROOT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        notify('ArcFace transfer evaluation failed','Evaluation requires inspection: '+type(e).__name__+'. No patient information is included.');raise

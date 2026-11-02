"""Fixed CHOP expert cohort; GMDB-selected encoders only, no expert tuning."""
import os,sys,time,subprocess,collections
from pathlib import Path
import numpy as np
from common import *
OUT=WORK/'expert';SOURCE=PROJECT/'chop_expert_comparison_20261027'

def prepare():
    import csv,cv2,hashlib,importlib.util
    spec=importlib.util.spec_from_file_location('matched_expert_audit_source',WORK/'prepare.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);phash=module.phash
    release=load(SOURCE/'release.json');path=SOURCE/'private/released.json';assert digest(path)==release['release_sha256']
    rows=load(path)['rows'];pins=load(WORK/'data_pins.json')
    assert digest(WORK/'private/records.private.json')==pins['records_sha256']
    inventory=load(WORK/'private/records.private.json')['rows']
    training=[r for r in inventory if r['split'] in ['train','validation']]
    # The new benchmark's training images are a subset of the previously audited set.
    old=load(PROJECT/'gestaltmatcher_arc_comparison_20261027/private/records.private.json')['rows'];old_hashes={r['crop_sha256'] for r in old if r['split']=='train'}
    new_hashes={r['crop_sha256'] for r in training if r['split']=='train'};assert new_hashes<=old_hashes
    cp=load(PROJECT/'transface_arcface_identity_cpt_20261026/protocol.json')['train_manifest'];assert digest(cp['path'])==cp['sha256']
    with Path(cp['path']).open() as f:clinical=list(csv.DictReader(f,delimiter='\t'))
    groups={r['group_id'] for r in clinical};keys={r['patient_key'] for r in clinical};hashes={r.get(k) for r in clinical+training for k in ['crop_sha256','pixel_sha256','source_sha256'] if r.get(k)}
    blocked={r['patient'] for r in rows if r.get('group_id') in groups or r.get('patient_key') in keys or any(r.get(k) in hashes for k in ['crop_sha256','pixel_sha256','source_sha256'] if r.get(k))}
    # New RGB112 perceptual audit reuses private hashes from the matched benchmark.
    public=[__import__('json').loads(l) for l in (SHARED/'inputs.private.jsonl').open()];allrows={r['image_id']:r for r in old+public}
    order=sorted(allrows);train_ids={r['image_id'] for r in training if r['split']=='train'}
    public_hash=np.load(WORK/'private/public_phashes.npy',allow_pickle=False)
    # Audit index order also includes the original (before-quarantine) source rows.
    original=load(PROJECT/'gmdb_arcface_finetune_20261029/private/records.private.json')['rows'];keys_order=sorted({r['image_id'] for r in original+public})
    assert len(keys_order)==len(public_hash)
    gallery_hashes=public_hash[[i for i,k in enumerate(keys_order) if k in train_ids]]
    clinical_hash=np.load(WORK/'private/clinical_phashes.npy',allow_pickle=False)
    for row in rows:
        if digest(row['crop_path'])!=row['crop_sha256']:raise RuntimeError('expert_crop_integrity_changed')
        bgr=cv2.imread(row['crop_path'])
        if bgr is None:raise RuntimeError('expert_crop_decode_failed')
        rgb=cv2.cvtColor(cv2.resize(bgr,(112,112)),cv2.COLOR_BGR2RGB);h=phash(rgb)
        if any(np.any(np.minimum(np.bitwise_count(h^a[:,0]),np.bitwise_count(h^a[:,1]))<=4) for a in [gallery_hashes,clinical_hash]):blocked.add(row['patient'])
    kept=[r for r in rows if r['patient'] not in blocked];patients=sorted({r['patient'] for r in kept});labels={r['patient']:r['label'] for r in kept}
    if len(patients)<30:raise RuntimeError('insufficient_expert_cohort_after_audit')
    cohorts=collections.defaultdict(list)
    for patient in patients:cohorts[labels[patient]].append(patient)
    # Fix support/query patients across k; at least ten support patients per disease.
    split={}
    for label,ids in sorted(cohorts.items()):
        if len(ids)<11:continue
        rng=np.random.default_rng(20261031);ids=np.asarray(sorted(ids));rng.shuffle(ids);nquery=min(max(1,round(.2*len(ids))),len(ids)-10)
        split[label]=dict(query=ids[:nquery].tolist(),support=ids[nquery:].tolist())
    save(OUT/'private/cohort.json',dict(rows=kept,patients=patients,labels=labels,fewshot_split=split))
    report=dict(state='passed',source_release_sha256=release['release_sha256'],cohort_sha256=digest(OUT/'private/cohort.json'),patients=len(patients),images=len(kept),diseases=len(cohorts),
        additional_quarantined_patients=len(blocked),prior_quarantine=release['near_duplicate_patient_quarantine'],CPT_group_and_patient_key_overlap_after_filter=0,
        current_GMDB_training_subset_of_previously_audited_training=True,fresh_RGB112_CPT_and_GMDB_training_perceptual_check=True,
        fewshot_common_diseases=len(split),fewshot_fixed_query_patients=sum(len(v['query']) for v in split.values()),
        expert_used_for_model_selection=False,expert_used_for_optimization=False,
        limitations=['Previously inspected expert-source cohort; not untouched or prospective validation.','Expert-source diagnoses await new clinical adjudication.','Identity aliases without detectable visual overlap remain incompletely resolved.','Few-shot support is an expert reference gallery, not gradient-based training.'],updated_at_utc=now())
    save(OUT/'audit.json',report)

def worker(name):
    import torch,cv2
    from torch.utils.data import Dataset,DataLoader
    from network import Encoder
    audit=load(OUT/'audit.json');assert digest(OUT/'private/cohort.json')==audit['cohort_sha256'];rows=load(OUT/'private/cohort.json')['rows']
    initial=name.startswith('init__');case=dict(arm=name.removeprefix('init__'),regime='wce_feature_only') if initial else load(WORK/'cases.json')['cases'][name]
    m=Encoder(case['arm'],case['regime'],load(WORK/'audit.json')['diseases'])
    if not initial:
        selected=load(WORK/'cases'/name/'result.json');assert digest(selected['path'])==selected['sha256'];s=torch.load(selected['path'],map_location='cpu',weights_only=False);assert s['epoch']==50;m.load_state_dict(s['model'],strict=True)
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;m=m.cuda().eval().requires_grad_(False)
    class Inputs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('expert_crop_changed')
            x=cv2.imread(r['crop_path']);x=cv2.cvtColor(cv2.resize(x,(112,112)),cv2.COLOR_BGR2RGB)
            return torch.from_numpy(x.transpose(2,0,1).copy()).float().div_(127.5).sub_(1)
    pieces=[]
    with torch.inference_mode():
        for x in DataLoader(Inputs(),batch_size=128,num_workers=2,pin_memory=True):pieces.append(m.embed(x.cuda(non_blocking=True)).cpu().numpy())
    np.save(OUT/'private'/f'{name}.npy',np.concatenate(pieces));save(OUT/'receipts'/f'{name}.json',dict(state='complete',sha256=digest(OUT/'private'/f'{name}.npy'),updated_at_utc=now()))

def score():
    import importlib.util
    spec=importlib.util.spec_from_file_location('matched_expert_statistics',WORK/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    normalized,retrieval,macro_score,interval=module.normalized,module.retrieval,module.macro_score,module.interval
    selection=load(WORK/'selection.json');data=load(OUT/'private/cohort.json');rows=data['rows'];patients=data['patients'];labels=[data['labels'][i] for i in patients];counts=collections.Counter(labels)
    names=sorted({n for cells in selection['settings'].values() for names in cells.values() for n in names}|set(selection['frozen_initializations']));metrics={};hits={};queries=[i for i,l in enumerate(labels) if counts[l]>=2]
    for name in names:
        path=OUT/'private'/f'{name}.npy';assert digest(path)==load(OUT/'receipts'/f'{name}.json')['sha256'];f=np.load(path);grouped=collections.defaultdict(lambda:collections.defaultdict(list))
        for row,v in zip(rows,f):grouped[row['patient']][row['view']].append(v)
        x=normalized(np.stack([np.mean([normalized(np.mean(v,axis=0,keepdims=True))[0] for v in grouped[i].values()],axis=0) for i in patients]))
        h,_=retrieval(x[queries],x,[patients[i] for i in queries],patients,[{labels[i]} for i in queries],[{l} for l in labels]);truth=[labels[i] for i in queries]
        metrics[name]=dict(retrieval={f'top{k}':macro_score(v,[{l} for l in truth]) for k,v in h.items()},query_patients=len(queries),gallery_patients=len(patients),query_diseases=len(set(truth)),fewshot={})
        hits[name]=h[1]
        index={i:j for j,i in enumerate(patients)};split=data['fewshot_split'];vocab=sorted(split)
        if len(vocab)>=2:
            qids=[i for l in vocab for i in split[l]['query']];qy=[data['labels'][i] for i in qids]
            assert not set(qids)&{i for l in vocab for i in split[l]['support']}
            for k in [1,5,10]:
                rng=np.random.default_rng(20261031+k);scores={n:[] for n in [1,5,10]}
                for _ in range(20):
                    proto=normalized(np.stack([x[[index[i] for i in rng.choice(split[l]['support'],size=k,replace=False)]].mean(0) for l in vocab]));order=np.argsort(-(x[[index[i] for i in qids]]@proto.T),axis=1,kind='stable')
                    for n in scores:scores[n].append(macro_score([qy[i] in {vocab[j] for j in rank[:n]} for i,rank in enumerate(order)],[{v} for v in qy]))
                metrics[name]['fewshot'][str(k)]={f'top{n}':float(np.mean(v)) for n,v in scores.items()}|dict(query_patients=len(qids),candidate_diseases=len(vocab),support_repeats=20)
    summary={};paired={}
    for setting,cells in selection['settings'].items():
        summary[setting]={cell:dict(retrieval={f'top{k}_mean':float(np.mean([metrics[n]['retrieval'][f'top{k}'] for n in names])) for k in [1,5,10]},retrieval_top1_SD=float(np.std([metrics[n]['retrieval']['top1'] for n in names],ddof=1)),fewshot={str(k):{f'top{n}_mean':float(np.mean([metrics[c]['fewshot'][str(k)]['top'+str(n)] for c in names])) for n in [1,5,10]} for k in [1,5,10]} if all(str(k) in metrics[c]['fewshot'] for c in names for k in [1,5,10]) else {}) for cell,names in cells.items()}
        paired[setting]={}
        for regime in load(WORK/'protocol.json')['regimes']:
            for base in ['r50','transface_original']:
                a=cells[base+'__'+regime];b=cells['transface_chop_cpt__'+regime];paired[setting]['CPT_minus_'+base+'__'+regime]=interval(np.stack([hits[n] for n in a]),np.stack([hits[n] for n in b]),truth)
    save(OUT/'results.json',dict(state='complete',audit=load(OUT/'audit.json'),selection_source_sha256=digest(WORK/'selection.json'),summary=summary,paired_retrieval_top1=paired,frozen_initializations={n:metrics[n] for n in selection['frozen_initializations']},all_seeds_retained=True,expert_labels_used_for_model_selection=False,updated_at_utc=now()))

def main():
    import fcntl
    sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
    from matched_runtime import free_jobs,environment,SLURM,notify
    os.umask(0o077)
    for folder in ['private','receipts','logs']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not load(OUT/'audit.json'):prepare()
    save(OUT/'status.json',dict(state='waiting_for_GMDB_locked_selection',updated_at_utc=now()))
    while not load(WORK/'gpu_work_complete.json'):time.sleep(30)
    selection=load(WORK/'selection.json');names=sorted({n for cells in selection['settings'].values() for names in cells.values() for n in names}|set(selection['frozen_initializations']))
    for name in names:
        if load(OUT/'receipts'/f'{name}.json').get('state')=='complete':continue
        while not free_jobs():time.sleep(20)
        job=free_jobs()[0];e=environment();e['PYTHONPATH']=str(WORK/'dependencies')+':'+str(WORK/'vendor')+':'+str(PROJECT/'arcface_cpt_20261026/dependencies')+':'+e['PYTHONPATH']
        cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name=matched-CHOP-expert-'+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'expert_evaluate.py'),'worker',name]
        save(OUT/'status.json',dict(state='expert_inference',model=name,completed=sum(load(OUT/'receipts'/f'{n}.json').get('state')=='complete' for n in names),total=len(names),updated_at_utc=now()))
        with (OUT/'logs'/(name+'.private.log')).open('ab') as f:subprocess.run(cmd,env=e,stdout=f,stderr=f,check=True)
    score();save(OUT/'status.json',dict(state='complete',updated_at_utc=now()));notify('Matched CHOP expert comparison complete',__import__('json').dumps(load(OUT/'results.json'),indent=2))

if __name__=='__main__':
    try:
        if len(sys.argv)>1:worker(sys.argv[2])
        else:main()
    except Exception as e:save(OUT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

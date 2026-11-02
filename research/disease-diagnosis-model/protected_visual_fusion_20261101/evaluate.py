"""Frozen, independent visual/HPO branches. Patient artifacts remain on HPC."""
import collections
import csv
import fcntl
import subprocess
import time
import numpy as np
from common import *

STAGED = PROJECT / 'hpo_fusion_staged_20261031'
PRIOR = PROJECT / 'chop_hpo_increment_20261101'


def norm(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def mix_scores(visual, text, available, alpha):
    if alpha == 0:
        return visual.copy()
    return np.where(available, (1-alpha)*visual + alpha*text, visual)


def macro(hit, labels):
    return float(np.mean([hit[...,labels==c].mean() for c in np.unique(labels)]))


def classify(scores, labels):
    order=np.argsort(-scores,axis=1,kind='stable')
    return {k:(order[:,:k]==labels[:,None]).any(1) for k in [1,5,10]}


def retrieve(scores, truth, gallery_truth):
    order=np.argsort(-scores,axis=1,kind='stable')
    return {k:(gallery_truth[order[:,:k]]==truth[:,None]).any(1) for k in [1,5,10]}


def scalar(hits, labels):
    return {f'top{k}':macro(v,labels) for k,v in hits.items()}


def prepare():
    protocol=load(WORK/'protocol.json')
    receipt=load(STAGED/'data_receipt.json')
    for name in ['gmdb.npz','term_index.json']:
        assert digest(WORK/'private'/name)==receipt['files'][name]
    patient_path=PROJECT/'hpo_fusion_public_20261031/private/patient_index.json'
    patients=load(patient_path)['patients']
    classes=sorted({r['label'] for r in patients if r['split']=='train'})
    mapping={v:i for i,v in enumerate(classes)}
    with np.load(WORK/'private/gmdb.npz') as d:
        train,val=d['train'],d['validation']
        assert len(d['labels'])==len(patients)
        assert all(mapping.get(r['label'],-1)==d['labels'][i] for i,r in enumerate(patients))
        assert not set(train)&set(val)
        assert not {patients[i]['subject'] for i in train}&{patients[i]['subject'] for i in val}
        assert len({patients[i]['subject'] for i in train})==len(train)
        assert len({patients[i]['subject'] for i in val})==len(val)
        assert len(val)==320 and np.isfinite(d['features']).all()
        counts=dict(train_patients=len(train),validation_patients=len(val),diseases=len(classes),
                    validation_diseases=len(np.unique(d['labels'][val])),visual_dimensions=d['features'].shape[1])
    baseline_labels=load(MATCH/'private/records.private.json')['labels']
    assert all(c in baseline_labels for c in classes)
    class_indices=[baseline_labels.index(c) for c in classes]
    save(WORK/'private/class_indices.json',dict(indices=class_indices))
    vis=protocol['visual_encoder'];assert digest(vis['path'])==vis['sha256']
    pins={}
    for seed in protocol['seeds']:
        for key,source in [('hpo',STAGED/'cases'/f'public_pretrained_knowledge__hpo__seed{seed}'),('reference',PRIOR/'cases'/f'no_chop__seed{seed}')]:
            r=load(source/'result.json')
            assert r['state']=='complete' and digest(r['checkpoint_path'])==r['checkpoint_sha256']
            pins[f'{key}_{seed}']=dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'])
    ep=STAGED/'features/term_embeddings.npy';assert digest(ep)==load(STAGED/'term_embedding_receipt.json')['sha256']
    save(WORK/'input_receipt.json',dict(state='passed',counts=counts,model_counts=receipt['counts'],checkpoints=pins,
        gmdb_sha256=digest(WORK/'private/gmdb.npz'),term_embeddings_sha256=digest(ep),
        patient_index_sha256=digest(patient_path),class_indices_sha256=digest(WORK/'private/class_indices.json'),
        visual_checkpoint_sha256=vis['sha256'],patient_overlap=0,new_training=False,clinical_test_used=False,updated_at_utc=now()))


def worker():
    import torch
    from model import PhenotypeModel
    torch.set_num_threads(4)
    protocol=load(WORK/'protocol.json');receipt=load(WORK/'input_receipt.json')
    assert digest(WORK/'private/gmdb.npz')==receipt['gmdb_sha256']
    assert digest(WORK/'private/class_indices.json')==receipt['class_indices_sha256']
    with np.load(WORK/'private/gmdb.npz') as d:data={k:np.array(d[k]) for k in d.files}
    visual=data['features'];train,val=data['train'],data['validation'];truth=data['labels'][val]
    has=data['ids'].any(1)
    # The raw visual branch is separate from all fusion operations.
    raw_retrieval=norm(visual[val])@norm(visual[train]).T
    base_hits=retrieve(raw_retrieval,truth,data['labels'][train])
    path=protocol['visual_encoder']['path'];assert digest(path)==receipt['visual_checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)['model']
    weights=state['head.weight'][load(WORK/'private/class_indices.json')['indices']].float()
    del state
    logits=64*torch.from_numpy(norm(visual)).float()@torch.nn.functional.normalize(weights,dim=1).T
    visual_prob=(logits/protocol['visual_probability_temperature']).softmax(-1).numpy()
    del logits,weights
    tensors={k:torch.from_numpy(data[k]).cuda() for k in ['features','ids','states']}
    ep=STAGED/'features/term_embeddings.npy';assert digest(ep)==receipt['term_embeddings_sha256']
    embeddings=torch.from_numpy(np.load(ep));counts=receipt['model_counts']
    records={};all_hits={}
    with torch.inference_mode():
        for seed in protocol['seeds']:
            output={}
            for mode in ['hpo','reference']:
                pin=receipt['checkpoints'][f'{mode}_{seed}'];assert digest(pin['path'])==pin['sha256']
                model=PhenotypeModel(embeddings,counts['GMDB_classes'],counts['public_training_diseases'],'hpo' if mode=='hpo' else 'fusion_selected').cuda().eval().requires_grad_(False)
                state=torch.load(pin['path'],map_location='cpu',weights_only=False)['model']
                if state['assertion.weight'].shape[0]==2:
                    w=state['assertion.weight'];state['assertion.weight']=torch.cat([w,w.mean(0,keepdim=True)],0)
                model.load_state_dict(state,strict=True);del state
                ss,zz=[],[]
                for start in range(0,len(visual),64):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        s,z=model(tensors['features'][start:start+64],tensors['ids'][start:start+64],tensors['states'][start:start+64])
                    ss.append(s.float());zz.append(z.float().cpu().numpy())
                output[mode]=(torch.cat(ss).softmax(-1).cpu().numpy(),norm(np.concatenate(zz)))
                del model
            hprob,hvec=output['hpo'];rprob,rvec=output['reference']
            text_retrieval=hvec[val]@hvec[train].T
            availability=has[val,None]&has[train][None,:]
            diag_available=has[val,None]
            rng=np.random.default_rng(seed+20261101);order=rng.permutation(len(val));permutation=np.empty(len(val),np.int64)
            permutation[order]=np.roll(order,1)
            assert np.all(permutation!=np.arange(len(val)))
            records[str(seed)]={'grid':{},'reference':{}}
            reference=dict(diagnosis=classify(rprob[val],truth),retrieval=retrieve(rvec[val]@rvec[train].T,truth,data['labels'][train]))
            for task,hits in reference.items():
                records[str(seed)]['reference'][task]=scalar(hits,truth)
                for k,v in hits.items():all_hits[f's{seed}_reference_{task}_{k}']=v
            for alpha in protocol['alphas']:
                tag=str(alpha);scores=mix_scores(raw_retrieval,text_retrieval,availability,alpha)
                prob=mix_scores(visual_prob[val],hprob[val],diag_available,alpha)
                if alpha==0:
                    assert np.array_equal(scores,raw_retrieval)
                    assert np.array_equal(prob,visual_prob[val])
                assert np.array_equal(scores[~availability],raw_retrieval[~availability])
                for shuffled in [False,True]:
                    kind='shuffled' if shuffled else 'real'
                    ss=mix_scores(raw_retrieval,text_retrieval[permutation],has[val[permutation],None]&has[train][None,:],alpha) if shuffled else scores
                    pp=mix_scores(visual_prob[val],hprob[val[permutation]],has[val[permutation],None],alpha) if shuffled else prob
                    outcomes=dict(diagnosis=classify(pp,truth),retrieval=retrieve(ss,truth,data['labels'][train]))
                    records[str(seed)]['grid'][f'{tag}_{kind}']={task:scalar(h,truth) for task,h in outcomes.items()}
                    for task,h in outcomes.items():
                        for k,v in h.items():all_hits[f's{seed}_a{tag}_{kind}_{task}_{k}']=v
            save(WORK/'status.json',dict(state='frozen_branch_evaluation',completed_seeds=len(records),total_seeds=4,updated_at_utc=now()))
    assert digest(WORK/'private/gmdb.npz')==receipt['gmdb_sha256']
    assert digest(path)==receipt['visual_checkpoint_sha256']
    np.savez(WORK/'private/hits.npz',**all_hits,labels=truth)
    save(WORK/'per_seed_metrics.json',dict(seeds=records,raw_visual_retrieval=scalar(base_hits,truth),
        visual_branch_bitwise_preserved=True,missing_HPO_falls_back_exactly=True,no_network_weights_updated=True,updated_at_utc=now()))


def interval(a,b,labels):
    delta=b.astype(float)-a.astype(float);groups=[np.flatnonzero(labels==c) for c in np.unique(labels)];rng=np.random.default_rng(20261101);values=[]
    for _ in range(2000):
        seeds=rng.choice(len(delta),len(delta),replace=True);chosen=rng.choice(len(groups),len(groups),replace=True)
        values.append(np.mean([delta[seeds][:,rng.choice(groups[c],len(groups[c]),replace=True)].mean() for c in chosen]))
    return dict(delta=macro(delta,labels),ci95=np.quantile(values,[.025,.975]).tolist(),interpretation='development interval conditional on selected weight; not selection-adjusted')


def summarize():
    protocol=load(WORK/'protocol.json');report=load(WORK/'per_seed_metrics.json');seeds=protocol['seeds'];grid={};reference={}
    for task in ['diagnosis','retrieval']:
        grid[task]={}
        for alpha in protocol['alphas']:
            rs=[report['seeds'][str(s)]['grid'][f'{alpha}_real'][task] for s in seeds]
            grid[task][str(alpha)]={f'top{k}_{stat}':float(fn([r[f'top{k}'] for r in rs])) for k in [1,5,10] for stat,fn in [('mean',np.mean),('SD',lambda x:np.std(x,ddof=1))]}
        reference[task]={f'top{k}_mean':float(np.mean([report['seeds'][str(s)]['reference'][task][f'top{k}'] for s in seeds])) for k in [1,5,10]}
    selection={task:max(protocol['alphas'],key=lambda a:(grid[task][str(a)]['top1_mean'],grid[task][str(a)]['top5_mean'],-a)) for task in grid}
    effects={}
    with np.load(WORK/'private/hits.npz') as h:
        labels=h['labels']
        for task,alpha in selection.items():
            effects[task]={}
            for control in ['visual','HPO','shuffled','cross_attention']:
                effects[task][control]={}
                for k in [1,5]:
                    b=np.stack([h[f's{s}_a{alpha}_real_{task}_{k}'] for s in seeds])
                    suffix=f'a0.0_real_{task}_{k}' if control=='visual' else f'a1.0_real_{task}_{k}' if control=='HPO' else f'a{alpha}_shuffled_{task}_{k}' if control=='shuffled' else f'reference_{task}_{k}'
                    a=np.stack([h[f's{s}_{suffix}'] for s in seeds]);effects[task][control][f'top{k}']=interval(a,b,labels)
    save(WORK/'results.json',dict(state='complete',grid=grid,existing_cross_attention=reference,development_selected_alpha=selection,
        selected_effects=effects,raw_visual_retrieval=report['raw_visual_retrieval'],
        visual_branch_bitwise_preserved=True,missing_HPO_falls_back_exactly=True,new_training=False,clinical_test_used=False,
        scope='same GMDB patient-disjoint development only; weight selection is exploratory; no CHOP expert retuning',updated_at_utc=now()))
    save(WORK/'selection.json',dict(state='development_only_not_clinical_deployment',independent_visual_encoder=protocol['visual_encoder'],
        native512_visual_retrieval_always_retained=True,optional_multimodal_alpha=selection,
        diagnosis_replace_existing=grid['diagnosis'][str(selection['diagnosis'])]['top1_mean']>reference['diagnosis']['top1_mean'],
        expert_or_BCH_used=False,updated_at_utc=now()))


def main():
    os.umask(0o077);lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in load(WORK/'source_receipt.json').items():assert digest(WORK/n)==h
    if load(WORK/'results.json').get('state')=='complete':return
    if not load(WORK/'input_receipt.json'):prepare()
    if not load(WORK/'per_seed_metrics.json'):
        save(WORK/'status.json',dict(state='waiting_for_free_GPU',updated_at_utc=now()))
        jobs=free_jobs()
        while not jobs:time.sleep(15);jobs=free_jobs()
        job=jobs[0];save(WORK/'launch.json',dict(GPU_job=job['id'],updated_at_utc=now()))
        cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name=protected-visual-HPO-eval','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=24G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'evaluate.py'),'worker']
        with (WORK/'logs/worker.private.log').open('ab') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=f,check=True)
    for _ in range(30):
        if len(load(WORK/'per_seed_metrics.json').get('seeds',{}))==4:break
        time.sleep(5)
    if len(load(WORK/'per_seed_metrics.json').get('seeds',{}))!=4:raise RuntimeError('metrics_not_visible')
    save(WORK/'status.json',dict(state='summarizing',updated_at_utc=now()))
    summarize();save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))
    try:
        notify('Independent visual HPO comparison complete',json.dumps(load(WORK/'results.json'),indent=2))
        save(WORK/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:save(WORK/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))


if __name__=='__main__':
    try:
        if len(sys.argv)>1:worker()
        else:main()
    except Exception as e:
        save(WORK/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise

"""Frozen expert transfer test. Only aggregate metrics leave private HPC artifacts."""
import collections
import csv
import re
import sqlite3
import subprocess
import time
import fcntl
import numpy as np
from common import *

OUT = WORK / 'expert'
SOURCE = MATCH / 'expert'
LINK = PROJECT / 'chop_patient_linkage_20261101'


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prepare():
    for name in ['private', 'logs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    source_audit = load(SOURCE / 'audit.json')
    cohort_path = SOURCE / 'private/cohort.json'
    assert source_audit['state'] == 'passed' and digest(cohort_path) == source_audit['cohort_sha256']
    cohort = load(cohort_path)
    selection = load(WORK / 'visual_selection.json')
    feature_path = SOURCE / 'private' / (selection['case'] + '.npy')
    feature_receipt = load(SOURCE / 'receipts' / (selection['case'] + '.json'))
    assert digest(feature_path) == feature_receipt['sha256']
    assert digest(selection['path']) == selection['sha256']
    vectors = np.load(feature_path, allow_pickle=False)
    assert len(vectors) == len(cohort['rows']) and np.isfinite(vectors).all()
    new_train_groups = set(load(WORK / 'private/chop_groups.json')['groups'])
    stage = PROJECT / 'transface_arcface_identity_cpt_20261026'
    spec = load(stage / 'protocol.json')['train_manifest']
    assert digest(spec['path']) == spec['sha256']
    with Path(spec['path']).open() as f:
        train = list(csv.DictReader(f, delimiter='\t'))
    train_keys = {r['patient_key'] for r in train}
    train_groups = {r['group_id'] for r in train} | new_train_groups
    train_hashes = {r['crop_sha256'] for r in train}
    grouped = collections.defaultdict(list)
    for index, r in enumerate(cohort['rows']):
        assert r['patient_key'] not in train_keys and r['group_id'] not in train_groups
        assert r['crop_sha256'] not in train_hashes
        grouped[r['group_id'] or r['patient_key']].append((index, r))
    # A conflicting alias group cannot become independent observations.
    conflict = {g for g, rr in grouped.items() if len({r['label'] for _, r in rr}) != 1}
    for g in conflict: del grouped[g]
    term_ids = load(WORK / 'private/term_index.json')['term_ids']
    term_map = {h:i+1 for i,h in enumerate(term_ids)}
    hp = collections.defaultdict(set)
    keys = {r['patient_key'] for rr in grouped.values() for _,r in rr}
    db = sqlite3.connect(f'file:{LINK / "private/linked_metadata.sqlite"}?mode=ro', uri=True)
    for key, code, assertion in db.execute("SELECT patient_key,code,assertion FROM hpo WHERE source='Arcus_clinician_HPO'"):
        if key in keys:
            assert assertion == 'recorded_polarity_unresolved'
            if code in term_map: hp[key].add(code)
    db.close()
    # Rebuild the trained class order and check it against the sealed GMDB arrays.
    patients = load(PROJECT / 'hpo_fusion_public_20261031/private/patient_index.json')['patients']
    classes = sorted({r['label'] for r in patients if r['split'] == 'train'})
    class_map = {v:i for i,v in enumerate(classes)}
    with np.load(WORK / 'private/gmdb.npz') as d:
        assert len(patients) == len(d['labels'])
        assert all(class_map.get(r['label'],-1) == d['labels'][i] for i,r in enumerate(patients))
    xrefs = collections.defaultdict(set)
    ontology = PROJECT / 'ontology/mondo.obo'
    for block in ontology.read_text().split('[Term]')[1:]:
        mid = re.search(r'^id: (MONDO:\d{7})$', block, re.M)
        if not mid or 'is_obsolete: true' in block: continue
        for line in block.splitlines():
            m = re.match(r'xref: (OMIM:\d+)',line)
            if m and 'MONDO:equivalentTo' in line: xrefs[m[1]].add(mid[1])
            m = re.match(r'property_value: skos:exactMatch (OMIM:\d+)',line)
            if m: xrefs[m[1]].add(mid[1])
    metadata = Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/gmdb_syndromes_v1.1.0.tsv')
    mapping = {}
    with metadata.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            omim = sorted(set(re.findall(r'\d{6}', r['OMIM'])))
            mids = xrefs.get('OMIM:'+omim[0],set()) if len(omim)==1 else set()
            if len(mids)==1: mapping[r['syndrome_id']] = next(iter(mids))
    # Keep unmapped classifier outputs as distractors; never shrink to expert diagnoses.
    candidate_for_class = [mapping.get(c,'UNMAPPED_GMDB:'+str(c)) for c in classes]
    candidates = sorted(set(candidate_for_class))
    index = {c:i for i,c in enumerate(candidates)}
    class_to_candidate = np.array([index[c] for c in candidate_for_class])
    features, ids, states, labels, group_ids = [], [], [], [], []
    for g, rr in sorted(grouped.items()):
        codes = sorted(set().union(*(hp[r['patient_key']] for _,r in rr)))
        ii, ss = np.zeros(128,np.int64), np.zeros(128,np.int64)
        for j,c in enumerate(codes[:128]): ii[j],ss[j]=term_map[c],2
        ids.append(ii);states.append(ss)
        # Same per-patient image mean as the GMDB fusion training pipeline.
        features.append(normalize(vectors[[i for i,_ in rr]].mean(0,keepdims=True))[0])
        labels.append(rr[0][1]['label']);group_ids.append(g)
    y = np.array([index.get(l,-1) for l in labels],np.int64)
    ids = np.stack(ids);states=np.stack(states)
    has = ids.any(axis=1);covered=y>=0
    counts = collections.Counter(labels)
    queries = np.array([i for i,l in enumerate(labels) if counts[l]>=2],np.int64)
    np.savez(OUT / 'private/arrays.npz',features=np.stack(features),ids=ids,states=states,labels=y,
        has_HPO=has,covered=covered,queries=queries,class_to_candidate=class_to_candidate)
    save(OUT / 'private/index.json',dict(groups=group_ids,labels=labels,candidates=candidates))
    cases = load(WORK / 'cases.json')['cases']
    pins = {}
    for name in cases:
        r=load(WORK / 'cases' / name / 'result.json')
        assert r['state']=='complete' and digest(r['checkpoint_path'])==r['checkpoint_sha256']
        pins[name]=dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'],GMDB_selected_epoch=r['selected_epoch'])
    save(OUT / 'protocol.json',dict(state='locked_before_expert_inference',cases=pins,
        primary='closed-set disease classification on expert groups with recorded HPO and covered diagnosis',
        secondary=['all covered expert groups','all expert groups including uncovered diseases','leave-one-group-out phenotype retrieval'],
        diagnosis_candidates='full GMDB classifier vocabulary; sum class probabilities for exact MONDO equivalents; retain unmapped distractors',
        HPO_state='2 = clinician recorded, polarity unresolved; never relabeled positive',
        retrieval='cosine top-k patient groups; query self excluded; same gallery and queries in every arm',
        expert_checkpoint_selection=False,expert_training=False,retuning=False,
        model_inference_precision='BF16 forward, FP32 probability and ranking; stable ties',
        arrays_sha256=digest(OUT / 'private/arrays.npz'),index_sha256=digest(OUT / 'private/index.json'),
        model_source_sha256=digest(WORK / 'model.py'),feature_sha256=feature_receipt['sha256'],
        mondo_sha256=digest(ontology),GMDB_disease_metadata_sha256=digest(metadata),updated_at_utc=now()))
    audit=dict(state='passed',expert_patient_groups=len(labels),expert_images=sum(len(rr) for rr in grouped.values()),
        expert_diseases=len(counts),conflicting_alias_groups_excluded=len(conflict),HPO_paired_groups=int(has.sum()),
        classifier_covered_groups=int(covered.sum()),paired_and_covered_groups=int((has&covered).sum()),
        paired_and_covered_diseases=len({l for l,h,c in zip(labels,has,covered) if h and c}),
        retrieval_queries=len(queries),retrieval_query_diseases=len({labels[i] for i in queries}),
        classifier_classes=len(classes),ranked_diagnosis_candidates=len(candidates),
        prior_CPT_and_new_CHOP_group_overlap=0,prior_CPT_patient_key_and_crop_hash_overlap=0,
        prior_GMDB_and_CPT_near_duplicate_audit_reused=True,clinical_images_read=False,
        limitations=['Previously inspected expert cohort; exploratory transfer, not untouched external validation',
                    'Expert-source diagnoses and clinician HPO timing/polarity remain unadjudicated',
                    'Known patient/group and image overlap checked; unknown identity aliases not certified'],updated_at_utc=now())
    save(OUT / 'audit.json',audit)


def hit_summary(hits, labels):
    if not len(labels):return dict(patients=0,diseases=0)
    groups=[np.flatnonzero(np.asarray(labels)==c) for c in sorted(set(labels))]
    return dict(patients=len(labels),diseases=len(groups),**{f'top{k}':float(np.mean([v[g].mean() for g in groups])) for k,v in hits.items()})


def retrieval(x, labels, queries):
    x=normalize(x);score=x[queries]@x.T
    score[np.arange(len(queries)),queries]=-np.inf
    ranks=np.argsort(-score,axis=1,kind='stable')
    labs=np.asarray(labels)
    return {k:(labs[ranks[:,:k]]==labs[queries,None]).any(1) for k in [1,5,10]}


def worker():
    import torch
    from model import PhenotypeModel
    torch.set_num_threads(4)
    protocol=load(OUT / 'protocol.json')
    assert digest(WORK / 'model.py')==protocol['model_source_sha256']
    assert digest(OUT / 'private/arrays.npz')==protocol['arrays_sha256']
    assert digest(OUT / 'private/index.json')==protocol['index_sha256']
    with np.load(OUT / 'private/arrays.npz') as d: data={k:np.array(d[k]) for k in d.files}
    index=load(OUT / 'private/index.json');labels=index['labels'];queries=data['queries']
    counts=load(WORK / 'data_receipt.json')['counts']
    embeddings=torch.from_numpy(np.load(WORK / 'features/term_embeddings.npy'))
    features=torch.from_numpy(data['features']).cuda();ids=torch.from_numpy(data['ids']).cuda();states=torch.from_numpy(data['states']).cuda()
    masks=dict(paired_covered=data['has_HPO']&data['covered'],covered=data['covered'],all_expert=np.ones(len(labels),bool),missing_HPO_covered=~data['has_HPO']&data['covered'])
    metrics={};private={}
    with torch.inference_mode():
        for name,pin in protocol['cases'].items():
            assert digest(pin['path'])==pin['sha256']
            model=PhenotypeModel(embeddings,counts['GMDB_classes'],counts['public_training_diseases'],'fusion_selected').cuda().eval()
            model.load_state_dict(torch.load(pin['path'],map_location='cpu',weights_only=False)['model'],strict=True)
            logits,representations=[],[]
            for start in range(0,len(labels),64):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    a,b=model(features[start:start+64],ids[start:start+64],states[start:start+64])
                logits.append(a.float());representations.append(b.float().cpu().numpy())
            probabilities=torch.cat(logits).softmax(-1).cpu().numpy()
            scores=np.zeros((len(labels),len(index['candidates'])),np.float32)
            for i,c in enumerate(data['class_to_candidate']):scores[:,c]+=probabilities[:,i]
            rank=np.argsort(-scores,axis=1,kind='stable')
            hits={k:(rank[:,:k]==data['labels'][:,None]).any(1) for k in [1,5,10]}
            rh=retrieval(np.concatenate(representations),labels,queries)
            metrics[name]={group:hit_summary({k:v[mask] for k,v in hits.items()},np.asarray(labels)[mask]) for group,mask in masks.items()}
            metrics[name]['retrieval']=hit_summary(rh,np.asarray(labels)[queries])
            private[name]={**{f'diagnosis_{k}':v for k,v in hits.items()},**{f'retrieval_{k}':v for k,v in rh.items()}}
            np.savez(OUT / 'private' / (name+'_hits.npz'),**private[name])
            save(OUT / 'status.json',dict(state='expert_inference',completed=len(metrics),total=len(protocol['cases']),updated_at_utc=now()))
            del model
    raw=retrieval(data['features'],labels,queries)
    save(OUT / 'case_metrics.json',dict(models=metrics,frozen_visual_retrieval=hit_summary(raw,np.asarray(labels)[queries]),updated_at_utc=now()))


def summarize():
    metrics=load(OUT/'case_metrics.json');conditions=load(WORK/'protocol.json')['conditions'];seeds=load(WORK/'protocol.json')['seeds']
    summary={}
    for c in conditions:
        names=[f'{c}__seed{s}' for s in seeds];summary[c]={}
        for task in metrics['models'][names[0]]:
            first=metrics['models'][names[0]][task]
            summary[c][task]={k:first[k] for k in ['patients','diseases']}
            for k in [1,5,10]:
                if f'top{k}' not in first:continue
                vals=[metrics['models'][n][task][f'top{k}'] for n in names]
                summary[c][task].update({f'top{k}_mean':float(np.mean(vals)),f'top{k}_SD':float(np.std(vals,ddof=1))})
    with np.load(OUT/'private/arrays.npz') as d:
        mask=d['has_HPO']&d['covered'];queries=d['queries']
    labels=np.array(load(OUT/'private/index.json')['labels'])
    effects={}
    for task,truth,indices in [('paired_covered',labels[mask],mask),('retrieval',labels[queries],slice(None))]:
        effects[task]={}
        groups=[np.flatnonzero(truth==c) for c in sorted(set(truth))]
        if not groups:continue
        for control in ['no_chop','chop_shuffled']:
            effects[task][control]={}
            for k in [1,5]:
                field=f'diagnosis_{k}' if task=='paired_covered' else f'retrieval_{k}'
                def read(c):
                    vv=[]
                    for seed in seeds:
                        with np.load(OUT/'private'/f'{c}__seed{seed}_hits.npz') as d:vv.append(d[field][indices].astype(float))
                    return np.stack(vv)
                delta=read('chop_recorded')-read(control);rng=np.random.default_rng(20261101);boot=[]
                for _ in range(2000):
                    ss=rng.choice(len(seeds),len(seeds),replace=True);gg=rng.choice(len(groups),len(groups),replace=True)
                    boot.append(np.mean([delta[ss][:,rng.choice(groups[g],len(groups[g]),replace=True)].mean() for g in gg]))
                effects[task][control][f'top{k}']=dict(delta=float(np.mean([delta[:,g].mean() for g in groups])),ci95=np.quantile(boot,[.025,.975]).tolist())
    save(OUT/'results.json',dict(state='complete',summary=summary,CHOP_increment_paired=effects,
        frozen_visual_retrieval=metrics['frozen_visual_retrieval'],audit=load(OUT/'audit.json'),
        all_four_seeds_retained=True,expert_used_for_checkpoint_selection=False,scope='exploratory CHOP expert transfer',updated_at_utc=now()))


def main():
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(OUT/'results.json').get('state')=='complete':return
    if not load(OUT/'protocol.json'):prepare()
    if not load(OUT/'case_metrics.json'):
        save(OUT/'status.json',dict(state='waiting_for_free_GPU',updated_at_utc=now()))
        jobs=free_jobs()
        while not jobs:time.sleep(15);jobs=free_jobs()
        job=jobs[0]
        cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name=CHOP-expert-fusion-test','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'expert_evaluate.py'),'worker']
        save(OUT/'launch.json',dict(job_id=job['id'],updated_at_utc=now()))
        with (OUT/'logs/worker.private.log').open('ab') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=f,check=True)
    # Shared filesystem metadata may lag behind a successful remote worker.
    for _ in range(24):
        if len(load(OUT/'case_metrics.json').get('models',{})) == 12:break
        time.sleep(5)
    if len(load(OUT/'case_metrics.json').get('models',{})) != 12:
        raise RuntimeError('expert_metric_receipt_not_visible')
    summarize()
    save(OUT/'status.json',dict(state='complete',completed=12,total=12,updated_at_utc=now()))
    if (OUT/'failure.json').exists():
        (OUT/'failure.json').replace(OUT/'previous_failure.json')
    try:
        notify('CHOP expert paired-data comparison complete',json.dumps(load(OUT/'results.json'),indent=2))
        save(OUT/'notification.json',dict(state='local_notifier_invoked',updated_at_utc=now()))
    except Exception as e:save(OUT/'notification_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))


if __name__=='__main__':
    try:
        if len(sys.argv)>1:worker()
        else:main()
    except Exception as e:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        raise

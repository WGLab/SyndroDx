"""Patient-level metrics and paired uncertainty; private IDs never exported."""
import os,sys,csv,json,collections
import numpy as np
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import V1
sys.path.insert(0,str(V1))
from evaluation_core import normalized,macro_score,retrieval

def aggregate(rows,feature,lookup):
    groups=collections.defaultdict(set);labels={}
    for r in rows:
        if r['subject'] in labels and labels[r['subject']]!=r['label']:raise RuntimeError('conflicting_patient_diagnoses')
        labels[r['subject']]=r['label'];groups[r['subject']].add(lookup[r['image_id']])
    ids=sorted(groups)
    if not ids:raise RuntimeError('empty_patient_cohort')
    return ids,normalized(np.stack([feature[sorted(groups[i])].mean(0) for i in ids])),[labels[i] for i in ids]

def scalar(hits,labels):
    return {f'top{k}':macro_score(hits[k],[{v} for v in labels]) for k in [1,5,10]}|{f'micro_top{k}':float(hits[k].mean()) for k in [1,5,10]}

def episodes(x,q,y,qy,k,common):
    groups={d:np.flatnonzero(np.array(y)==d) for d in sorted(set(y))}
    vocab=[d for d,g in groups.items() if len(g)>=(10 if common else k)]
    qi=np.array([i for i,d in enumerate(qy) if d in vocab]);truth=[qy[i] for i in qi]
    if len(vocab)<2 or not len(qi):return None
    rng=np.random.default_rng(20261031+k);hits={n:[] for n in [1,5,10]}
    for _ in range(20):
        support=[rng.choice(groups[d],size=k,replace=False) for d in vocab]
        proto=normalized(np.stack([x[g].mean(0) for g in support]));rank=np.argsort(-(q[qi]@proto.T),axis=1,kind='stable')
        for n in hits:hits[n].append(np.array([truth[i] in {vocab[j] for j in order[:n]} for i,order in enumerate(rank)],bool))
    hits={n:np.stack(v) for n,v in hits.items()}
    return dict(patients=len(qi),diseases=len(set(truth)),candidate_diseases=len(vocab),repeats=20,
        **{f'top{n}':float(np.mean([macro_score(v,[{d} for d in truth]) for v in h])) for n,h in hits.items()}),hits,truth,qi

def score_case(name,rows,tables,lookup,support):
    path=WORK/'private'/f'{name}_features.npy';receipt=load(WORK/'inference'/f'{name}.json');assert digest(path)==receipt['sha256']
    feature=np.load(path,allow_pickle=False);report={};private={};folder=WORK/'private/hits'/name;folder.mkdir(parents=True,exist_ok=True)
    for task in ['frequent','rare']:
        result=[];arrays={k:collections.defaultdict(list) for k in [1,5,10]};patient_labels={}
        for fold in ([None] if task=='frequent' else range(10)):
            g=tables['gmdb_'+task+'_gallery_images'];q=tables['gmdb_'+task+'_test_images']
            if fold is not None:g=[r for r in g if int(r['split'])==fold];q=[r for r in q if int(r['split'])==fold]
            gids,gf,gy=aggregate(g,feature,lookup);qids,qf,qy=aggregate(q,feature,lookup)
            h,covered=retrieval(qf,gf,qids,gids,[{v} for v in qy],[{v} for v in gy]);result.append(scalar(h,qy))
            for i,subject in enumerate(qids):
                if subject in patient_labels:assert patient_labels[subject]==qy[i]
                patient_labels[subject]=qy[i]
                for k in h:arrays[k][subject].append(float(h[k][i]))
        identities=sorted(patient_labels);truth=[patient_labels[i] for i in identities]
        patient_hits={k:np.array([np.mean(v[i]) for i in identities]) for k,v in arrays.items()}
        report[task]=dict(patients=len(identities),diseases=len(set(truth)),folds=len(result),
            **scalar(patient_hits,truth),**{'fold_mean_'+k:float(np.mean([r[k] for r in result])) for k in result[0]},
            aggregation='Each query patient contributes once; average over its query appearances in supplied folds')
        np.savez(folder/(task+'.npz'),**{f'top{k}':v for k,v in patient_hits.items()})
        save(folder/(task+'_labels.json'),dict(patient_ids=identities,labels=truth))
    # Whole test cohort, deduplicated patients; support strictly from audited train.
    tests=tables['gmdb_frequent_test_images']+tables['gmdb_rare_test_images']
    gids,x,y=aggregate(support,feature,lookup);qids,q,qy=aggregate(tests,feature,lookup);assert not set(gids)&set(qids)
    for mode,common in [('fewshot_common',True),('fewshot_all_eligible',False)]:
        report[mode]={}
        for k in [1,5,10]:
            result=episodes(x,q,y,qy,k,common)
            if result is None:report[mode][str(k)]=dict(state='not_evaluable');continue
            summary,h,truth,qi=result;report[mode][str(k)]=summary
            np.savez(folder/f'{mode}_{k}.npz',**{f'top{n}':v for n,v in h.items()})
            save(folder/f'{mode}_{k}_labels.json',dict(patient_ids=[qids[i] for i in qi],labels=truth))
    save(WORK/'case_metrics'/f'{name}.json',report)
    return report

_BOOTSTRAP_CACHE={}
def interval(first,second,labels,repetitions=2000):
    # first/second [seed, (episode), patient]. Same patient/support pairing throughout.
    a=np.asarray(first);b=np.asarray(second);assert a.shape==b.shape
    if a.ndim==2:a=a[:,None,:];b=b[:,None,:]
    classes=sorted(set(labels));groups=[np.flatnonzero(np.array(labels)==c) for c in classes]
    def mean(z):return float(np.mean([z[...,g].mean() for g in groups]))
    key=(tuple(labels),a.shape[:2],repetitions)
    if key not in _BOOTSTRAP_CACHE:
        rng=np.random.default_rng(20261031);pw=np.zeros((repetitions,len(labels)),np.float64)
        for rep in range(repetitions):
            dc=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
            for d in np.flatnonzero(dc):
                g=groups[d];counts=rng.multinomial(len(g)*dc[d],np.full(len(g),1/len(g)))
                pw[rep,g]=counts/(len(groups)*len(g))
        sw=rng.multinomial(a.shape[0],np.full(a.shape[0],1/a.shape[0]),size=repetitions)/a.shape[0]
        ew=rng.multinomial(a.shape[1],np.full(a.shape[1],1/a.shape[1]),size=repetitions)/a.shape[1]
        _BOOTSTRAP_CACHE[key]=(sw,ew,pw)
    sw,ew,pw=_BOOTSTRAP_CACHE[key]
    samples=np.einsum('rs,re,sep,rp->r',sw,ew,b.astype(float)-a.astype(float),pw,optimize=True)
    return dict(delta=mean(b)-mean(a),hierarchical_paired_95_CI=np.quantile(samples,[.025,.975]).tolist(),bootstrap_repetitions=repetitions,
        resampling='paired downstream seeds, support episodes when present, diseases, and patients within disease',secondary_intervals_unadjusted=True)

def main():
    os.umask(0o077);selection=load(WORK/'selection.json');p=load(WORK/'protocol.json');pins=load(WORK/'data_pins.json');assert digest(SHARED/'inputs.private.jsonl')==pins['public_inputs_sha256']
    rows=[json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()];lookup={r['image_id']:i for i,r in enumerate(rows)}
    blocked=set(load(WORK/'private/blocked_patients.json')['subjects']);tables={}
    for n,sha in pins['splits_sha256'].items():
        f=SHARED/'split_snapshot'/(n+'.csv');assert digest(f)==sha
        with f.open() as s:tables[n]=[r for r in csv.DictReader(s) if r['subject'] not in blocked]
    support_ids=set(load(WORK/'private/support_ids.json')['image_ids']);support=[r for r in rows if r['image_id'] in support_ids]
    names=sorted({n for setting in selection['settings'].values() for cell in setting.values() for n in cell}|set(selection.get('frozen_initializations',[])))
    case_metrics={n:score_case(n,rows,tables,lookup,support) for n in names};summary={};paired={}
    for setting,cells in selection['settings'].items():
        summary[setting]={};paired[setting]={}
        for cell,names in cells.items():
            vals=[case_metrics[n] for n in names];r={}
            for task in ['frequent','rare']:
                r[task]={f'{m}_{s}':float(fn([v[task][m] for v in vals])) for m in ['top1','top5','top10'] for s,fn in [('mean',np.mean),('SD',lambda x:np.std(x,ddof=1))]}
            r['fewshot_common']={str(k):{f'top{n}_mean':float(np.mean([v['fewshot_common'][str(k)]['top'+str(n)] for v in vals])) for n in [1,5,10]} for k in [1,5,10]}
            summary[setting][cell]=r
        for regime in p['regimes']:
            for base in ['transface_original','r50']:
                first=cells[base+'__'+regime];second=cells['transface_chop_cpt__'+regime];key='transface_chop_cpt_minus_'+base+'__'+regime;out={}
                tasks=['frequent','rare']+[f'fewshot_common_{k}' for k in [1,5,10]]
                for task in tasks:
                    lab=load(WORK/'private/hits'/first[0]/(task+'_labels.json'))
                    for n in first+second:assert load(WORK/'private/hits'/n/(task+'_labels.json'))==lab
                    aa=np.stack([np.load(WORK/'private/hits'/n/(task+'.npz'))['top1'] for n in first]);bb=np.stack([np.load(WORK/'private/hits'/n/(task+'.npz'))['top1'] for n in second])
                    out[task]=interval(aa,bb,lab['labels'])
                paired[setting][key]=out
    save(WORK/'results.json',dict(state='complete',protocol=p,selection=selection,audit=load(WORK/'audit.json'),summary=summary,paired_top1=paired,
        seeds=p['seeds'],frozen_initializations={n:case_metrics[n] for n in selection.get('frozen_initializations',[])},no_best_seed_selection=True,exploratory_repeated_test=True,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:save(WORK/'score_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

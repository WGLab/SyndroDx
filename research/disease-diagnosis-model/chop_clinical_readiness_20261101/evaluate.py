"""Strict missing-evidence routing and matched CHOP controls; no model updates."""
import collections
import sys
from pathlib import Path
import numpy as np
import torch

WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
sys.path.insert(0,str(PROJECT/'clinical_evidence_agent_20261101'))
from common import load,save,sha,now
from frozen_v1 import FrozenV1


def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def mapped_scores(scores,mapping,candidate_count):
    output=np.zeros((len(scores),candidate_count),np.float32)
    for column,target in enumerate(mapping):output[:,target]+=scores[:,column]
    return output


def classify(scores,truth,available):
    rank=np.argsort(-scores,axis=1,kind='stable')
    return {k:(rank[:,:k]==truth[:,None]).any(1)&available&(truth>=0) for k in [1,5,10]}


def summarize(hits,diseases,mask,available):
    selected=np.flatnonzero(mask);accepted=selected[available[selected]]
    result={'patients':len(selected),'diseases':len(set(diseases[selected])),
            'ranked_patients':len(accepted),'abstained_patients':len(selected)-len(accepted)}
    for name,subset in [('all',selected),('ranked',accepted)]:
        result[name]={f'top{k}_{kind}':float(value) if value is not None else None
            for k,h in hits.items() for kind,value in [
                ('micro',h[subset].mean() if len(subset) else None),
                ('macro',np.mean([h[subset][diseases[subset]==c].mean() for c in set(diseases[subset])]) if len(subset) else None)]}
    return result


def retrieve_scores(scores,labels,queries,available):
    scores=scores.copy();scores[np.arange(len(queries)),queries]=-np.inf
    rank=np.argsort(-scores,axis=1,kind='stable');labels=np.asarray(labels)
    return {k:((labels[rank[:,:k]]==labels[queries,None]) & (rank[:,:k]!=queries[:,None])).any(1)&available for k in [1,5,10]}


def main():
    torch.set_num_threads(2)
    audit=load(WORK/'audit.json');protocol=load(WORK/'protocol.json')
    p=WORK/'private/strict_arrays.npz';ip=WORK/'private/strict_index.json'
    if sha(p)!=audit['arrays_sha256'] or sha(ip)!=audit['index_sha256']:raise ValueError('audited_inputs_changed')
    with np.load(p) as arrays:data={k:arrays[k].copy() for k in arrays.files}
    index=load(ip);truth=data['labels'];diseases=np.asarray(index['labels']);n=len(truth)
    adapter=FrozenV1('cuda' if torch.cuda.is_available() else 'cpu')
    if sha(adapter.manifest_path)!=audit['model_manifest_sha256']:raise ValueError('model_changed')
    available=data['eligible_HPO'];face=np.linalg.norm(data['features'],axis=1)>0
    probabilities={};routes={}
    for arm,mode in [('visual','face_only'),('HPO','hpo_only'),('cross_attention_routed','face_hpo')]:
        scores=[];route=[]
        for start in range(0,n,32):
            s,r=adapter.probabilities(data['features'][start:start+32],data['ids'][start:start+32],data['states'][start:start+32],mode)
            scores.append(s);route.extend(r)
        probabilities[arm]=np.concatenate(scores);routes[arm]=route
    mixed=probabilities['visual'].copy()
    mixed[available]=.5*probabilities['visual'][available]+.5*probabilities['HPO'][available]
    probabilities['score_fusion_routed']=mixed
    routes['score_fusion_routed']=['face_hpo' if x else 'face_only' for x in available]
    masks={'all_expert':np.ones(n,bool),'covered':data['covered'],
           'recorded_HPO_and_covered':data['recorded_HPO']&data['covered'],
           'eligible_HPO_and_covered':available&data['covered']}
    metrics={};stored={}
    for arm,prob in probabilities.items():
        mapped=mapped_scores(prob,data['class_to_candidate'],len(index['candidates']))
        avail=available if arm=='HPO' else face
        hits=classify(mapped,truth,avail)
        metrics[arm]={name:summarize(hits,diseases,mask,avail) for name,mask in masks.items()}
        stored[arm+'_scores']=mapped
        for k,h in hits.items():stored[arm+'_hit'+str(k)]=h
    q=data['queries'];native=norm(data['features']);raw=native[q]@native.T
    rh=retrieve_scores(raw,diseases,q,np.ones(len(q),bool))
    retrieval={'visual':summarize(rh,diseases[q],np.ones(len(q),bool),np.ones(len(q),bool))}
    if not available.any():
        if not np.array_equal(probabilities['visual'],probabilities['cross_attention_routed']):raise ValueError('cross_attention_missing_HPO_fallback_changed')
        if not np.array_equal(probabilities['visual'],probabilities['score_fusion_routed']):raise ValueError('score_fusion_missing_HPO_fallback_changed')
        retrieval['score_fusion_routed']=retrieval['visual']
        retrieval['cross_attention_routed']=retrieval['visual']
        retrieval['HPO']={'state':'not_estimable_no_eligible_HPO','ranked_queries':0,'abstained_queries':len(q)}
    else:raise ValueError('known_polarity_found_requires_additional_time_and_source_admission')
    # Sensitivity controls explain the old computation; NOT eligible deployment.
    old=PROJECT/'chop_hpo_increment_20261101/expert'
    with np.load(old/'private/arrays.npz') as f:
        legacy_ids=f['ids'].copy();legacy_states=f['states'].copy()
    sensitivity={}
    with torch.inference_mode():
        for name,ids,states in [('legacy_unresolved_state2',legacy_ids,legacy_states),
                               ('reference_head_without_HPO',np.zeros_like(legacy_ids),np.zeros_like(legacy_states))]:
            pp=[];zz=[]
            for start in range(0,n,32):
                x=torch.as_tensor(data['features'][start:start+32],device=adapter.device)
                i=torch.as_tensor(ids[start:start+32],device=adapter.device)
                s=torch.as_tensor(states[start:start+32],device=adapter.device)
                logits,z=adapter.models['reference_11'](x,i,s)
                pp.append(logits.float().softmax(-1).cpu().numpy());zz.append(z.float().cpu().numpy())
            prob=mapped_scores(np.concatenate(pp),data['class_to_candidate'],len(index['candidates']))
            hits=classify(prob,truth,face);vectors=norm(np.concatenate(zz))
            rr=retrieve_scores(vectors[q]@vectors.T,diseases,q,np.ones(len(q),bool))
            sensitivity[name]={'classification':{key:summarize(hits,diseases,mask,face) for key,mask in masks.items()},
                'retrieval':summarize(rr,diseases[q],np.ones(len(q),bool),np.ones(len(q),bool)),
                'not_eligible_multimodal_efficacy':True}
    np.savez_compressed(WORK/'private/strict_outputs.npz',**stored)
    legacy_index=load(old/'private/index.json')
    with np.load(old/'private/arrays.npz') as f:
        mapping_changed=legacy_index['candidates']!=index['candidates'] or not np.array_equal(f['class_to_candidate'],data['class_to_candidate'])
    report={'state':'complete','purpose':'CHOP input validity audit and strict routing controls',
        'patients':n,'model_seed':11,'precision':'FP32','candidate_classes':len(index['candidates']),
        'classification':metrics,'retrieval':retrieval,'routing':{k:dict(collections.Counter(v)) for k,v in routes.items()},
        'sensitivity_not_clinical_efficacy':sensitivity,'multimodal_efficacy':'not_estimable_no_verified_polarity_HPO',
        'HPO_only_zero_input_is_abstention_not_random_prediction':True,'strict_fusion_fallbacks_equal_visual':True,
        'diagnosis_candidate_mapping_changed':mapping_changed,'patient_groups_used_for_training':0,'model_parameters_updated':False,
        'historical_results_overwritten':False,'protected_cohort_exploratory':True,'BCH_accessed':False,
        'protocol_sha256':sha(WORK/'protocol.json'),'audit_sha256':sha(WORK/'audit.json'),
        'patient_records_in_output':False,'updated_at_utc':now()}
    save(WORK/'results.json',report)
    print(json.dumps({'state':'complete','patients':n,'multimodal_efficacy':report['multimodal_efficacy']}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        save(WORK/'evaluation_failure.json',{'state':'failed','error_type':type(e).__name__,
            'locations':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)],'updated_at_utc':now()})
        raise SystemExit(1) from None

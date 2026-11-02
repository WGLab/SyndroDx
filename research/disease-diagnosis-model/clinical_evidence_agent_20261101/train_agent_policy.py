"""Fit/select an evidence policy on training patients, then run a locked replay.

This updates only question selection. The diagnostic model remains frozen.
"""
import collections
import fcntl
import importlib.util
import traceback
import numpy as np
import torch
from common import *
from acquisition_replay import observations, initial_observations, answer, infer, hit_matrix, metrics, paired, select_question
from frozen_v1 import FrozenV1
from learned_questions import CONFIGS, LearnedQuestions, fit_tables, split_policy_patients

WORK=ROOT/'acquisition_policy_v2'


def progress(state, **kwargs):
    save(WORK/'status.json',dict(state=state,**kwargs,updated_at_utc=now()))


def rollout(adapter, features, recorded, initial, labels, mode, policy, inverse,
            budget, old_context=None):
    known=[dict(r) for r in initial];asked=[set() for _ in recorded]
    probability=infer(adapter,features,known,inverse,mode)
    hits=[hit_matrix(probability,labels)];counts=[];actions=[]
    for step in range(budget):
        excluded=[asked[row]|set(known[row]) for row in range(len(known))]
        if old_context is None:
            questions,_=policy.choose(probability,excluded)
        else:
            questions=[select_question('adaptive',probability[row],adapter.labels,excluded[row],[],[],
                        old_context['associations'],old_context['sha'],adapter.terms,inverse,None)
                       for row in range(len(known))]
        step_counts=collections.Counter();step_actions=[]
        for row,term in enumerate(questions):
            before=dict(known[row])
            outcome=answer(recorded[row],known[row],asked[row],term)
            if outcome=='unknown' and known[row]!=before: raise ValueError('unknown_changed_patient')
            step_counts[outcome]+=1
            step_actions.append([term or 0,{'no_question':-1,'present':0,'absent':1,'unknown':2}[outcome]])
        probability=infer(adapter,features,known,inverse,mode)
        hits.append(hit_matrix(probability,labels));counts.append(dict(step_counts));actions.append(step_actions)
    return np.stack(hits),counts,np.asarray(actions),probability


def mean_steps(hits,counts,labels,budget):
    result=[]
    for step in range(budget+1):
        row=metrics(np.stack([h[step] for h in hits]),labels)
        if step:
            total=collections.Counter()
            for run in counts:
                for value in run[:step]:total.update(value)
            questions=sum(v for k,v in total.items() if k!='no_question')
            row.update(questions=questions,recorded_answers=total['present']+total['absent'],
                       unknown_answers=total['unknown'],recorded_answer_rate=(total['present']+total['absent'])/max(questions,1))
        result.append(row)
    return result


def main():
    WORK.mkdir(exist_ok=True,mode=0o700)
    lock=(WORK/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (WORK/'results.json').exists():return
    protocol=load(WORK/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source_changed_after_protocol_lock')
    tests=load(WORK/'tests.json')
    if tests['state']!='passed':raise ValueError('tests_not_passed')
    torch.set_num_threads(2)
    source=PROJECT/'protected_visual_fusion_20261101/private/gmdb.npz'
    receipt=load(PROJECT/'protected_visual_fusion_20261101/input_receipt.json')
    if sha(source)!=receipt['gmdb_sha256'] or receipt['patient_overlap']!=0:raise ValueError('unverified_arrays')
    # No development rows (features, labels or HPO) are accessed until selection is locked.
    with np.load(source) as data:
        train=data['train'].copy()
        if len(set(train))!=len(train):raise ValueError('duplicate_training_patient')
        tf=data['features'][train].copy();ty=data['labels'][train].copy()
        tr=[observations(i,s) for i,s in zip(data['ids'][train],data['states'][train])]
    fit,selection=split_policy_patients(ty,protocol['split_seed'])
    if set(fit)&set(selection) or set(fit)|set(selection)!=set(range(len(train))):raise ValueError('policy_split_overlap')
    (WORK/'private').mkdir(exist_ok=True,mode=0o700)
    np.savez_compressed(WORK/'private/policy_patient_split.npz',fit=train[fit],selection=train[selection])
    progress('loading_model',fit_patients=len(fit),selection_patients=len(selection))
    adapter=FrozenV1('cuda' if torch.cuda.is_available() else 'cpu')
    inverse={i:t for t,i in adapter.terms.items()}
    profile_path=ROOT/'assets/class_hpo_associations.json';profile_receipt=load(ROOT/'hpo_question_receipt.json')
    if sha(profile_path)!=profile_receipt['output_sha256']:raise ValueError('public_profile_changed')
    associations=load(profile_path)['associations']
    pool=sorted({adapter.terms[t] for ts in associations.values() for t in ts})
    tables=fit_tables([tr[i] for i in fit],ty[fit],pool,len(adapter.labels))
    records=[tr[i] for i in selection];features=tf[selection];labels=ty[selection]
    statistics={};completed=0
    for config in CONFIGS:
        policy=LearnedQuestions(tables,config,adapter.labels,inverse)
        hits=[];counts=[]
        for seed in protocol['selection_mask_seeds']:
            initial=[initial_observations(r,3,seed,row) for row,r in enumerate(records)]
            h,c,a,p=rollout(adapter,features,records,initial,labels,'face_hpo',policy,inverse,protocol['action_budget'])
            hits.append(h);counts.append(c)
            np.savez_compressed(WORK/'private'/f'selection__{config["name"]}__{seed}.npz',hits=h,actions=a)
            completed+=1
            progress('internal_policy_selection',completed_rollouts=completed,
                     total_rollouts=len(CONFIGS)*len(protocol['selection_mask_seeds']),fit_patients=len(fit),selection_patients=len(selection))
        statistics[config['name']]=mean_steps(hits,counts,labels,protocol['action_budget'])
    # Fixed is included in model selection, ties prefer the simpler policy.
    chosen=max(CONFIGS,key=lambda c:(statistics[c['name']][-1]['top1_macro'],statistics[c['name']][-1]['top5_macro'],c['name']=='fixed'))
    selection_record={'state':'locked_before_development_evaluation','selected_config':chosen,
        'fit_patients':len(fit),'selection_patients':len(selection),'selection_diseases':len(np.unique(labels)),
        'results':statistics,'selection_criterion':'macro Top1, then macro Top5 at k3/budget5; tie prefers fixed',
        'base_ranker_previously_trained_on_full_training_pool':True,
        'policy_fit_selection_disjoint':True,'development_used_for_policy_selection':False,
        'patient_split_sha256':sha(WORK/'private/policy_patient_split.npz'),'updated_at_utc':now()}
    save(WORK/'selection.json',selection_record)
    selection_hash=sha(WORK/'selection.json')
    # Refit sufficient statistics on all training patients after choosing config.
    all_tables=fit_tables(tr,ty,pool,len(adapter.labels))
    candidate=WORK/'candidate';candidate.mkdir(exist_ok=True,mode=0o700)
    np.savez_compressed(candidate/'question_model.npz',**all_tables)
    save(candidate/'vocabulary.json',{'labels':adapter.labels,'inverse':inverse})
    save(candidate/'config.json',chosen)
    save(candidate/'lock.json',{'selection_sha256':selection_hash,'model_sha256':sha(candidate/'question_model.npz'),
        'config_sha256':sha(candidate/'config.json'),'locked_before_development_rows_accessed':True,'updated_at_utc':now()})
    # Development is an already-inspected research cohort, not a new clinical test.
    with np.load(source) as data:
        val=data['validation'].copy()
        if set(train)&set(val) or len(set(val))!=len(val) or len(val)!=320:raise ValueError('development_split_overlap')
        features=data['features'][val].copy();labels=data['labels'][val].copy()
        records=[observations(i,s) for i,s in zip(data['ids'][val],data['states'][val])]
    np.savez_compressed(WORK/'private/development_indices.npz',validation=val)
    policies={'selected':LearnedQuestions(all_tables,chosen,adapter.labels,inverse),
              'fixed':LearnedQuestions(all_tables,CONFIGS[0],adapter.labels,inverse),'previous':None}
    curves={};answer_counts={};progress('development_replay',selected_config=chosen['name'])
    completed=0;total=len(protocol['initial_positive_hpo'])*len(protocol['evaluation_mask_seeds'])*2*3
    for k in protocol['initial_positive_hpo']:
        for seed in protocol['evaluation_mask_seeds']:
            initial=[initial_observations(r,k,seed,row) for row,r in enumerate(records)]
            for mode in ['face_hpo','hpo_only']:
                for name,policy in policies.items():
                    key=f'k{k}__seed{seed}__{mode}__{name}'
                    context={'associations':associations,'sha':profile_receipt['output_sha256']} if name=='previous' else None
                    h,c,a,p=rollout(adapter,features,records,initial,labels,mode,policy,inverse,protocol['action_budget'],context)
                    curves[key],answer_counts[key]=h,c
                    np.savez_compressed(WORK/'private'/(key+'.npz'),hits=h,actions=a,final_probabilities=p)
                    completed+=1;progress('development_replay',selected_config=chosen['name'],completed_rollouts=completed,total_rollouts=total)
    results={};comparisons={}
    def gather(k,mode,policy,step):
        return np.stack([curves[f'k{k}__seed{s}__{mode}__{policy}'][step] for s in protocol['evaluation_mask_seeds']])
    for k in protocol['initial_positive_hpo']:
        results[str(k)]={};comparisons[str(k)]={}
        for mode in ['face_hpo','hpo_only']:
            results[str(k)][mode]={};comparisons[str(k)][mode]={}
            for name in policies:
                hs=[curves[f'k{k}__seed{s}__{mode}__{name}'] for s in protocol['evaluation_mask_seeds']]
                cs=[answer_counts[f'k{k}__seed{s}__{mode}__{name}'] for s in protocol['evaluation_mask_seeds']]
                results[str(k)][mode][name]=mean_steps(hs,cs,labels,protocol['action_budget'])
            for step in [1,3,5]:
                active=gather(k,mode,'selected',step)
                comparisons[str(k)][mode][str(step)]={
                    'selected_minus_fixed':paired(active,gather(k,mode,'fixed',step),labels),
                    'selected_minus_previous':paired(active,gather(k,mode,'previous',step),labels),
                    'selected_minus_initial':paired(active,gather(k,mode,'selected',0),labels)}
        comparisons[str(k)]['facial_increment']={str(step):paired(gather(k,'face_hpo','selected',step),gather(k,'hpo_only','selected',step),labels) for step in [0,1,3,5]}
    primary=comparisons['3']['face_hpo']['5']['selected_minus_fixed']
    benefit=chosen['name']!='fixed' and primary['top1']['paired_disease_bootstrap_ci95'][0]>0 and primary['top5']['delta_macro']>=0
    # Predeclared release gate: keep fixed unless the selected policy meets this
    # development benefit gate; no alternative candidate is tried on development.
    release_config=chosen if benefit else CONFIGS[0]
    decision='release_selected_research_policy' if benefit else 'retain_fixed_research_policy'
    output={'state':'complete','scope':'retrospective_development_annotation_masking',
        'patients':len(val),'diseases':len(np.unique(labels)),'selection':selection_record,
        'selected_config':chosen,'release_config':release_config,'decision':decision,
        'development_rollouts':completed,'results':results,'paired_comparisons':comparisons,
        'benefit_gate_passed':benefit,'clinical_efficacy_established':False,'LLM_controller_used':False,
        'variant_diagnostic_scoring':False,'backbone_updated':False,'protected_test_accessed':False,
        'policy_selection_used_external_development':False,'selection_sha256':selection_hash,
        'source_sha256':sha(source),'protocol_sha256':sha(WORK/'protocol.json'),
        'interval_scope':'paired disease bootstrap after averaging masks per patient; exploratory reused development, no multiplicity adjustment',
        'updated_at_utc':now()}
    if sha(WORK/'selection.json')!=selection_hash:raise ValueError('selection_changed')
    save(WORK/'results.json',output)
    # A research runtime is released with an explicit evidence scope, no clinical claim.
    release=ROOT/'agent_releases/available_evidence_v2';release.mkdir(parents=True,exist_ok=True,mode=0o700)
    import shutil
    for src,name in [(candidate/'question_model.npz','question_model.npz'),(candidate/'vocabulary.json','vocabulary.json'),(ROOT/'learned_questions.py','learned_questions.py')]:
        shutil.copy2(src,release/name)
    save(release/'manifest.json',{'state':'locked_research_policy','config':release_config,
        'artifacts':{key:{'path':name,'sha256':sha(release/name)} for key,name in [('tables','question_model.npz'),('vocabulary','vocabulary.json'),('policy_code','learned_questions.py')]},
        'base_model_manifest_sha256':sha(adapter.manifest_path),'selection_sha256':selection_hash,
        'evaluation_sha256':sha(WORK/'results.json'),'decision':decision,'variant_scoring':False,
        'clinical_efficacy_established':False,'LLM_controller_used':False,'updated_at_utc':now()})
    save(ROOT/'agent_releases/CURRENT.json',{'manifest':str(release/'manifest.json'),'sha256':sha(release/'manifest.json'),'updated_at_utc':now()})
    progress('complete',completed_rollouts=completed,total_rollouts=total,decision=decision,
             release_config=release_config['name'],selected_config=chosen['name'])
    try:
        spec=importlib.util.spec_from_file_location('local_notifier',PROJECT/'transface_objective_comparison_20261028/matched_runtime.py')
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        runtime.notify('agent policy comparison completed',json.dumps({'decision':decision,'patients':len(val),
            'primary_k3':results['3']['face_hpo'],'results_path':str(WORK/'results.json')},indent=2))
        save(WORK/'notification.json',{'state':'local_notifier_invoked','updated_at_utc':now()})
    except Exception as e:save(WORK/'notification.json',{'state':'notifier_failed','error_type':type(e).__name__,'updated_at_utc':now()})
    print(json.dumps({'state':'complete','decision':decision,'patients':len(val)}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        progress('failed',error_type=type(e).__name__,locations=[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)])
        raise SystemExit(1) from None

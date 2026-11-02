"""Audit completed trajectories and publish an aggregate research-stage report."""
import numpy as np
from common import *
from acquisition_replay import observations, initial_observations, answer, hit_matrix
from learned_questions import current_selector

WORK=ROOT/'acquisition_policy_v2'


def main():
    result=load(WORK/'results.json');protocol=load(WORK/'protocol.json')
    if result['state']!='complete':raise ValueError('experiment_not_complete')
    source=PROJECT/'protected_visual_fusion_20261101/private/gmdb.npz'
    if sha(source)!=result['source_sha256']:raise ValueError('source_changed')
    selection=load(WORK/'selection.json');lock=load(WORK/'candidate/lock.json')
    if sha(WORK/'selection.json')!=lock['selection_sha256'] or lock['selection_sha256']!=result['selection_sha256']:
        raise ValueError('selection_changed_after_lock')
    with np.load(source) as data:
        train,val=data['train'],data['validation'];truth=data['labels'][val]
        records=[observations(i,s) for i,s in zip(data['ids'][val],data['states'][val])]
    with np.load(WORK/'private/policy_patient_split.npz') as split:
        fit,select=set(split['fit'].tolist()),set(split['selection'].tolist())
        if fit&select or fit|select!=set(train) or (fit|select)&set(val):raise ValueError('patient_split_violation')
    rollouts=0;unknowns=0;recorded_answers=0
    for k in protocol['initial_positive_hpo']:
        for seed in protocol['evaluation_mask_seeds']:
            initial=[initial_observations(r,k,seed,row) for row,r in enumerate(records)]
            for mode in protocol['evaluation_modes']:
                baseline=None
                for policy in protocol['evaluation_policies']:
                    key=f'k{k}__seed{seed}__{mode}__{policy}'
                    with np.load(WORK/'private'/(key+'.npz')) as data:
                        hits,actions,prob=data['hits'],data['actions'],data['final_probabilities']
                    if baseline is None:baseline=hits[0]
                    elif not np.array_equal(baseline,hits[0]):raise ValueError('initial_context_mismatch')
                    if not np.array_equal(hit_matrix(prob,truth),hits[-1]):raise ValueError('final_metric_mismatch')
                    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1,atol=1e-5):raise ValueError('invalid_ranker_probabilities')
                    for row in range(len(records)):
                        known=dict(initial[row]);asked=set()
                        for term,state in actions[:,row]:
                            before=dict(known);outcome=answer(records[row],known,asked,int(term) or None)
                            if int(state)!={'no_question':-1,'present':0,'absent':1,'unknown':2}[outcome]:raise ValueError('unrecorded_answer')
                            if outcome=='unknown':
                                if before!=known:raise ValueError('unknown_used_as_finding')
                                unknowns+=1
                            if outcome in ['present','absent']:recorded_answers+=1
                    # The previous-policy control must reproduce the prior locked experiment.
                    if policy in {'previous','fixed'}:
                        old_name='adaptive' if policy=='previous' else 'fixed'
                        old=ROOT/'acquisition_replay_v1/private'/f'k{k}__seed{seed}__{mode}__{old_name}.npz'
                        with np.load(old) as previous:
                            if not np.array_equal(previous['hits'],hits) or not np.array_equal(previous['actions'],actions):
                                raise ValueError('control_policy_replay_drift')
                    rollouts+=1
    policy=current_selector()
    # Production callback and batch planner must select exactly the same question.
    rng=np.random.default_rng(20261102)
    probabilities=rng.dirichlet(np.ones(len(policy.labels)),size=4)
    excluded=[set(map(int,policy.pool[:i])) for i in range(4)]
    terms,_=policy.choose(probabilities,excluded)
    for row,term in enumerate(terms):
        action=policy({d:float(np.log(p)) for d,p in zip(policy.labels,probabilities[row])},
                      {policy.inverse[t] for t in excluded[row]})
        if action['concept_id']!=policy.inverse[term]:raise ValueError('runtime_policy_callback_mismatch')
    receipt={'state':'passed','verified_development_rollouts':rollouts,'patient_split_overlap':0,
        'policy_fit_patients':len(fit),'policy_selection_patients':len(select),'development_patients':len(val),
        'previous_and_fixed_policies_reproduced_exactly':True,'shared_initial_predictions':True,'repeated_questions':0,
        'unknown_actions_verified':unknowns,'recorded_answers_verified':recorded_answers,
        'unrecorded_negative_answers':0,'runtime_policy_matches_batch_planner':True,
        'results_sha256':sha(WORK/'results.json'),'clinical_efficacy_established':False,'updated_at_utc':now()}
    save(WORK/'verification.json',receipt)
    root_status=load(ROOT/'status.json')
    root_status.update(state='research_agent_v2_complete_and_verified',
        active_scope='Frozen facial/HPO diagnosis, released question policy, and source-pinned variant review',
        acquisition_replay='v2_training_only_policy_selection_and_development_evaluation_complete',
        acquisition_results=str(WORK/'results.json'),acquisition_decision=result['decision'],
        agent_policy_manifest=load(ROOT/'agent_releases/CURRENT.json')['manifest'],
        diagnostic_metrics_available='Development masked-HPO comparison; clinical agent efficacy not established',
        LLM_agent_deployed=False,variant_scoring_enabled=False,updated_at_utc=now())
    root_status['pending']=['Independent clinical evaluation with reviewed HPO and real decision times',
        'Variant detection, genotype and condition-specific interpretation verification',
        'Optional LLM controller only if an additional benefit is demonstrated']
    save(ROOT/'status.json',root_status)
    registry=load(PROJECT/'LATEST_MULTIMODAL_EXPERIMENT.json')
    registry.update(latest_agent_evaluation={'state':'complete_verified_development_simulation',
        'results':str(WORK/'results.json'),'verification':str(WORK/'verification.json'),
        'decision':result['decision'],'release_manifest':root_status['agent_policy_manifest']},updated_at_utc=now())
    save(PROJECT/'LATEST_MULTIMODAL_EXPERIMENT.json',registry)
    values=result['results']['3']['face_hpo']
    rows=[('No questions',values['fixed'][0]),('Previous heuristic',values['previous'][5]),
          ('Fixed order',values['fixed'][5]),('Training-selected policy',values['selected'][5])]
    lines=['# Evidence agent v2: completed research-stage results','',
        'The frozen diagnostic model was unchanged. Question policies were fitted and selected using only the existing GMDB training pool.',
        '',f'Policy fit: {len(fit)} patients. Policy selection: {len(select)} patients. Development replay: {len(val)} patients.',
        f'Selected configuration: `{result["selected_config"]["name"]}`. Released configuration: `{result["release_config"]["name"]}`.',
        f'Decision: `{result["decision"]}`.','',
        'Primary setting: up to three recorded positive HPO terms initially, explicit negative findings retained, five questions. Disease-macro means over three paired masking seeds.',
        '', '| Method | Top1 | Top5 | Recorded answer rate |','|---|---:|---:|---:|']
    for name,v in rows:
        rate=f'{100*v["recorded_answer_rate"]:.2f}%' if 'recorded_answer_rate' in v else '—'
        lines.append(f'| {name} | {100*v["top1_macro"]:.2f}% | {100*v["top5_macro"]:.2f}% | {rate} |')
    delta=result['paired_comparisons']['3']['face_hpo']['5']['selected_minus_fixed']
    for k in [1,5]:
        x=delta[f'top{k}'];lo,hi=x['paired_disease_bootstrap_ci95']
        lines += ['',f'Selected minus fixed Top{k}: {100*x["delta_macro"]:+.2f} percentage points; exploratory paired-disease bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}].']
    lines += ['', 'The release rule was specified before running: adopt the training-selected policy only if its primary Top1 lower confidence bound exceeds zero and its Top5 point difference is nonnegative; otherwise retain fixed ordering. No other candidate was tried on development.',
        '', 'The probability model describes recorded answers, not clinical penetrance or calibrated disease risk. Unknown answers consume a question and never become negative findings. The policy receives current model scores and already observed/asked terms, never hidden answers or diagnostic truth.',
        '', 'The diagnostic model had already been trained on the full GMDB training pool; the internal split isolates policy fitting only. Development has been inspected previously. These are retrospective annotation-masking results, not prospective evidence or a clinical time-to-diagnosis result.',
        '',f'Verification: {rollouts} development trajectories, zero repeated questions, zero unrecorded negative answers, exact reproduction of the previous policy, and runtime callback parity.',
        '', '## Local runtime', '',
        'Use `clinical_evidence_agent_20261101/run_agent.py` with local JSONL input and output paths. Inputs contain a typed case, optional provenance-checked native512 facial feature and actual reviewed answer records. Patient data and detailed traces must stay on HPC. The CLI prints aggregate counts only.',
        '', 'Output includes the differential, evidence-source checks, requested observations and outcomes, stop reason, policy/model hashes, and optional variant review cards. Variant cards are separate from diagnostic scoring. No LLM or external API is used.',
        '', 'The variant branch contains 600 source-pinned records for 119 existing development patients; 326 have linked public knowledge and 115 have one exact nucleotide ClinVar candidate. It does not assert patient detection or molecular diagnosis.',
        '', 'A specific SharePoint folder is still needed before checkpoint sharing. No upload was made.', '']
    (WORK/'README.md').write_text('\n'.join(lines))
    print(json.dumps(receipt))


if __name__=='__main__':guarded_main(main)

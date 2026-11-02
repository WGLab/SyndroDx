"""Corrected all-candidate ranking on the same frozen checkpoints and patients."""
import collections,fcntl,os,subprocess,time
import numpy as np
from runtime import *
from controller import jobs,launch
def score(arms,world,destination='diagnosis_results_exhaustive.json',send_notification=True):
    patients=load(WORK/'private/sft_patients.json');truth={i:r['label'] for i,r in enumerate(patients['patients'])}
    classes={r['label'] for r in patients['vocabulary']};all_rows={};results={};summed={};likelihood_checks=[]
    for arm in arms:
        rows=[]
        for i in range(world):
            p=WORK/'ranking_exhaustive'/arm/'private'/f'{i}.json';receipt=load(WORK/'ranking_exhaustive'/arm/f'{i}.receipt.json')
            if digest(p)!=receipt['sha256']:raise RuntimeError('ranking_receipt_mismatch')
            likelihood_checks.append(receipt['direct_likelihood_check_error']);rows+=load(p)['records']
        rows=sorted(rows,key=lambda r:r['index']);all_rows[arm]=rows
        if len(rows)!=load(WORK/'sft_data_receipt.json')['validation_patients'] or len({r['index'] for r in rows})!=len(rows):raise RuntimeError('ranking_patient_coverage_error')
        if any(r['truth']!=truth[r['index']] or len(r['ranked'])!=len(classes) or set(r['ranked'])!=classes or set(r['ranked_sum'])!=classes for r in rows):raise RuntimeError('incomplete_or_mislabeled_disease_ranking')
        grouped=collections.defaultdict(list)
        for row in rows:grouped[row['truth']].append(row)
        for field,target in [('ranked',results),('ranked_sum',summed)]:
            target[arm]={f'macro_top{k}':float(np.mean([np.mean([r['truth'] in r[field][:k] for r in rr]) for rr in grouped.values()])) for k in [1,5,10]}
            target[arm].update({f'micro_top{k}':float(np.mean([r['truth'] in r[field][:k] for r in rows])) for k in [1,5,10]})
            target[arm].update(patients=len(rows),diseases=len(grouped))
    a=all_rows['fusion'];b=all_rows['hpo']
    if [r['index'] for r in a]!=[r['index'] for r in b]:raise RuntimeError('unmatched_control_patients')
    deltas={};rng=np.random.default_rng(20261027)
    for k in [1,5,10]:
        grouped=collections.defaultdict(list)
        for x,y in zip(a,b):grouped[x['truth']].append(int(x['truth'] in x['ranked'][:k])-int(y['truth'] in y['ranked'][:k]))
        boot=np.zeros(2000)
        for v in grouped.values():boot+=rng.choice(v,(2000,len(v)),replace=True).mean(1)/len(grouped)
        deltas[f'top{k}']=dict(difference=float(np.mean([np.mean(v) for v in grouped.values()])),ci95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(state='complete',results=results,fusion_minus_HPO_only=deltas,supplementary_sum_log_probability=summed,
        scope='exploratory patient-disjoint GMDB development; same selected checkpoints; not external test',
        candidate_diseases=len(classes),all_candidates_scored_for_every_patient=True,
        method='full teacher forcing, FP32 output head; mean conditional token log probability of canonical disease name including EOS; all 216 candidates',
        scoring_policy_selected_before_rescoring=True,maximum_batch_vs_single_NLL_error=max(likelihood_checks),
        compute_precision='BF16 backbone, FP32 output head; fixed candidate batches of up to eight',
        uncertainty='patient bootstrap within fixed validation diseases; does not include training-seed or new-disease uncertainty',
        prior_beam_results='superseded: incomplete unique candidate lists; unchanged checkpoints rescored',
        updated_at_utc=now())
    save(WORK/destination,result)
    if send_notification:notify('corrected_diagnosis_ranking_complete','Corrected exhaustive diagnosis ranking completed on unchanged checkpoints and the same 321 GMDB development patients. Aggregate results: '+str(WORK/destination)+'. Previous beam lists were incomplete and are superseded. No patient records included.')

def main():
    os.umask(0o077);lock=(WORK/'evaluation_exhaustive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(WORK/'exhaustive_tests_receipt.json').get('state')!='passed':raise RuntimeError('exhaustive_scoring_tests_required')
    if not all(load(WORK/'sft'/a/'training_complete.json') for a in ['fusion','hpo']):raise RuntimeError('completed_SFT_required')
    while not jobs():time.sleep(20)
    plan=load(WORK/'ranking_exhaustive_plan.json') or dict(world=len(jobs()),arms=['fusion','hpo','base_hpo'],
        scoring_implementation='full teacher forcing, no KV reuse, FP32 output head',
        primary_score='mean conditional token log probability, including EOS',secondary_score='sum conditional token log probability, including EOS',
        checkpoints={a:load(WORK/'sft'/a/'best_checkpoint.json')['sha256'] for a in ['fusion','hpo']},
        patient_sha256=load(WORK/'sft_data_receipt.json')['patient_sha256'],created_at_utc=now())
    save(WORK/'ranking_exhaustive_plan.json',plan);world=plan['world'];arms=plan['arms']
    active={};tries={};finished={};tasks=[(a,i) for a in arms for i in range(world)]
    save(PROJECT/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(WORK),workflow_job=os.environ.get('SLURM_JOB_ID'),
        phase='corrected_exhaustive_diagnosis_ranking',status_file=str(WORK/'ranking_exhaustive_status.json'),results_file=str(WORK/'diagnosis_results_exhaustive.json'),updated_at_utc=now()))
    while True:
        done={key for key in tasks if load(WORK/'ranking_exhaustive'/key[0]/f'{key[1]}.receipt.json').get('state')=='complete'}
        if len(done)==len(tasks):break
        used={j for c,j in active.values() if c.poll() is None};free=[j for j in jobs() if j['id'] not in used]
        for key in tasks:
            if key in done or (key in active and active[key][0].poll() is None):continue
            if key in active and active[key][0].returncode==0:
                finished.setdefault(key,time.monotonic())
                if time.monotonic()-finished[key]<180:continue
            if not free:break
            tries[key]=tries.get(key,0)+1
            if tries[key]>2:raise RuntimeError('repeated_exhaustive_ranking_failure')
            arm,i=key;job=free.pop(0)
            active[key]=(launch(job,'ranking_exhaustive.py',[arm,i,world],f'cranio-qwen-exact-{arm}-{i}-a{tries[key]}'),job['id'])
        save(WORK/'ranking_exhaustive_status.json',dict(state='running',completed_shards=len(done),total_shards=len(tasks),allocated_gpus=len(jobs()),updated_at_utc=now()));time.sleep(15)
    score(arms,world);save(WORK/'ranking_exhaustive_status.json',dict(state='complete',updated_at_utc=now()))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'ranking_exhaustive_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('exhaustive_ranking_failure','Corrected local diagnosis ranking requires inspection: '+type(e).__name__+'. Data remain on HPC.')
        except Exception:pass
        raise

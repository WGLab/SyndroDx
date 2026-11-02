"""Run local matched diagnosis ranking after both adapter arms finish."""
import collections,fcntl,os,subprocess,time
import numpy as np
from runtime import *
from controller import jobs,launch
def score(arms,world):
    all_rows={};results={}
    for arm in arms:
        rows=[]
        for i in range(world):
            p=WORK/'ranking'/arm/'private'/f'{i}.json';receipt=load(WORK/'ranking'/arm/f'{i}.receipt.json')
            if digest(p)!=receipt['sha256']:raise RuntimeError('ranking_receipt_mismatch')
            rows+=load(p)['records']
        rows=sorted(rows,key=lambda r:r['index']);all_rows[arm]=rows
        if len(rows)!=load(WORK/'sft_data_receipt.json')['validation_patients'] or len({r['index'] for r in rows})!=len(rows):raise RuntimeError('ranking_patient_coverage_error')
        grouped=collections.defaultdict(list)
        for row in rows:grouped[row['truth']].append(row)
        results[arm]={f'macro_top{k}':float(np.mean([np.mean([r['truth'] in r['ranked'][:k] for r in rr]) for rr in grouped.values()])) for k in [1,5,10]}
        results[arm].update(patients=len(rows),diseases=len(grouped))
    a=all_rows['fusion'];b=all_rows['hpo']
    if [r['index'] for r in a]!=[r['index'] for r in b]:raise RuntimeError('unmatched_control_patients')
    deltas={};rng=np.random.default_rng(20261027)
    for k in [1,5,10]:
        grouped=collections.defaultdict(list)
        for x,y in zip(a,b):grouped[x['truth']].append(int(x['truth'] in x['ranked'][:k])-int(y['truth'] in y['ranked'][:k]))
        boot=np.zeros(2000)
        for v in grouped.values():boot+=rng.choice(v,(2000,len(v)),replace=True).mean(1)/len(grouped)
        deltas[f'top{k}']=dict(difference=float(np.mean([np.mean(v) for v in grouped.values()])),ci95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(state='complete',results=results,fusion_minus_HPO_only=deltas,
        scope='exploratory patient-disjoint GMDB development, selected checkpoints; not external test',
        candidate_diseases=load(WORK/'sft_data_receipt.json')['classes'],method=load(WORK/'sft_protocol.json')['ranking'],
        clinical_reasoning_or_foundation_model_superiority_established=False,updated_at_utc=now())
    save(WORK/'diagnosis_results.json',result)
    notify('diagnosis_validation_complete','Matched Qwen 32B fusion / HPO-only / unchanged HPO-only language model validation completed. Aggregate results are saved at '+str(WORK/'diagnosis_results.json')+'. No patient-level records or images were included.')

def main():
    os.umask(0o077);lock=(WORK/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(WORK/'ranking_tests_receipt.json').get('state')!='passed':raise RuntimeError('generation_tests_required')
    while not all(load(WORK/'sft'/a/'training_complete.json') for a in ['fusion','hpo']):
        if load(WORK/'followon_failure.json'):raise RuntimeError('SFT_failed_before_ranking')
        time.sleep(20)
    # Ensure successful workers have exited before reusing their independent leases.
    while True:
        lines=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%i|%j'],env=env(),capture_output=True,text=True,check=True).stdout
        if 'cranio-qwen-sft-' not in lines:break
        time.sleep(10)
    while not jobs():time.sleep(20)
    plan=load(WORK/'ranking_plan.json') or dict(world=len(jobs()),arms=['fusion','hpo','base_hpo'])
    save(WORK/'ranking_plan.json',plan);world=plan['world'];arms=plan['arms']
    active={};tries={};finished={}
    tasks=[(a,i) for a in arms for i in range(world)]
    while True:
        done={key for key in tasks if load(WORK/'ranking'/key[0]/f'{key[1]}.receipt.json').get('state')=='complete'}
        if len(done)==len(tasks):break
        used={j for c,j in active.values() if c.poll() is None};free=[j for j in jobs() if j['id'] not in used]
        for key in tasks:
            if key in done or (key in active and active[key][0].poll() is None):continue
            if key in active and active[key][0].returncode==0:
                finished.setdefault(key,time.monotonic())
                if time.monotonic()-finished[key]<180:continue
            if not free:break
            tries[key]=tries.get(key,0)+1
            if tries[key]>2:raise RuntimeError('repeated_ranking_failure')
            arm,i=key;job=free.pop(0)
            active[key]=(launch(job,'ranking.py',[arm,i,world],f'cranio-qwen-rank-{arm}-{i}-a{tries[key]}'),job['id'])
        save(WORK/'ranking_status.json',dict(state='running',completed_shards=len(done),total_shards=len(tasks),updated_at_utc=now()));time.sleep(15)
    score(arms,world);save(WORK/'ranking_status.json',dict(state='complete',updated_at_utc=now()))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'ranking_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('ranking_failure','Local diagnosis ranking requires inspection: '+type(e).__name__+'. Data remain on HPC.')
        except Exception:pass
        raise

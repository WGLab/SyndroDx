"""Persistent classifier comparison and aggregate-results email delivery."""
import fcntl,os,subprocess,time
import numpy as np
from runtime import *
from controller import jobs,launch,stop,wait_receipt
def summarize():
    root=WORK/'classifier';results={};differences={}
    source=load(WORK/'private/sft_patients.json');mapping={r['label']:i for i,r in enumerate(source['vocabulary'])}
    val=[r for r in source['patients'] if r['split']=='validation' and r['label'] in mapping];truth=np.array([mapping[r['label']] for r in val]);rng=np.random.default_rng(20261028)
    for stage,filename in [('frozen_probe','probe_scores.npy'),('joint','best_scores.npy')]:
        hits={};results[stage]={}
        for arm in ['fusion','hpo']:
            s=np.load(root/arm/'private'/filename)
            if s.shape!=(len(val),len(mapping)) or not np.isfinite(s).all():raise RuntimeError('classifier_score_coverage_error')
            order=np.argsort(-s,axis=1,kind='stable');hits[arm]={k:np.any(order[:,:k]==truth[:,None],axis=1).astype(float) for k in [1,5,10]}
            results[stage][arm]={f'macro_top{k}':float(np.mean([v[truth==c].mean() for c in np.unique(truth)])) for k,v in hits[arm].items()}
        differences[stage]={}
        for k in [1,5,10]:
            d=hits['fusion'][k]-hits['hpo'][k];boot=np.zeros(2000)
            for c in np.unique(truth):
                v=d[truth==c];boot+=rng.choice(v,(2000,len(v)),replace=True).mean(1)/len(np.unique(truth))
            differences[stage][f'top{k}']=dict(difference=float(np.mean([d[truth==c].mean() for c in np.unique(truth)])),ci95=np.quantile(boot,[.025,.975]).tolist())
    report=dict(state='complete',results=results,facial_increment=differences,patients=len(val),validation_diseases=len(np.unique(truth)),candidate_diseases=len(mapping),
        scope='GMDB patient-disjoint development; exploratory; no CHOP/BCH test used',uncertainty='within fixed development diseases; excludes training-seed uncertainty',
        arms={a:load(root/a/'result.json') for a in ['fusion','hpo']},updated_at_utc=now())
    save(root/'results.json',report)
    body=['Qwen 32B classification-head comparison completed.',f'GMDB development: {len(val)} patients; {len(mapping)} candidate diseases. Disease-macro metrics.','']
    for stage in ['frozen_probe','joint']:
        body.append(stage)
        for arm in ['hpo','fusion']:
            m=results[stage][arm];body.append(f"{arm}: Top1 {100*m['macro_top1']:.2f}%, Top5 {100*m['macro_top5']:.2f}%, Top10 {100*m['macro_top10']:.2f}%")
        for k,d in differences[stage].items():body.append(f"Facial increment {k}: {100*d['difference']:+.2f} percentage points; paired bootstrap 95% CI [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}].")
        body.append('')
    body+=['Checkpoints were selected by development macro Top5 with Top1 tie-break. This is not external clinical validation.',f'Aggregate report: {root}/results.json','All clinical records, IDs, images, and features remain on HPC.']
    save(root/'notification_pending.json',dict(event='classifier_results_complete',body='\n'.join(body),updated_at_utc=now()))
    if not load(root/'notification_sent.json'):
        notify('classifier_results_complete','\n'.join(body));save(root/'notification_sent.json',dict(state='notifier_invoked',updated_at_utc=now()))
def main():
    os.umask(0o077);root=WORK/'classifier';lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if load(root/'tests.json').get('state')!='passed':raise RuntimeError('classifier_tests_required')
    save(PROJECT/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(root),workflow_job=os.environ.get('SLURM_JOB_ID'),phase='Qwen_classification_head_probe_and_joint_training',status_file=str(root/'status.json'),results_file=str(root/'results.json'),updated_at_utc=now()))
    for arm in ['fusion','hpo']:
        out=root/arm;attempt=load(out/'attempt.json').get('attempt',0);failures=0
        while not load(out/'result.json'):
            pool=jobs()
            if not pool:save(root/'status.json',dict(state='waiting_for_GPU',arm=arm,updated_at_utc=now()));time.sleep(20);continue
            attempt+=1;save(out/'attempt.json',dict(attempt=attempt));save(out/'resize_requested.json',dict(world_size=len(pool)))
            batch=load(out/'batch_override.json').get('batch',2)
            clients=[launch(j,'classifier_train.py',[arm],f'cranio-qwen-clf-{arm}-a{attempt}-r{r}',dict(RANK=str(r),WORLD_SIZE=str(len(pool)),LOCAL_RANK='0',MASTER_ADDR=pool[0]['node'],MASTER_PORT=str(34500+attempt+(100 if arm=='hpo' else 0)),CRANIO_ATTEMPT=str(attempt),CRANIO_BATCH=str(batch))) for r,j in enumerate(pool)]
            save(root/'active_training.json',dict(arm=arm,attempt=attempt,jobs=pool,world_size=len(pool),updated_at_utc=now()))
            while any(c.poll() is None for c in clients):
                if any(c.poll() not in (None,0) for c in clients):stop(clients);break
                available=jobs();save(out/'resize_requested.json',dict(world_size=len(available)))
                save(root/'status.json',dict(state='workers_active',arm=arm,world_size=len(pool),available_gpus=len(available),progress=load(out/'status.json'),updated_at_utc=now()));time.sleep(20)
            if all(c.returncode==0 for c in clients):
                resize=load(out/'resize_requested.json').get('world_size',len(pool))!=len(pool)
                wait_receipt(out/('latest_checkpoint.json' if resize else 'result.json'),180)
            errors=[load(out/'workers'/f'{attempt}_{r}.failure.json') for r in range(len(pool))]
            if any(errors) or any(c.returncode for c in clients):
                failures+=1
                if any(e.get('oom') for e in errors) and batch>1:save(out/'batch_override.json',dict(batch=max(1,batch//2),reason='OOM_recovery'))
                elif failures>=2:raise RuntimeError('repeated_classifier_worker_failure')
            if failures>=3:raise RuntimeError('repeated_classifier_worker_failure')
    summarize();save(root/'status.json',dict(state='complete',allocations_preserved=True,updated_at_utc=now()))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'classifier/failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        try:notify('classifier_failure','Qwen classifier workflow requires inspection: '+type(e).__name__+'. Checkpoints and GPU allocations are retained. All diagnostic logs remain local.')
        except Exception:pass
        raise

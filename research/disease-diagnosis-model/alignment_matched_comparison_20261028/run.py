"""Frozen projection comparison, then paired development diagnosis probes."""
import os,sys,time,subprocess,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import load,save,digest,now,free_jobs,environment,SLURM,PYTHON,notify
ALIGN=PROJECT/'arcface1000_hpo_alignment_20261028';TRANSFER=PROJECT/'arcface_transfer_evaluation_20261028'

def main():
    os.umask(0o077)
    import numpy as np,torch,shutil
    receipt=load(ALIGN/'data_receipt.json')
    assert digest(ALIGN/'private/patients.json')==receipt['patient_file_sha256']
    pats=load(ALIGN/'private/training_index.json')['patients']
    tr={p['subject'] for p in pats if p['split']=='train'};va={p['subject'] for p in pats if p['split']=='validation'}
    assert not tr&va
    selected=load(ALIGN/'best_checkpoint.json');assert digest(selected['path'])==selected['sha256']
    save(ROOT/'protocol.json',dict(alignment_sha256=selected['sha256'],visual_checkpoint=load(TRANSFER/'selection.json'),
        fixed_patient_splits=True,train_validation_overlap=0,clinical_tests_used=False,
        diagnosis_selection='same validation macro Top5 rule, maximum50epochs, matched seed/batches/LRs',
        caution='Diagnosis arms differ in modality-specific architecture and parameter count; repeated developer validation, not untouched clinical test.',
        unknown_cross_source_identity_and_near_duplicate_status=load(TRANSFER/'audit.json'),updated_at_utc=now()))
    for name in ['selection.json','input_pins.json']:shutil.copy2(TRANSFER/name,ROOT/name)
    before=np.load(TRANSFER/'private/selected.npy',allow_pickle=False)
    weights=torch.load(selected['path'],map_location='cpu',weights_only=False)['model']['image_projection.weight'].numpy()
    after=before@weights.T;after/=np.maximum(np.linalg.norm(after,axis=1,keepdims=True),1e-12)
    np.save(ROOT/'private/original.npy',before);np.save(ROOT/'private/selected.npy',after)
    spec=importlib.util.spec_from_file_location('comparison_retrieval',ROOT/'retrieval_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.evaluate()
    r=load(ROOT/'results.json');r['model_interpretation']={'original':'ArcFace CPT step1000 BEFORE alignment, raw512 features','selected':'same frozen encoder AFTER learned alignment projection, normalized256 features'}
    save(ROOT/'retrieval_results.json',r)
    with (ROOT/'logs/prepare_diagnosis.private.log').open('ab') as f:
        subprocess.run([str(PYTHON),str(ALIGN/'prepare_fusion.py')],env=environment(),stdout=f,stderr=f,check=True)
    for mode in ['hpo','fusion']:
        if load(ALIGN/'fusion'/mode/'result.json').get('state')=='complete':continue
        paused=None
        if not free_jobs():
            queue=load(TRANSFER/'seed_queue_status.json')
            if queue.get('path'):
                seedroot=Path(queue['path'])
                for arm in load(seedroot/'protocol.json').get('arms',{}):
                    flag=seedroot/arm/'pause_requested.json'
                    if load(seedroot/arm/'exit.json').get('state')!='complete' and not flag.exists():
                        save(flag,dict(reason='temporary_checkpoint_yield_for_matched_diagnosis',updated_at_utc=now()));paused=flag;break
        while not free_jobs():
            save(ROOT/'status.json',dict(state='waiting_for_free_gpu',phase=mode,updated_at_utc=now()));time.sleep(20)
        job=free_jobs()[0];e=environment();e.update(RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',MASTER_ADDR=job['node'],MASTER_PORT='49137')
        cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name=alignment-diagnosis-'+mode,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(ALIGN/'train_fusion.py'),mode]
        save(ROOT/'status.json',dict(state='diagnosis_training',mode=mode,allocation=job['id'],updated_at_utc=now()))
        with (ROOT/'logs'/f'{mode}.private.log').open('ab') as f:
            proc=subprocess.Popen(cmd,env=e,stdout=f,stderr=f)
            while paused and proc.poll() is None:
                out=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%j'],env=environment(),capture_output=True,text=True,timeout=40,check=True).stdout
                if 'alignment-diagnosis-'+mode in out.splitlines():paused.unlink(missing_ok=True);paused=None
                else:time.sleep(5)
            code=proc.wait()
            if paused:paused.unlink(missing_ok=True)
            if code:raise RuntimeError('diagnosis_worker_failed')
    results=dict(state='complete',retrieval=load(ROOT/'retrieval_results.json'),diagnosis={a:load(ALIGN/'fusion'/a/'result.json') for a in ['hpo','fusion']},protocol=load(ROOT/'protocol.json'),updated_at_utc=now())
    save(ROOT/'results.json',results);save(ROOT/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('alignment matched comparison complete',__import__('json').dumps(results,indent=2))
    save(ROOT/'notification.json',dict(state='notifier_invoked',updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
        notify('alignment comparison requires inspection',type(e).__name__+'. Private logs remain on HPC.');raise

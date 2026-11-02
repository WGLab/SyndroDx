"""Use existing CPU allocation to evaluate already-trained primary R50 seeds."""
import os,sys,time,subprocess
from pathlib import Path
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import environment,SLURM,notify

def main():
    os.umask(0o077);cases=['pretrained']+['code_keep_bn_'+str(s) for s in load(WORK/'protocol.json')['seeds']]
    active={};attempts={}
    while True:
        done={c for c in cases if load(WORK/(c+'_inference.json')).get('state')=='complete'}
        if len(done)==len(cases):break
        for c,p in list(active.items()):
            if p.poll() is not None:
                if p.returncode and attempts[c]>=2:raise RuntimeError('CPU_inference_failed')
                del active[c]
        for case in cases:
            if case in done or case in active or len(active)>=2:continue
            attempts[case]=attempts.get(case,0)+1
            e=environment();e['R50_EVAL_DEVICE']='cpu';e['MPLCONFIGDIR']='/tmp/gm_arc_reproduction_mpl'
            cmd=[str(SLURM/'srun'),'--jobid=26242395','--job-name=R50-CPU-test-'+case,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=8G','--overlap','--exact','--mpi=none',str(PYTHON),str(WORK/'evaluate.py'),case]
            with (WORK/'logs'/('cpu_'+case+'.private.log')).open('ab') as f:active[case]=subprocess.Popen(cmd,env=e,stdout=f,stderr=f,start_new_session=True)
        save(WORK/'cpu_evaluation_status.json',dict(state='evaluating_trained_primary_seeds',completed=len(done),total=len(cases),active=list(active),updated_at_utc=now()));time.sleep(15)
    e=environment();e['R50_PRIMARY_ONLY']='1'
    with (WORK/'logs/cpu_primary_score.private.log').open('ab') as f:subprocess.run([str(PYTHON),str(WORK/'evaluate.py'),'score'],env=e,stdout=f,stderr=f,check=True)
    save(WORK/'cpu_evaluation_status.json',dict(state='complete',updated_at_utc=now()))
    notify('Clean R50 primary test complete',__import__('json').dumps(load(WORK/'primary_results.json'),indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:save(WORK/'cpu_evaluation_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

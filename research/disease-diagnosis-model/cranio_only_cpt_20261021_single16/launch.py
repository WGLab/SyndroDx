"""Start a persistent CPU-only coordinator inside an existing GPU allocation."""
import json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
BASE=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
sys.path.insert(0,str(BASE))
from common import environment,PYTHON,SLURM,save,now

def main():
    if not (WORK/'cutover.json').exists():raise RuntimeError('old_steps_not_confirmed_stopped')
    env=environment()
    env['PYTHONPATH']=os.pathsep.join(map(str,[WORK,WORK.parent/'training_diagnostics_20261017/vendor/dinov3_fino',BASE,BASE.parent/'dinov3_craniofacial_v1',WORK.parent/'training_diagnostics_20261017']))
    command=[str(SLURM/'srun'),'--jobid=25530883','--job-name=cranio-single16-coordinator',
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=2G','--gres=none',
        '--overlap','--exact','--cpu-bind=none','--mpi=none','--time=11:00:00',str(PYTHON),str(WORK/'controller.py')]
    with (WORK/'coordinator.private.log').open('ab') as stream:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    save(WORK/'coordinator_launch.json',dict(state='launched',parent_job='25530883',client_pid=proc.pid,world_size=16,independent_models=1,launched_at_utc=now()))
    print(json.dumps(dict(state='coordinator_submitted',world_size=16,independent_models=1)))

if __name__=='__main__':main()

import time,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'transface_objective_comparison_20261028'))
from matched_runtime import load,environment,SLURM,PYTHON
# Do not duplicate the live first worker after the old CPU coordinator failed.
while not load(ROOT/'original_lr0.0001_seed11/result.json'):
 time.sleep(10)
q=load(ROOT.parent/'arcface_transfer_evaluation_20261028/seed_queue_status.json')
if q.get('path'):
 for p in Path(q['path']).glob('*/pause_requested.json'):
  if load(p).get('reason')=='temporary_checkpoint_yield_for_GMDB_disease_finetuning':p.unlink(missing_ok=True)
cmd=[str(SLURM/'srun'),'--jobid=26242395','--job-name=WCE-controller-recovery','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=2G','--overlap','--exact','--mpi=none',str(PYTHON),str(ROOT/'controller.py')]
with (ROOT/'logs/recovered_controller.private.log').open('ab') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=f,check=True)

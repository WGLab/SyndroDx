"""Restore background seed scheduling after the finite WCE main task completes."""
import os,sys,time,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'transface_objective_comparison_20261028'))
from matched_runtime import load,environment,SLURM,PYTHON,save,now
while load(ROOT/'status.json').get('state')!='complete':time.sleep(30)
(ROOT.parent/'arcface_extra_seeds_20261028/stop_continuous_seeds.json').unlink(missing_ok=True)
for host,job in [('c-11-04','26242395'),('c-11-07','26247158')]:
    record=ROOT/('parked_cpu_'+host+'.json')
    code='''import os,signal,json
from pathlib import Path
data=json.loads(Path(__import__('sys').argv[1]).read_text())
for item in data['processes']:
 if item['script']=='controller.py':continue
 p=Path('/proc')/str(item['pid'])
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if any(a.endswith(('/'+item['script']).encode()) for a in args):os.kill(item['pid'],signal.SIGCONT)
 except OSError:pass
'''
    subprocess.run([str(SLURM/'srun'),'--jobid='+job,'--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--overlap','--exact','--mpi=none',str(PYTHON),'-c',code,str(record)],env=environment(),check=True)
save(ROOT/'background_resumed.json',dict(state='seed_submission_resumed',updated_at_utc=now()))

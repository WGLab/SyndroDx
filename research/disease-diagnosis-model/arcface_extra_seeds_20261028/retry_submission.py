import subprocess,time,json,os
from pathlib import Path
root=Path(__file__).resolve().parent
slurm='/opt/site/slurm/bin/'
while True:
 try:
  q=subprocess.run([slurm+'squeue','-u','research_user','-h','-o','%i|%j'],capture_output=True,text=True,timeout=40,check=True).stdout
  matches=[l.split('|')[0] for l in q.splitlines() if l.endswith('|arcface-extra-seeds')]
  if matches:
   (root/'submission.json').write_text(json.dumps({'state':'submitted','job_ids':matches}));break
  s=subprocess.run([slurm+'sbatch','--parsable',str(root.parent/'arcface_transfer_evaluation_20261028/extra_seeds.sbatch')],capture_output=True,text=True,timeout=45,check=True)
  (root/'submission.json').write_text(json.dumps({'state':'submitted','job_id':s.stdout.strip()}));break
 except (subprocess.SubprocessError,OSError):time.sleep(30)

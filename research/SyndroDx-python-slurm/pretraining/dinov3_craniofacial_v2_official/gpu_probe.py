"""Node-local aggregate GPU telemetry, restricted to this training entrypoint."""
import os,json,subprocess,time,socket
from pathlib import Path
from common import ROOT
owned=set()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if p.stat().st_uid==os.getuid() and os.fsencode(ROOT/'train_entry.py') in args:
   allocation=(p/'environ').read_bytes().split(b'\0')
   if ('SLURM_JOB_ID='+os.environ.get('SLURM_JOB_ID','')).encode() in allocation:owned.add(int(p.name))
 except OSError:pass
try:
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
 uuids={l.split(',')[1].strip() for l in result.stdout.splitlines() if l.split(',')[0].strip().isdigit() and int(l.split(',')[0].strip()) in owned}
 samples=[]
 for _ in range(3):
  result=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
  for line in result.stdout.splitlines():
   uuid,*numbers=[x.strip() for x in line.split(',')]
   if uuid in uuids:samples.append([float(x) for x in numbers])
  if _<2:time.sleep(2)
 receipt=dict(node=socket.gethostname(),observed_project_GPUs=len(uuids),sample_seconds=4,state='sampled' if samples else 'no_project_GPU_process')
 if samples:receipt.update(mean_utilization_percent=sum(v[0] for v in samples)/len(samples),minimum_utilization_percent=min(v[0] for v in samples),maximum_memory_used_GiB=max(v[1] for v in samples)/1024)
 print(json.dumps(receipt))
except Exception as exc:print(json.dumps(dict(node=socket.gethostname(),state='probe_failed',error_type=type(exc).__name__)))

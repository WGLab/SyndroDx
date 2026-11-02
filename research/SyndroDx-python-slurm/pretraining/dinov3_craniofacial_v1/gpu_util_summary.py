import os,subprocess,time,json
from pathlib import Path
from runtime import ROOT,RUN
scripts={str(ROOT/'independent_qc.py'),str(ROOT/'dual_crop_qc_worker.py'),str(ROOT/'extra_crop_worker.py'),str(ROOT/'extra_qc_tail_worker.py'),str(RUN/'crop_pipeline/crop_worker.py')}
owned=set()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  if proc.stat().st_uid==os.getuid() and any(x in scripts for x in args):owned.add(int(proc.name))
 except (OSError,UnicodeError):continue
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
uuids={line.split(',')[1].strip() for line in apps.splitlines() if int(line.split(',')[0].strip()) in owned}
values={u:[] for u in uuids}
for i in range(6):
 result=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 for line in result.splitlines():
  uuid,value=[v.strip() for v in line.split(',')]
  if uuid in values:values[uuid].append(float(value))
 if i<5:time.sleep(3)
means=sorted(round(sum(v)/len(v),1) for v in values.values() if v)
print(json.dumps({'owned_GPUs_with_worker_processes':len(means),'utilization_percent_per_GPU_sorted':means,'mean_utilization':round(sum(means)/len(means),1) if means else None,'sampling_seconds':15}))

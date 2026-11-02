"""Local-only orchestration shared by QC, hardware probes and SSL training."""
import datetime,fcntl,json,os,re,subprocess
from pathlib import Path
ROOT=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
RUN=Path('/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d')
SLURM=Path('/opt/site/slurm/bin')
# Cron does not inherit the interactive environment; Slurm otherwise attempts
# DNS discovery and both recovery watchdogs fail before inspecting any jobs.
os.environ.setdefault('SLURM_CONF','/opt/site/slurm/slurm.conf')
VENDOR=ROOT/'vendor/dinov3'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {} if default is None else default
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def digest(path):
 import hashlib
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def environment():
 env=os.environ.copy()
 # Binding masks from a CPU coordinator step do not apply on a different GPU node.
 # Slurm still enforces the destination allocation's CPU/GPU cgroups.
 for k in ['SLURM_CPU_BIND','SLURM_CPU_BIND_LIST','SLURM_CPU_BIND_TYPE','SLURM_CPU_BIND_VERBOSE','SLURM_GPU_BIND']:env.pop(k,None)
 for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN']:env.pop(k,None)
 env.update(HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DO_NOT_TRACK='1',WANDB_DISABLED='true',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',NCCL_DEBUG='WARN',NCCL_P2P_DISABLE='1',NCCL_CUMEM_HOST_ENABLE='0',NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
 # The SHM transport passed cross-allocation tests. Do not bypass Slurm GPU isolation.
 env['PYTHONPATH']=str(VENDOR)+os.pathsep+str(ROOT)
 return env
def scheduler():
 r=subprocess.run([str(SLURM/'squeue'),'-h','-u','research_user','-o','%i|%T|%N|%b|%j'],capture_output=True,text=True,timeout=25,check=True)
 jobs={}
 for line in r.stdout.splitlines():
  fields=line.strip().split('|')
  if len(fields)!=5:continue
  job,state,node,gres,name=fields
  if not(name.startswith(('facecrop-local-','sslpool-','sslmax-','sslprep-')) and ('h100' in gres or node.startswith('g-44-'))):continue
  n=re.search(r'gpu(?::h100)?:(\d+)',gres)
  jobs[job]={'job_id':job,'state':state,'node':node,'gpus':int(n[1]) if n else 1,'name':name}
 r=subprocess.run([str(SLURM/'squeue'),'-h','--steps','-u','research_user','-o','%i'],capture_output=True,text=True,timeout=25,check=True)
 busy={s.strip().split('.')[0] for s in r.stdout.splitlines() if s.strip() and not s.strip().endswith(('.extern','.batch'))}
 return jobs,busy
def claim_jobs(jobs):
 locks=[]
 try:
  for job in sorted(set(jobs)):
   path=ROOT/'resource_locks'/('allocation_'+str(job)+'.lock');path.parent.mkdir(exist_ok=True)
   f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  return locks
 except BaseException:
  for f in locks:f.close()
  raise

def notify(event,title,message):
 import importlib.util
 path=RUN/'notify_pipeline_stages.py'
 spec=importlib.util.spec_from_file_location('local_pipeline_notifications',path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.notify(event,title,message)

"""Local operational helpers; no clinical content is printed or transmitted."""
import os,sys,json,datetime,subprocess
from pathlib import Path
ROOT=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
V1=ROOT.parent/'dinov3_craniofacial_v1'
UPSTREAM=V1/'vendor/dinov3'
PYTHON=V1/'venv/bin/python'
SLURM=Path('/opt/site/slurm/bin')
os.environ.setdefault('SLURM_CONF','/opt/site/slurm/slurm.conf')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {} if default is None else default

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def digest(path):
 import hashlib
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def environment():
 sys.path.insert(0,str(V1));from runtime import environment as legacy_environment
 env=legacy_environment();env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,UPSTREAM,V1]));return env

def notify(event,title,message):
 sys.path.insert(0,str(V1));from runtime import notify as local_notify
 local_notify('official CPT v2 '+event,title,message)
def install_rope_dtype_compatibility():
 # Upstream factory in the pinned commit omits this declared config field.
 # Preserve FP32 positional arithmetic of the published checkpoint; no change
 # to the architecture, losses, optimizer or training loop.
 import dinov3.models.vision_transformer as vit
 if getattr(vit.vit_7b,'_public_rope_compat',False):return
 original=vit.vit_7b
 def factory(**kwargs):
  kwargs.setdefault('pos_embed_rope_dtype','fp32')
  return original(**kwargs)
 factory._public_rope_compat=True;vit.vit_7b=factory

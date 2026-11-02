"""Offline workflow helpers; clinical records never enter operational logs."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
STAGE1=PROJECT/'transface_arcface_identity_cpt_20261026'
SHARED=PROJECT/'gmdb_checkpoint_eval_20261026'
LEASES=PROJECT/'cranio_only_cpt_20261025_converge20'
PYTHON=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
SLURM=Path('/opt/site/slurm/bin')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.chmod(tmp,0o600);tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') and k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
    env.update(PYTHONPATH=str(WORK)+':'+str(STAGE1/'dependencies'),PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        WANDB_MODE='disabled',WANDB_DISABLED='true',DO_NOT_TRACK='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        SLURM_CONF='/opt/site/slurm/slurm.conf')
    return env
def notify(event,title,message):
    folder=PYTHON.parents[2]
    spec=importlib.util.spec_from_file_location('cranio_follow_on_mail_runtime',folder/'runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    full_event='Craniofacial automatic workflow '+event
    module.notify(full_event,title,message)
    receipt=folder.parent.parent/'face_ssl_annotation/run_20261012_baed142d/notifications'/(full_event+'.json')
    if load(receipt).get('state')!='accepted_by_local_mail_system':
        raise RuntimeError('mail_delivery_not_confirmed_by_local_mail_system')

def verify_code():
    receipt=load(WORK/'code_receipt.json')
    if not receipt:raise RuntimeError('pinned_workflow_code_required')
    for name,sha in receipt.items():
        if digest(WORK/name)!=sha:raise RuntimeError('workflow_code_integrity_failure')

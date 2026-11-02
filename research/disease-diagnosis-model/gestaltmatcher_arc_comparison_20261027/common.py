"""Private HPC benchmark operations and immutable experiment inputs."""
import datetime,hashlib,json,os
from pathlib import Path
WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
STAGE1=PROJECT/'transface_arcface_identity_cpt_20261026'
SHARED=PROJECT/'gmdb_checkpoint_eval_20261026'
PREVIOUS=PROJECT/'gmdb_supervised_transface_20261026'
PYTHON=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return {} if default is None else default
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(d,indent=2)+'\n');os.chmod(tmp,0o600);tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def verify_code():
    for name,sha in load(WORK/'code_receipt.json').items():
        if digest(WORK/name)!=sha:raise RuntimeError('benchmark_code_changed')

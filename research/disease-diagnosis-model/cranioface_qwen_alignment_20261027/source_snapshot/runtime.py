"""Local-only craniofacial connector experiment utilities."""
import datetime, hashlib, importlib.util, json, os
from pathlib import Path
WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
PREVIOUS=PROJECT/'transface_biomedbert_alignment_20261027'
STAGE1=PROJECT/'transface_arcface_identity_cpt_20261026'
GMDB=PROJECT/'gmdb_supervised_transface_20261026'
PYTHON=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
SLURM=Path('/opt/site/slurm/bin')
MODEL=Path('/controlled-data/cache_storage/huggingface_hub_20261014/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return {} if default is None else default
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(d,indent=2)+'\n');os.chmod(t,0o600);t.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def env():
    e={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SALLOC_')) and k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
    e.update(PYTHONPATH=str(WORK)+':'+str(STAGE1/'dependencies'),PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        WANDB_MODE='disabled',WANDB_DISABLED='true',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        SLURM_CONF='/opt/site/slurm/slurm.conf',NCCL_DEBUG='WARN',
        NCCL_P2P_DISABLE='1',NCCL_CUMEM_HOST_ENABLE='0',NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return e
def notify(event,text):
    s=importlib.util.spec_from_file_location('existing_local_notifier',PYTHON.parents[2]/'runtime.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.notify('Craniofacial Qwen '+event,'Craniofacial Qwen: '+event,text)

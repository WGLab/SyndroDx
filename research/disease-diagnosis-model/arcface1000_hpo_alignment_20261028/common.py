"""HPC-only alignment utilities; never log patient-level inputs."""
import datetime
import hashlib
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
STAGE1 = PROJECT / 'transface_arcface_identity_cpt_20261026'
GMDB_STAGE = PROJECT / 'gmdb_supervised_transface_20261026'
SHARED = PROJECT / 'gmdb_checkpoint_eval_20261026'
METADATA = Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata')
HPO = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/HPO/hp.obo')
PYTHON = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
SLURM = Path('/opt/site/slurm/bin')
POOL = Path('/opt/craniofacial/operations/gpu_salloc20')
MODEL_ID = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return {} if default is None else default
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n'); os.chmod(temp, 0o600); temp.replace(path)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def environment():
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'SALLOC_', 'SBATCH_')) and k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
    env.update(PYTHONPATH=str(WORK)+':'+str(STAGE1/'dependencies'), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', WANDB_DISABLED='true', DO_NOT_TRACK='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        SLURM_CONF='/opt/site/slurm/slurm.conf', NCCL_DEBUG='WARN',
        NCCL_P2P_DISABLE='1', NCCL_CUMEM_HOST_ENABLE='0', NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return env

def notify(event, message):
    import importlib.util
    spec=importlib.util.spec_from_file_location('alignment_mail', PYTHON.parents[2]/'runtime.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.notify('TransFace BiomedBERT alignment '+event, 'Craniofacial alignment: '+event, message)

"""Local aggregate-only control helpers for ArcFace clinical SSL."""
import datetime
import hashlib
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
OLD = WORK.parent / 'cranio_only_cpt_20261025_converge20'
V1 = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
BASE = V1.parent / 'dinov3_craniofacial_v2_official'
PYTHON = V1 / 'venv/bin/python'
SLURM = Path('/opt/site/slurm/bin')
ONNX = WORK.parent / 'arcface_glint360k_eval_20261026/assets/glint360k_r100.onnx'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.chmod(temporary, 0o600)
    temporary.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def environment():
    env = os.environ.copy()
    for key in ['SLURM_CPU_BIND','SLURM_CPU_BIND_LIST','SLURM_CPU_BIND_TYPE','SLURM_CPU_BIND_VERBOSE',
                'SLURM_GPU_BIND','SLURM_MEM_PER_NODE','SLURM_MEM_PER_CPU','SLURM_MEM_PER_GPU',
                'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN']:
        env.pop(key, None)
    env.update(PYTHONPATH=str(WORK / 'dependencies'), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', WANDB_DISABLED='true',
        DO_NOT_TRACK='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        SLURM_CONF='/opt/site/slurm/slurm.conf',
        NCCL_DEBUG='WARN', NCCL_P2P_DISABLE='1', NCCL_CUMEM_HOST_ENABLE='0',
        NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return env

def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def alive(record):
    return bool(record.get('pid') and record.get('start_identity') and
                identity(record['pid']) == record['start_identity'])

def notify(event, title, message):
    import sys
    sys.path.insert(0, str(V1))
    from runtime import notify as send
    send('TransFace identity ArcFace CPT ' + event, title, message)

WEIGHTS = WORK.parent / 'face_vitl_acquisition_20261026/assets/glint360k_model_TransFace_L.pt'

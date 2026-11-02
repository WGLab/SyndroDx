"""HPC-local matched unfreezing experiment. Aggregate receipts only."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
MATCH = PROJECT / 'matched_craniofacial_benchmark_20261031'
BASE = PROJECT / 'hpo_fusion_staged_20261031'
QUERY = PROJECT / 'hpo_query_fusion_20261101'
PYTHON = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
SLURM = Path('/opt/site/slurm/bin')
os.umask(0o077)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return {}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.chmod(0o600); tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SALLOC_', 'SBATCH_')) and k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN')}
    env.update(PYTHONPATH=str(WORK), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
               WANDB_DISABLED='true', DO_NOT_TRACK='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
               SLURM_CONF='/opt/site/slurm/slurm.conf')
    return env


def runtime():
    spec = importlib.util.spec_from_file_location('unfreeze_allocation_runtime', PROJECT / 'transface_objective_comparison_20261028/matched_runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def free_jobs(): return runtime().free_jobs()


def notify(event, body): return runtime().notify('matched unfreezing ' + event, body)


def guarded(fn, failure_path):
    try: fn()
    except Exception as e:
        import traceback
        frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(e.__traceback__)]
        save(failure_path, dict(error_type=type(e).__name__,code_locations=frames,updated_at_utc=now()))
        raise SystemExit(1) from None

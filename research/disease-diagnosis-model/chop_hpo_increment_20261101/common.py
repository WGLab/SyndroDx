"""Offline experiment helpers. Logs and notifications contain aggregates only."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
MATCH = PROJECT / 'matched_craniofacial_benchmark_20261031'
OLD = PROJECT / 'arcface1000_hpo_alignment_20261028'
SHARED = PROJECT / 'gmdb_checkpoint_eval_20261026'
PYTHON = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1/venv/bin/python')
SLURM = Path('/opt/site/slurm/bin')
TEXT = OLD / 'assets/biomedbert'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.chmod(temp, 0o600)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SLURM_', 'SALLOC_', 'SBATCH_'))
           and k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN')}
    env.update(PYTHONPATH=str(WORK) + ':' + str(PROJECT / 'transface_arcface_identity_cpt_20261026/dependencies'),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
               WANDB_DISABLED='true', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MPLCONFIGDIR='/tmp/hpo_fusion_mpl',
               SLURM_CONF='/opt/site/slurm/slurm.conf')
    return env


def free_jobs():
    import importlib.util
    spec = importlib.util.spec_from_file_location('hpo_pool_runtime', PROJECT / 'transface_objective_comparison_20261028/matched_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.free_jobs()


def notify(event, body):
    import importlib.util
    runtime = PYTHON.parents[2] / 'runtime.py'
    spec = importlib.util.spec_from_file_location('hpo_local_mail', runtime)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.notify('HPO fusion ' + event, 'Craniofacial HPO fusion: ' + event, body)

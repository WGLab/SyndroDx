"""Shared, offline experiment utilities; private data never appear in reports."""
import os, sys, json, hashlib, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ID = PROJECT/'transface_arcface_identity_cpt_20261026'
SSL = PROJECT/'transface_cpt_low_lr_20261026'
sys.path.insert(0, str(ID))
sys.path.insert(0, str(ID/'dependencies'))
from control import PYTHON, SLURM, V1, BASE, WEIGHTS, load, save, digest, now
POOL = Path('/opt/craniofacial/operations/gpu_salloc20')
PAIRS = PROJECT/'transface_biomedbert_alignment_20261027'
GMDB = PROJECT/'gmdb_supervised_transface_20261026'
ARMS = ['arcface_seed_20262184']

def environment():
    e={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SALLOC_','SBATCH_')) and k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN')}
    e.update(PYTHONPATH=str(ROOT)+':'+str(ID/'dependencies'), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        WANDB_MODE='disabled',WANDB_DISABLED='true',DO_NOT_TRACK='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',SLURM_CONF='/opt/site/slurm/slurm.conf',
        NCCL_DEBUG='WARN',NCCL_P2P_DISABLE='1',NCCL_CUMEM_HOST_ENABLE='0',NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return e

def notify(event, body):
    import importlib.util
    spec=importlib.util.spec_from_file_location('matched_mail',V1/'runtime.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.notify('TransFace objective comparison '+event,'Craniofacial encoder comparison: '+event,body)

def command(args):
    import subprocess
    return subprocess.run([str(SLURM/args[0]),*args[1:]],env=environment(),capture_output=True,text=True,check=True,timeout=40).stdout

def free_jobs():
    text=command(['squeue','-u','research_user','-h','-o','%i|%j|%T|%N'])
    steps=command(['squeue','--steps','-u','research_user','-h','-o','%i'])
    busy={s.split('.')[0] for s in steps.splitlines() if s.rsplit('.',1)[-1].isdigit()}
    return sorted([dict(id=v[0],node=v[3]) for line in text.splitlines()
        if len(v:=line.split('|'))==4 and v[1].startswith('h100-hold20-') and v[2]=='RUNNING' and v[0] not in busy],key=lambda j:(j['node'],int(j['id'])))

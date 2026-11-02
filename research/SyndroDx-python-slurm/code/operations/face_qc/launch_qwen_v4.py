"""Launch the frozen ten-worker expansion in existing Slurm allocations."""
import json
import os
import subprocess
import time

from stream_qwen_review_v4 import OUT, ROOT

manifest = json.loads((OUT / 'manifest.json').read_text())
assert manifest['workers'] == 10
assert not (OUT / 'launch.json').exists(), 'Already launched'
env = os.environ.copy()
env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
python = '/opt/craniofacial/user_home/miniconda3/envs/qwen-finetune/bin/python'
launches = []
for i in range(10):
    job = '25182707' if i < 7 else ('25185331' if i == 7 else '25185332')
    kind = 'h100' if i < 7 else 'a100'
    cmd = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--exact', '--cpus-per-task=4',
           f'--gres=gpu:{kind}:1', '--mem=60G', python, '-u', 'face_qc/stream_qwen_review_v4.py', 'run', '--worker', str(i)]
    log = ROOT / f'face_qc/logs/qwen_v4_worker_{i}.out'
    with log.open('xb') as output:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    launches.append({'worker': i, 'allocation': job, 'gpu_type': kind, 'launcher_pid': proc.pid,
                     'command': cmd, 'log': str(log)})
    (OUT / 'launch.json').write_text(json.dumps({'at_unix': time.time(), 'workers': launches}, indent=2))
cmd = ['srun', '--jobid=25182707', '--nodes=1', '--ntasks=1', '--exact', '--cpus-per-task=1',
       '--gres=none', '--mem=2G', python, '-u', 'face_qc/summarize_qwen_stream_v4.py']
with (ROOT / 'face_qc/logs/qwen_v4_summary.out').open('xb') as output:
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
(OUT / 'summary_launch.json').write_text(json.dumps({'pid': proc.pid, 'command': cmd}, indent=2))
print(json.dumps({'launched_workers': len(launches), 'summary_pid': proc.pid}))

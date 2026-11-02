"""Local-only evaluation of the user's R50 reproduction against CHOP CPT."""
import os,sys,time,subprocess,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import load,save,digest,now,free_jobs,environment,PYTHON,SLURM,notify
sys.path.insert(0,str(PROJECT/'arcface_cpt_20261026/dependencies'))

def model():
    import torch
    from onnx2torch import convert
    topology=ROOT/'topology/recognition/model.onnx'
    m=convert(str(topology));s=torch.load(ROOT/'assets/s1_glint360k_r50_512d_gmdb_v1.1.2_e50.pt',map_location='cpu',weights_only=True)
    translated={}
    for k,v in s.items():
        if k.startswith('base.'):translated[k[5:]]=v
        elif k.startswith('features.Gemm_0.'):translated[k.replace('features.Gemm_0.','Gemm_128.')]=v
        elif k.startswith('features.BatchNormalization_129.'):translated[k.replace('features.','')]=v
        elif not k.startswith('classifier.'):raise RuntimeError('unrecognized_weight_key')
    assert set(translated)==set(m.state_dict())
    m.load_state_dict(translated,strict=True)
    save(ROOT/'model_audit.json',dict(state='strict_load_passed',weight_keys=len(translated),
        topology_sha256=digest(topology),checkpoint_sha256=digest(ROOT/'assets/s1_glint360k_r50_512d_gmdb_v1.1.2_e50.pt'),
        reference_topology='immich-app/buffalo_l recognition R50; every checkpoint parameter/buffer matched by exact operator key and shape',
        missing_or_unexpected_keys=0,classifier_excluded=True,original_training_manifest_available=False,updated_at_utc=now()))
    return m.eval().requires_grad_(False)

def worker():
    import torch,numpy as np
    sys.path.insert(0,str(PROJECT/'transface_gmdb_diagnostic_20261026'))
    # Load just the validated image decoder under an explicit module name.
    spec=importlib.util.spec_from_file_location('gmdb_decoder',PROJECT/'transface_gmdb_diagnostic_20261026/worker.py')
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    import json
    rows=[json.loads(l) for l in (PROJECT/'gmdb_checkpoint_eval_20261026/inputs.private.jsonl').open()]
    assert digest(PROJECT/'gmdb_checkpoint_eval_20261026/inputs.private.jsonl')==load(ROOT/'input_pins.json')['inputs']
    torch.set_num_threads(4);torch.cuda.set_device(0);m=model().cuda();values=[]
    from torch.utils.data import DataLoader
    with torch.inference_mode():
        for x in DataLoader(decoder.Images(rows,list(range(len(rows)))),batch_size=256,num_workers=2,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):f=m(x.cuda(non_blocking=True))
            f=torch.nn.functional.normalize(f.float(),dim=-1)
            if not torch.isfinite(f).all():raise RuntimeError('nonfinite_embedding')
            values.append(f.cpu().numpy())
    np.save(ROOT/'private/selected.npy',np.concatenate(values))
    save(ROOT/'inference_complete.json',dict(state='complete',updated_at_utc=now()))

def main():
    os.umask(0o077)
    # Strict structural load is checked before any patient image is decoded.
    m=model();del m
    if not load(ROOT/'inference_complete.json'):
        while not free_jobs():
            save(ROOT/'status.json',dict(state='waiting_for_free_gpu',updated_at_utc=now()));time.sleep(20)
        j=free_jobs()[0]
        cmd=[str(SLURM/'srun'),'--jobid='+j['id'],'--job-name=gmdb-hf-R50-evaluation','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(ROOT/'evaluate.py'),'worker']
        save(ROOT/'status.json',dict(state='inference',allocation=j['id'],updated_at_utc=now()))
        with (ROOT/'logs/inference.private.log').open('ab') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=f,check=True)
    import shutil
    shutil.copy2(PROJECT/'arcface_transfer_evaluation_20261028/private/selected.npy',ROOT/'private/original.npy')
    spec=importlib.util.spec_from_file_location('r50_comparison_metrics',ROOT/'metric_driver.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.evaluate()
    r=load(ROOT/'results.json');r['models_description']={'original':'CHOP identity CPT TransFace-L step1000, no GMDB disease fine-tuning','selected':'aaronwzl R50 reproduction, GMDB v1.1.2 disease-fine-tuned'}
    r['comparison_limitations']=['R50 GMDB v1.1.2 training patient manifest unavailable: overlap with v1.1.0 evaluation cannot be excluded.','Not author-official GestaltMatcher-Arc weights.','Training exposures and supervision differ; this is a descriptive comparison.']
    save(ROOT/'results.json',r);save(ROOT/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('user supplied GMDB R50 comparison complete',__import__('json').dumps(r,indent=2))

if __name__=='__main__':
    try:
        if len(sys.argv)>1:worker()
        else:main()
    except Exception as e:
        save(ROOT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

"""Unmodified Glint360K iResNet-100, without GMDB training, evaluated offline."""
import ast
import concurrent.futures
import csv
import importlib.metadata
import json
import os
import sys
from pathlib import Path

WORK=Path(__file__).resolve().parent
SHARED=WORK.parent/'gmdb_checkpoint_eval_20261026'
V1=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
BASE=V1.parent/'dinov3_craniofacial_v2_official'
sys.path[:0]=[str(WORK/'dependencies'),str(SHARED),str(BASE),str(V1)]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',WANDB_MODE='disabled')
import numpy as np
import cv2
from common import load,save,digest,now,notify
from metrics import evaluate,TOP_K


def preprocess(image):
    """Matches official predict_ensemble.preprocess(gray=False, flip=False)."""
    image=cv2.resize(image,(112,112))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=np.transpose(image,(2,0,1)).astype(np.float32)
    result/=np.float32(255);result-=np.float32(.5);result/=np.float32(.5)
    return np.ascontiguousarray(result)


def read_image(row):
    if digest(row['crop_path'])!=row['crop_sha256']:
        raise RuntimeError('frozen_gmdb_crop_modified')
    image=cv2.imread(row['crop_path'],cv2.IMREAD_COLOR)
    if image is None:raise RuntimeError('gmdb_crop_decode_failed')
    return preprocess(image)


def metric_driver():
    # Reuse the exact driver used for the two DINOv3 encoders, without importing
    # their CUDA inference modules into this independent CPU benchmark.
    source=SHARED/'evaluate_checkpoint.py'
    parsed=ast.parse(source.read_text())
    function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run_metrics')
    scope=dict(np=np,evaluate=evaluate,TOP_K=TOP_K)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
    return scope['run_metrics']


def main():
    import onnxruntime as ort
    ort.disable_telemetry_events()
    os.umask(0o077);cv2.setNumThreads(1)
    protocol=load(WORK/'protocol.json');shared=load(SHARED/'protocol.json')
    if protocol.get('state')!='prepared':raise RuntimeError('prepared_protocol_required')
    for path,key in [(SHARED/'protocol.json','reference_protocol_sha256'),
                     (SHARED/'results.json','reference_results_sha256'),
                     (SHARED/'evaluate_checkpoint.py','metric_driver_sha256'),
                     (SHARED/'inputs.private.jsonl','inputs_sha256'),
                     (WORK/'assets/glint360k_r100.onnx','model_sha256')]:
        if digest(path)!=protocol[key]:raise RuntimeError('frozen_benchmark_input_modified')
    for name,sha in shared['split_sha256'].items():
        if digest(SHARED/'split_snapshot'/(name+'.csv'))!=sha:
            raise RuntimeError('frozen_split_modified')
    with (SHARED/'inputs.private.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    options=ort.SessionOptions();options.intra_op_num_threads=int(os.environ.get('SLURM_CPUS_PER_TASK','16'))
    options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level=3
    save(WORK/'status.json',dict(state='loading_model',updated_at_utc=now()))
    session=ort.InferenceSession(str(WORK/'assets/glint360k_r100.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name
    dimension=session.get_outputs()[0].shape[-1]
    if dimension!=512:raise RuntimeError('expected_official_512d_model')
    # Output metadata from this published ONNX says batch=1 even though its
    # input is dynamic. Verify actual output shapes before batching real data.
    smoke=session.run(None,{input_name:np.zeros((2,3,112,112),dtype=np.float32)})[0]
    if smoke.shape!=(2,512) or not np.isfinite(smoke).all():
        raise RuntimeError('model_batch_shape_or_finite_smoke_test_failed')
    save(WORK/'runtime_receipt.json',dict(state='passed',model_sha256=protocol['model_sha256'],
        onnxruntime=ort.__version__,numpy=np.__version__,providers=session.get_providers(),
        model_output_dimension=512,batch_two_smoke_test_passed=True,telemetry_disabled=True,updated_at_utc=now()))
    path=WORK/'features.private.npy';progress=load(WORK/'feature_progress.json')
    if progress and progress.get('inputs_sha256')!=protocol['inputs_sha256']:
        raise RuntimeError('cache_input_mismatch')
    completed=int(progress.get('completed',0));batch=32
    if completed and not path.exists():raise RuntimeError('incomplete_feature_cache')
    features=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(rows),512))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(completed,len(rows),batch):
            selected=rows[start:start+batch]
            inputs=np.stack(list(pool.map(read_image,selected)))
            values=session.run(None,{input_name:inputs})[0]
            if values.shape!=(len(selected),512) or not np.isfinite(values).all():
                raise RuntimeError('invalid_arcface_features')
            norms=np.linalg.norm(values,axis=1,keepdims=True)
            if np.min(norms)<1e-10:raise RuntimeError('zero_norm_arcface_feature')
            features[start:start+len(selected)]=values/norms;features.flush()
            state=dict(state='extracting',completed=start+len(selected),total=len(rows),batch_size=batch,
                inputs_sha256=protocol['inputs_sha256'],updated_at_utc=now())
            save(WORK/'feature_progress.json',state);save(WORK/'status.json',state)
    del session
    save(WORK/'status.json',dict(state='computing_retrieval_metrics',updated_at_utc=now()))
    from settings import TABLES
    tables={}
    for name in TABLES:
        with (SHARED/'split_snapshot'/(name+'.csv')).open(newline='') as stream:
            tables[name]=list(csv.DictReader(stream))
    arc=metric_driver()('arcface_r100_glint360k_no_gmdb',features,rows,tables)
    reference=load(SHARED/'results.json');models=dict(reference['models'])
    models['arcface_r100_glint360k_no_gmdb']=arc
    differences={variant:{case:{key:models['craniofacial_cpt'][variant][case]['macro'][key]-value
        for key,value in score['macro'].items()} for case,score in cases.items()} for variant,cases in arc.items()}
    save(WORK/'results.json',dict(state='complete',models=models,cpt_minus_unadapted_arcface=differences,
        cpt_checkpoint_step=reference['checkpoint_step'],gmdb_finetuned=False,single_model=True,tta=False,
        exploratory=True,confidence_intervals_computed=False,protocol=str(WORK/'protocol.json'),updated_at_utc=now()))
    save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))
    notify('unadapted ArcFace GMDB comparison complete','Unadapted ArcFace GMDB comparison completed',
        'The original Glint360K iResNet-100 baseline, original DINOv3 and craniofacial encoder have aggregate GMDB comparison results on HPC. No individual records or images are included. H100 pretraining continues independently.')

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise

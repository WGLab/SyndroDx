"""Reuse the previously validated GMDB metric driver without changing splits."""
import ast
import csv
import json
import sys
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent
TRAIN=WORK.parent/'transface_vitl_cpt_20261026'
SHARED=WORK.parent/'gmdb_checkpoint_eval_20261026'
sys.path[:0]=[str(TRAIN),str(SHARED)]
from control import load,save,digest,now
from metrics import evaluate,TOP_K

def main():
    p=load(WORK/'protocol.json')
    for name,key in [('evaluate_checkpoint.py','metric_driver_sha256'),('metrics.py','metrics_sha256'),('inputs.private.jsonl','inputs_sha256')]:
        if digest(SHARED/name)!=p[key]:raise RuntimeError('frozen_evaluation_source_changed')
    source=SHARED/'evaluate_checkpoint.py';parsed=ast.parse(source.read_text())
    function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run_metrics')
    scope=dict(np=np,evaluate=evaluate,TOP_K=TOP_K)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
    with (SHARED/'inputs.private.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    tables={}
    for name,sha in p['split_sha256'].items():
        path=SHARED/'split_snapshot'/(name+'.csv')
        if digest(path)!=sha:raise RuntimeError('frozen_gmdb_split_changed')
        with path.open(newline='') as stream:tables[name]=list(csv.DictReader(stream))
    models={}
    for kind in ('original','cpt4000'):
        full=np.empty((len(rows),512),np.float32)
        for rank in range(2):
            record=load(WORK/'progress'/f'{kind}_{rank}.json');path=WORK/'private'/f'{kind}_{rank}.npy'
            if record.get('state')!='complete' or digest(path)!=record.get('features_sha256'):raise RuntimeError('incomplete_feature_shard')
            shard=np.load(path,allow_pickle=False)
            if shard.shape!=full[rank::2].shape:raise RuntimeError('shard_shape_mismatch')
            full[rank::2]=shard
        np.save(WORK/'private'/f'{kind}_features.npy',full)
        if not np.isfinite(full).all() or not np.allclose(np.linalg.norm(full,axis=1),1.,atol=1e-5):
            raise RuntimeError('invalid_normalized_features')
        models[kind]=scope['run_metrics'](kind,full,rows,tables)
        save(WORK/'status.json',dict(state='scoring',completed_model=kind,updated_at_utc=now()))
    differences={variant:{case:{key:models['cpt4000'][variant][case]['macro'][key]-value
        for key,value in score['macro'].items()} for case,score in cases.items()} for variant,cases in models['original'].items()}
    save(WORK/'results.json',dict(state='complete',checkpoint_step=4000,models=models,cpt_minus_original=differences,
        gmdb_finetuned=False,encoder_frozen=True,precision='BF16 for both models',single_model=True,tta=False,
        exploratory=True,not_an_untouched_final_test=True,confidence_intervals_computed=False,updated_at_utc=now()))
    save(WORK/'status.json',dict(state='complete',updated_at_utc=now()))

if __name__=='__main__':main()

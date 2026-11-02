"""Same queries, galleries and disease macro scoring for every comparator."""
import ast,csv,importlib.util,numpy as np
from common import *

def paired_patient_interval(a,b,rows,tables):
    """Supplementary disease-macro patient means; paired disease-stratified bootstrap."""
    index={r['image_id']:i for i,r in enumerate(rows)};results={}
    rng=np.random.default_rng(20261028)
    for cohort in ['frequent','rare']:
        outcomes={}
        for fold in ([None] if cohort=='frequent' else range(10)):
            gallery=[r for r in tables['gmdb_'+cohort+'_gallery_images'] if fold is None or int(r['split'])==fold]
            gp={r['subject'] for r in gallery}
            queries=[r for r in tables['gmdb_'+cohort+'_test_images'] if (fold is None or int(r['split'])==fold) and r['subject'] not in gp]
            gallery_labels=np.asarray([r['label'] for r in gallery]);hits=[]
            for features in [a,b]:
                g=features[[index[r['image_id']] for r in gallery]];q=features[[index[r['image_id']] for r in queries]]
                rank=np.argsort(-(q@g.T),axis=1);values=[]
                for r,order in zip(queries,rank):
                    ordered=gallery_labels[order];ordered=ordered[np.sort(np.unique(ordered,return_index=True)[1])]
                    values.append([float(r['label'] in ordered[:k]) for k in [1,5,10]])
                hits.append(np.asarray(values))
            for i,r in enumerate(queries):outcomes.setdefault((r['label'],r['subject']),[]).append(hits[1][i]-hits[0][i])
        bylabel={}
        for (label,patient),values in outcomes.items():bylabel.setdefault(label,[]).append(np.mean(values,axis=0))
        point=np.mean([np.mean(v,axis=0) for v in bylabel.values()],axis=0)
        samples=np.zeros((2000,3))
        for values in bylabel.values():
            values=np.asarray(values);draw=rng.integers(len(values),size=(2000,len(values)));samples+=values[draw].mean(1)/len(bylabel)
        results[cohort]={f'top{k}':dict(transface_minus_arc_patient_macro=float(point[j]),paired_bootstrap_95_interval=np.quantile(samples[:,j],[.025,.975]).tolist()) for j,k in enumerate([1,5,10])}
    return dict(metric='supplementary disease-macro of patient means; not image-macro primary point estimate',repetitions=2000,results=results)

def main():
    ref=load(SHARED/'protocol.json');tables={}
    for name,sha in ref['split_sha256'].items():
        p=SHARED/'split_snapshot'/(name+'.csv')
        if digest(p)!=sha:raise RuntimeError('split_modified')
        with p.open() as f:tables[name]=list(csv.DictReader(f))
    with (SHARED/'inputs.private.jsonl').open() as f:rows=[__import__('json').loads(s) for s in f]
    spec=importlib.util.spec_from_file_location('fixed_metric',SHARED/'metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fn=next(n for n in ast.parse((SHARED/'evaluate_checkpoint.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_metrics')
    scope=dict(np=np,evaluate=m.evaluate,TOP_K=m.TOP_K);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SHARED/'evaluate_checkpoint.py'),'exec'),scope)
    data={}
    for arm in ['r50','r100','unadapted','transface']:
        receipt=load(WORK/'evaluation'/arm/'receipt.json');p=WORK/'evaluation'/arm/'features.private.npy'
        if arm=='r50' and receipt.get('state')!='complete':continue
        if receipt.get('state')!='complete' or digest(p)!=receipt['features_sha256']:raise RuntimeError('incomplete_test_features')
        data[arm]=np.load(p,mmap_mode='r')
    models={}
    for arm in data:
        models[arm+'_single']=scope['run_metrics'](arm,data[arm][:,0],rows,tables)
        models[arm+'_TTA']=scope['run_metrics'](arm,np.asarray(data[arm]).reshape(len(rows),-1)/2,rows,tables)
    if 'r50' in data:
        ensemble=np.concatenate([data[arm] for arm in ['r50','r100','unadapted']],axis=1).reshape(len(rows),-1)/np.sqrt(12)
        models['GestaltMatcher_Arc_local_ensemble_TTA']=scope['run_metrics']('GestaltMatcher_Arc_local_ensemble_TTA',ensemble,rows,tables)
    reference=ensemble if 'r50' in data else np.asarray(data['r100']).reshape(len(rows),-1)/2
    ours=np.asarray(data['transface']).reshape(len(rows),-1)/2
    uncertainty=paired_patient_interval(reference,ours,rows,tables)
    save(WORK/'results.json',dict(state='complete' if 'r50' in data else 'partial_R100_comparison_complete_R50_pending',dataset='GMDB v1.1.0',models=models,
        primary_protocol='patient_disjoint_query_supplement',training_patient_overlap_with_test=0,
        source='local controlled reference-based reproduction; not author-supplied GMDB-finetuned weights',
        multimodal_results_not_mixed_with_image_retrieval=True,clinical_holdout_used=False,
        known_deviations=['shared conservative crop/jitter and fixed convolutional batchnorm; development checkpoint selection rather than author last checkpoint'],
        paired_patient_bootstrap=uncertainty,confidence_intervals_computed=True,exploratory_GMDB_previously_examined=True,updated_at_utc=now()))

if __name__=='__main__':main()

"""Fixed-epoch, patient-isolated retrieval/few-shot for clean R50 training."""
import os,sys,importlib.util,shutil,json
from common import *

def worker(case):
    import numpy as np,torch,cv2
    from torch.utils.data import DataLoader,Dataset
    from network import R50
    arm=case.rsplit('_',1)[0] if case!='pretrained' else 'code_keep_bn'
    device=os.environ.get('R50_EVAL_DEVICE','cuda');torch.set_num_threads(4)
    receipt=load(WORK/'data_receipt.json');m=R50(receipt['diseases'],arm).to(device).eval().requires_grad_(False)
    if case!='pretrained':
        s=torch.load(WORK/case/'last.pt',map_location='cpu',weights_only=False);assert s['epoch']==50;m.load_state_dict(s['model'],strict=True)
    rows=[json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()];pin=load(WORK/'input_pins.json');assert digest(SHARED/'inputs.private.jsonl')==pin['inputs']
    class Inputs(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            if digest(r['crop_path'])!=r['crop_sha256']:raise RuntimeError('image_integrity_changed')
            x=cv2.imread(r['crop_path']);
            if x is None:raise RuntimeError('local_decode_failed')
            x=cv2.resize(x,(112,112)).transpose(2,0,1).copy()
            return torch.from_numpy(x).float().div_(127.5).sub_(1)
    torch.set_num_threads(4);cv2.setNumThreads(1);out=[]
    with torch.inference_mode():
        for x in DataLoader(Inputs(),batch_size=32 if device=='cpu' else 256,num_workers=2,pin_memory=device=='cuda'):
            y=m.embed(x.to(device,non_blocking=True))
            if not torch.isfinite(y).all():raise RuntimeError('invalid_embeddings')
            out.append(y.cpu().numpy())
            save(WORK/(case+'_inference_progress.json'),dict(state='extracting',device=device,completed=sum(len(v) for v in out),total=len(rows),updated_at_utc=now()))
    path=WORK/'private'/f'{case}_features.npy';np.save(path,np.concatenate(out));save(WORK/(case+'_inference.json'),dict(state='complete',sha256=digest(path),updated_at_utc=now()))

def score():
    import numpy as np
    p=load(WORK/'protocol.json');all_results={}
    baseline=WORK/'private/pretrained_features.npy'
    shutil.copy2(baseline,WORK/'private/original.npy')
    arms=['code_keep_bn'] if os.environ.get('R50_PRIMARY_ONLY')=='1' else p['arms']
    for arm in arms:
        all_results[arm]={}
        for seed in p['seeds']:
            case=arm+'_'+str(seed);shutil.copy2(WORK/'private'/f'{case}_features.npy',WORK/'private/selected.npy')
            save(WORK/'selection.json',dict(step=50,checkpoint=str(WORK/case/'last.pt'),selection='fixed epoch50 before test',arm=arm,seed=seed))
            spec=importlib.util.spec_from_file_location('clean_r50_metrics_'+str(seed),WORK/'metric_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.evaluate()
            result=load(WORK/'results.json');all_results[arm][str(seed)]=result['models']['selected'];save(WORK/(case+'_test.json'),result)
    summary={}
    for arm,cases in all_results.items():
        summary[arm]={task:{f'{metric}_mean':float(np.mean([v[task][metric] for v in cases.values()])) for metric in ['top1','top5','top10']} for task in ['frequent','rare']}
    report=dict(state='complete',protocol=p,split_audit=load(WORK/'split_audit.json'),models=all_results,summary=summary,
        pretrained_baseline=load(WORK/(p['arms'][0]+'_'+str(p['seeds'][0])+'_test.json'))['models']['original'],
        not_untouched_final_test=True,near_duplicate_identity_audit_complete=False,updated_at_utc=now())
    save(WORK/('primary_results.json' if os.environ.get('R50_PRIMARY_ONLY')=='1' else 'results.json'),report)

if __name__=='__main__':
    os.umask(0o077)
    if sys.argv[1]=='score':score()
    else:worker(sys.argv[1])

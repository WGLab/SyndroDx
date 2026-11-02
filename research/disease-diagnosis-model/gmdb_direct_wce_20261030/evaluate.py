"""Fixed test protocol after development-only selection."""
import sys,os,importlib.util,shutil
from common import *
sys.path.insert(0,str(PROJECT/'transface_objective_comparison_20261028'))
from matched_runtime import ID

def worker(arm):
    import torch,numpy as np,json
    from torch.utils.data import DataLoader
    from model import backbone
    spec=importlib.util.spec_from_file_location('fixed_image_decoder',PROJECT/'transface_gmdb_diagnostic_20261026/worker.py')
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    selected=load(WORK/'selected.json')[arm];assert digest(selected['path'])==selected['sha256']
    m=backbone();m.encoder.feature[-1]=torch.nn.Identity();m.load_state_dict(torch.load(selected['path'],map_location='cpu',weights_only=False)['backbone'],strict=True);m=m.cuda().eval().requires_grad_(False)
    rows=[json.loads(l) for l in (SHARED/'inputs.private.jsonl').open()];torch.set_num_threads(4);features=[]
    with torch.inference_mode():
        for x in DataLoader(decoder.Images(rows,list(range(len(rows)))),batch_size=256,num_workers=2,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):v=m(x.cuda(non_blocking=True))
            features.append(torch.nn.functional.normalize(v.float(),dim=-1).cpu().numpy())
    np.save(WORK/'private'/('original.npy' if arm=='original' else 'selected.npy'),np.concatenate(features))

def score():
    for name in ['selection.json','input_pins.json']:shutil.copy2(PROJECT/'arcface_transfer_evaluation_20261028'/name,WORK/name)
    spec=importlib.util.spec_from_file_location('fixed_metrics',WORK/'metric_driver.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.evaluate()
    r=load(WORK/'results.json');r['models_description']={'original':'original public TransFace directly fine-tuned with frozen-body WCE','selected':'CHOP identity CPT step1000 directly fine-tuned with frozen-body WCE'}
    r['development_selection']=load(WORK/'selected.json');r['R50_reference']=load(PROJECT/'gestaltmatcher_hf_evaluation_20261029/results.json').get('models',{}).get('selected')
    r['limitations']=['Repeated GMDB test, exploratory.','R50 reproduction training manifest unavailable; cross-version training overlap unresolved.','Known CHOP/BCH sources excluded; full cross-source identity audit pending.']
    save(WORK/'results.json',r)

if __name__=='__main__':
    os.umask(0o077)
    if sys.argv[1]=='score':score()
    else:worker(sys.argv[1])

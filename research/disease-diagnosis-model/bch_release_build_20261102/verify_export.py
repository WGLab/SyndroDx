"""Synthetic-only numerical parity against the original locked implementation."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import gc
import numpy as np
import torch
from torch.nn import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'model_releases/bch_evaluation_v1_20261102'
sys.path.insert(0,str(BUNDLE))
from predict import Predictor, read_cases, retrieval, image_tensor


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    pred = Predictor('cpu')
    x = torch.linspace(-1,1,2*3*112*112).reshape(2,3,112,112)
    with torch.inference_mode():
        exported = pred.visual(x).numpy()
    identity = ROOT/'transface_arcface_identity_cpt_20261026'
    sys.path[:0] = [str(identity),str(identity/'dependencies')]
    spec=importlib.util.spec_from_file_location('original_visual_author',identity/'model.py')
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    original=author.author_backbone(optimized=True).eval()
    state=torch.load(BUNDLE/'weights/visual.pt',map_location='cpu',weights_only=True)
    original.load_state_dict({(k.removeprefix('body.') if k.startswith('body.') else k):v for k,v in state.items()},strict=True)
    with torch.inference_mode():
        reference=F.normalize(original.feature(original.forward_features(x)[0]).float(),dim=-1).numpy()
    error=float(np.max(np.abs(exported-reference)))
    assert error < 1e-6, 'visual_parity_failed'
    del original,state
    gc.collect()
    sys.path.insert(0,str(ROOT/'clinical_evidence_agent_20261101'))
    from frozen_v1 import FrozenV1
    adapter=FrozenV1('cpu')
    synthetic={'hpo_present':['HP:0000316'],'hpo_absent':['HP:0001250'],'hpo_unknown':[]}
    ids,states,_=pred.pack(synthetic)
    assert ids.any()
    checks=0;maxerror=0.
    for with_face in [False,True]:
        f=exported[0] if with_face else np.zeros(512,np.float32)
        for with_hpo in [False,True]:
            i=ids if with_hpo else np.zeros_like(ids)
            s=states if with_hpo else np.zeros_like(states)
            for mode in ['face_only','hpo_only','face_hpo']:
                expected,routes=adapter.probabilities(f[None],i[None],s[None],mode)
                actual,route=pred.probabilities(f,i,s,mode)
                assert route==routes[0]
                if actual is None:assert not expected.any()
                else:
                    diff=float(np.max(np.abs(expected[0]-actual)));maxerror=max(maxerror,diff)
                    assert diff < 1e-6
                checks+=1
    with torch.inference_mode():
        _,z=adapter.models['hpo_11'](torch.zeros(1,512),torch.from_numpy(ids[None]),torch.from_numpy(states[None]))
        expected=F.normalize(z.float(),dim=-1)[0].numpy()
    hpoerror=float(np.max(np.abs(pred.hpo_feature(ids,states)-expected)))
    assert hpoerror < 1e-6
    # Unknown terms must not change the eligible input.
    u,_,q=pred.pack({'hpo_unknown':['HP:0000316']});assert not u.any() and q['unknown_terms_ignored']==1
    _,_,q=pred.pack({'hpo_present':['HP:9999999']});assert q['out_of_vocabulary_terms']==['HP:9999999']
    try:pred.pack({'hpo_present':list(pred.terms)[:129]})
    except ValueError:pass
    else:raise AssertionError('HPO_overflow_not_rejected')
    with tempfile.TemporaryDirectory() as tmp:
        path=pathlib.Path(tmp)/'cases.jsonl'
        path.write_text(json.dumps({'case_id':'synthetic','hpo_present':['HP:0000316'],'hpo_absent':['HP:0000316']})+'\n')
        try:read_cases(path)
        except ValueError:pass
        else:raise AssertionError('polarity_conflict_not_rejected')
        import cv2
        pixels=np.zeros((112,112,3),np.uint8);pixels[:,:,2]=255
        cv2.imwrite(str(pathlib.Path(tmp)/'red.png'),pixels)
        tensor=image_tensor(pathlib.Path(tmp)/'red.png')
        assert tensor[0].eq(1).all() and tensor[2].eq(-1).all()
    v=np.zeros(512,np.float32);v[0]=1
    h=np.zeros(256,np.float32);h[0]=1
    query={'case_id':'q','visual':v,'hpo':h}
    gallery=[query,{'case_id':'g','visual':v,'hpo':-h},{'case_id':'missing_hpo','visual':v,'hpo':np.zeros_like(h)}]
    r=retrieval(query,gallery,10)
    assert [x['gallery_case_id'] for x in r]==['missing_hpo','g']
    assert r[0]['similarity']==1. and r[1]['similarity']==0.
    receipt={'state':'passed','synthetic_inputs_only':True,'patient_records_or_images_accessed':False,
        'visual_max_abs_error':error,'routing_cases_checked':checks,'diagnosis_max_abs_error':maxerror,
        'HPO_retrieval_embedding_max_abs_error':hpoerror,'RGB_normalization_checked':True,
        'unknown_HPO_exclusion_checked':True,'conflicting_HPO_rejected':True,'HPO_limit_checked':True,
        'retrieval_self_exclusion_and_pairwise_fallback_checked':True,'device':'CPU','precision':'FP32'}
    (ROOT/'bch_release_build_20261102/verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()

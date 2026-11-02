"""Run the public command from a relocated copy with synthetic local images."""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import cv2
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'model_releases/bch_evaluation_v1_20261102'

with tempfile.TemporaryDirectory(prefix='bch-portable-') as tmp:
    work=pathlib.Path(tmp)
    bundle=work/'renamed_release'
    shutil.copytree(BUNDLE,bundle,ignore=shutil.ignore_patterns('__pycache__'))
    data=work/'synthetic';data.mkdir()
    pixels=np.arange(112*112*3,dtype=np.uint32).reshape(112,112,3).astype(np.uint8)
    cv2.imwrite(str(data/'pattern.png'),pixels)
    cv2.imwrite(str(data/'flipped_pattern.png'),pixels[:,::-1].copy())
    cases=[
        {'case_id':'SYNTHETIC_BOTH','images':['pattern.png','flipped_pattern.png'],'hpo_present':['HP:0000316'],'hpo_absent':['HP:0001250']},
        {'case_id':'SYNTHETIC_HPO','images':[],'hpo_present':['HP:0000316']},
        {'case_id':'SYNTHETIC_FACE','images':['pattern.png'],'hpo_unknown':['HP:0001250']},
        {'case_id':'SYNTHETIC_EMPTY','images':[]},
    ]
    gallery=[cases[0],{'case_id':'SYNTHETIC_GALLERY','images':['pattern.png']}]
    for name,rows in [('cases',cases),('gallery',gallery)]:
        (data/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN'}}
    env.update(PYTHON_BIN=sys.executable,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    command=['bash',str(bundle/'run.sh'),'--input',str(data/'cases.jsonl'),'--gallery',str(data/'gallery.jsonl'),
        '--output',str(work/'output'),'--device','cpu','--threads','2','--batch-size','2']
    result=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=600)
    assert result.returncode==0, 'CLI_smoke_failed_'+result.stderr[-100:]
    summary=json.loads((work/'output/summary.json').read_text())
    assert summary['state']=='complete' and summary['patients']==4
    rows=[json.loads(l) for l in (work/'output/predictions.jsonl').read_text().splitlines()]
    assert [r['predictions']['face_hpo']['route'] for r in rows]==['face_hpo','hpo_only','face_only','insufficient_input']
    assert rows[0]['image_count']==2
    assert rows[0]['retrieval'][0]['gallery_case_id']=='SYNTHETIC_GALLERY'
    assert not rows[1]['retrieval']
    for r in rows:
        for pred in r['predictions'].values():
            if pred['route']=='insufficient_input':assert not pred['ranking'] and pred['scores_in_diseases_json_order'] is None
            else:
                assert len(pred['ranking'])==10
                assert len(pred['scores_in_diseases_json_order'])==216
                assert abs(sum(pred['scores_in_diseases_json_order'])-1)<1e-5
    assert rows[2]['predictions']['face_only']['scores_in_diseases_json_order']==rows[2]['predictions']['face_hpo']['scores_in_diseases_json_order']
    with np.load(work/'output/features.npz') as f:
        assert f['visual'].shape==(4,512) and f['hpo'].shape==(4,256)
        assert not f['visual'][1].any() and not f['hpo'][2].any()
    repeat=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=60)
    assert repeat.returncode!=0 and 'output_directory_already_exists' in repeat.stderr
    receipt={'state':'passed','relocated_bundle':True,'external_working_directory':True,
        'synthetic_patients':4,'synthetic_gallery_cases':2,'patient_data_used':False,
        'multi_image_patient_checked':True,'three_modality_outputs_checked':True,
        'missing_modality_routes_checked':True,'retrieval_checked':True,'overwrite_rejected':True,
        'device':'CPU','python':sys.version.split()[0],
        'torch':summary['torch_version'],'numpy':summary['numpy_version'],'opencv':summary['opencv_version'],
        'fresh_dependency_install_tested':False,'GPU_execution_tested':False}
    (ROOT/'bch_release_build_20261102/smoke.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

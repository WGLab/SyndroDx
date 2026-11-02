import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from metrics import evaluate,TOP_K

class MetricTests(unittest.TestCase):
    def test_matches_unmodified_official_syndrome_macro_function(self):
        source=Path(__file__).parent/'official_source/evaluate_ensemble.py'
        parsed=ast.parse(source.read_text());fn=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='eval')
        namespace={'np':np,'pairwise_distances':pairwise_distances}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        rng=np.random.default_rng(31);features=rng.normal(size=(36,12)).astype(np.float32)
        index={str(i):i for i in range(36)}
        gallery=[dict(image_id=str(i),subject='g'+str(i),label=str(i//3)) for i in range(24)]
        queries=[dict(image_id=str(i),subject='q'+str(i),label=str((i-24)%8)) for i in range(24,36)]
        frame=pd.DataFrame(gallery);frame['synd_id']=[r['label'] for r in gallery]
        reference=namespace['eval'](frame,features[:24,None,:],features[24:,None,:],np.array([r['label'] for r in queries]))
        result=evaluate(gallery,queries,features,index)
        np.testing.assert_allclose([result['macro'][f'top{k}'] for k in TOP_K],reference,atol=1e-14,rtol=0)
    def test_uncovered_disease_stays_in_denominator(self):
        f=np.eye(3,dtype=np.float32);index={str(i):i for i in range(3)}
        g=[dict(image_id='0',subject='a',label='disease1')]
        q=[dict(image_id='1',subject='b',label='disease1'),dict(image_id='2',subject='c',label='uncovered')]
        r=evaluate(g,q,f,index);self.assertEqual(r['macro']['top1'],.5);self.assertEqual(r['gallery_coverage'],.5)
    def test_patient_disjoint_supplement_excludes_self_match_queries(self):
        f=np.eye(2,dtype=np.float32);index={'0':0,'1':1}
        g=[dict(image_id='0',subject='shared',label='a')]
        q=[dict(image_id='0',subject='shared',label='a'),dict(image_id='1',subject='new',label='a')]
        self.assertEqual(evaluate(g,q,f,index)['query_images'],2)
        self.assertEqual(evaluate(g,q,f,index,strict=True)['query_images'],1)
    def test_gallery_diseases_rank_by_best_image_not_duplicate_votes(self):
        f=np.array([[1.,0.],[.99,.01],[0.,1.],[1.,0.]],dtype=np.float32)
        g=[dict(image_id=str(i),subject=str(i),label='a' if i<2 else 'b') for i in range(3)]
        q=[dict(image_id='3',subject='q',label='b')]
        r=evaluate(g,q,f,{str(i):i for i in range(4)})
        self.assertEqual(r['macro']['top1'],0);self.assertEqual(r['macro']['top5'],1)

if __name__=='__main__':unittest.main()

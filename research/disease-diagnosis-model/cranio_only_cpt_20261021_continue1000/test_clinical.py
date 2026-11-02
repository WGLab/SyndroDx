"""Synthetic-only data-boundary, resource and prospective-guard checks."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from configuration import WORLD,ARMS,SEEDS,configuration,STEPS,BACKBONE_LR_RATIO
from allocation_pool import select_pools
from clinical_dataset import ClinicalFaces
from guard import decide,KEYS
from omegaconf import OmegaConf
from dinov3.train import train as official


def intervals(lo=0.,hi=0.):
    return {c:{m:{'paired_patient_bootstrap_95_interval':[lo,hi]} for m in ('ridge','retrieval')}
            for c in ('disease','structural')}


class GuardTests(unittest.TestCase):
    def setUp(self):self.base=dict.fromkeys(KEYS,.5)
    def test_equal_and_improved_pass(self):
        self.assertTrue(decide(self.base,self.base,comparisons=intervals())['continue_training'])
        self.assertTrue(decide(dict.fromkeys(KEYS,.51),self.base,comparisons=intervals(0.,.02))['continue_training'])
    def test_small_uncertain_change_warns(self):
        r=decide(dict(self.base,structural_retrieval_top1=.498),self.base,comparisons=intervals(-.01,0.))
        self.assertTrue(r['continue_training']);self.assertEqual(r['state'],'warning_validation_change')
    def test_large_supported_decline_stops(self):
        self.assertFalse(decide(dict(self.base,structural_ridge_top1=.47),self.base,comparisons=intervals(-.06,-.001))['continue_training'])
    def test_large_uncertain_decline_warns(self):
        self.assertTrue(decide(dict(self.base,structural_ridge_top1=.47),self.base,comparisons=intervals(-.1,.01))['continue_training'])
    def test_severe_drop_stops_even_with_wide_interval(self):
        self.assertFalse(decide(dict(self.base,structural_ridge_top1=.39),self.base,comparisons=intervals(-.3,.01))['continue_training'])
    def test_persistent_supported_small_decline_stops(self):
        key='structural_ridge_top1';values=dict(self.base,**{key:.495})
        first=decide(values,self.base,comparisons=intervals(-.01,-.001))
        self.assertTrue(first['continue_training'])
        self.assertFalse(decide(values,self.base,comparisons=intervals(-.01,-.001),previous_supported=first['supported_declines'])['continue_training'])
    def test_invalid_metrics_or_intervals_fail_closed(self):
        for v in (None,float('nan'),True,-.1,1.1):
            self.assertFalse(decide(dict(self.base,structural_ridge_top1=v),self.base,comparisons=intervals())['continue_training'])
        for c in (None,{},intervals(float('nan'),0.)):
            self.assertFalse(decide(self.base,self.base,comparisons=c)['continue_training'])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'clinical.tsv'
        self.rows=[dict(training_admitted='true',split='train',view=v,group_id='synthetic_group_'+str(i),crop_path='synthetic_image_'+str(i)+'_'+v) for i,v in enumerate(('frontal','profile','oblique'))]
        self.release=dict(state='ssl_data_ready',patient_split_disjoint=True,protected_pools_excluded=True,
                          manifests={'train':dict(path=str(self.path),sha256='fixture')})
    def tearDown(self):self.tmp.cleanup()
    def dataset(self):
        with self.path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]),delimiter='\t');w.writeheader();w.writerows(self.rows)
        with patch('clinical_dataset.load',return_value=self.release),patch('clinical_dataset.digest',return_value='fixture'):
            return ClinicalFaces(None)
    def test_only_clinical_manifest_required_and_all_indices_use_it(self):
        d=self.dataset();allowed={r['crop_path'] for r in self.rows}
        self.assertEqual(d.image_count,3)
        self.assertTrue(all(d.selected_path(i) in allowed for i in range(1000)))
    def test_validation_or_nonadmitted_or_nonface_records_fail(self):
        for key,value in [('split','validation'),('training_admitted','false'),('view','nonface')]:
            old=self.rows[0][key];self.rows[0][key]=value
            with self.assertRaises(RuntimeError):self.dataset()
            self.rows[0][key]=old
    def test_release_guards_required(self):
        for key in ('patient_split_disjoint','protected_pools_excluded'):
            self.release[key]=False
            with self.assertRaises(RuntimeError):self.dataset()
            self.release[key]=True
    def test_hash_mismatch_fails(self):
        self.release['manifests']['train']['sha256']='different'
        with self.assertRaises(RuntimeError):self.dataset()


class ConfigurationTests(unittest.TestCase):
    def test_replicates_differ_only_in_seed_and_output(self):
        values=[OmegaConf.to_container(configuration(a),resolve=True) for a in ARMS]
        for a,v in zip(ARMS,values):
            self.assertEqual(v['train']['seed'],SEEDS[a]);del v['train']['seed'];del v['train']['output_dir']
        self.assertEqual(values[0],values[1])
    def test_same_global_batch_and_conservative_schedule(self):
        cfg=configuration(ARMS[0]);self.assertEqual(cfg.train.batch_size_per_gpu*WORLD,512)
        with patch.object(official.distributed,'get_world_size',return_value=WORLD):s=official.build_schedulers(cfg)
        self.assertEqual(len(s[0]),STEPS);self.assertAlmostEqual(max(s[0])*BACKBONE_LR_RATIO,1e-6)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.specs={};self.records={};self.steps=[]
        for ai,a in enumerate(ARMS):
            self.specs[a]=[]
            for rank in range(WORLD):
                j=str(100+ai*WORLD+rank);node='node'+str(ai)
                self.specs[a].append(dict(job_id=j,lookup_id=j,node=node,host_memory_gib=68,allowed_idle_steps={j+'.0':'sleep'}))
                self.records[j]=dict(JobId=j,UserId='research_user(1)',JobState='RUNNING',NodeList=node,Partition='gpu-xe9680q',AllocTRES='cpu=4,mem=72G,gres/gpu=1')
                self.steps.append((j+'.0','sleep'))
    def test_disjoint_pools_admitted(self):self.assertEqual([len(v) for v in select_pools(self.specs,self.records,self.steps).values()],[8,8])
    def test_duplicate_or_wrong_owner_rejected(self):
        bad=copy.deepcopy(self.specs);bad[ARMS[1]][0]['job_id']='100'
        with self.assertRaises(RuntimeError):select_pools(bad,self.records,self.steps)
        self.records['100']['UserId']='other(2)'
        with self.assertRaises(RuntimeError):select_pools(self.specs,self.records,self.steps)
    def test_unrecognized_compute_step_blocks_launch(self):
        self.assertEqual(len(select_pools(self.specs,self.records,self.steps+[('100.9','python')])[ARMS[0]]),7)
    def test_only_exact_own_coordinator_step_can_share_its_lease(self):
        with patch.dict('os.environ',{'SLURM_JOB_ID':'100','SLURM_STEP_ID':'9'}):
            self.assertEqual(len(select_pools(self.specs,self.records,self.steps+[('100.9','cranio-only-coordinator')])[ARMS[0]]),8)
            self.assertEqual(len(select_pools(self.specs,self.records,self.steps+[('100.8','cranio-only-coordinator')])[ARMS[0]]),7)
            self.assertEqual(len(select_pools(self.specs,self.records,self.steps+[('100.9','other')])[ARMS[0]]),7)


if __name__=='__main__':unittest.main()

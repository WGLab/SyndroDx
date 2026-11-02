import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
import numpy as np
import crop_worker as c
import prepare_crops_cpu as p

LABEL=dict(content='human_face_photo',face_count='one',completeness='complete',
           quality='usable',orientation='upright',context='ordinary',prominence='adequate',
           certainty='high',view='profile',face_bbox=[250,250,750,750])


class PreparationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'synthetic.png'
        Image.new('RGB',(800,600),'blue').save(source)
        row={'photo_id':'synthetic-a','patient_key':'synthetic-p','original_path':str(source),
             'view':'profile','width':'800','height':'600','face_bbox':json.dumps(LABEL['face_bbox'])}
        return row,source

    def register(self,root,key,item):
        (root/'preparation').mkdir(exist_ok=True)
        (root/'preparation_plan.json').write_text(json.dumps({'worker_count':1}))
        db=sqlite3.connect(root/'preparation/slot_00.sqlite')
        db.execute('CREATE TABLE preparations(key TEXT,state TEXT,record TEXT)')
        db.execute('INSERT INTO preparations VALUES(?,?,?)',(key,item['state'],json.dumps(item)))
        db.commit();db.close()

    def test_preparation_preserves_source_and_exact_crop_pixels_without_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,source=self.fixture(root);before=source.read_bytes()
            item=p.prepare(row,root,c)
            self.assertEqual(item['state'],'prepared_candidate')
            self.assertFalse(item['training_admitted'])
            self.assertFalse(item['GPU_review_complete'])
            self.assertEqual(source.read_bytes(),before)
            with Image.open(source) as original,Image.open(root/item['output_path']) as prepared:
                self.assertEqual(prepared.tobytes(),original.crop(item['crop_box']).tobytes())

    def test_missing_bbox_and_dimension_change_are_held(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,source=self.fixture(root)
            self.assertEqual(p.prepare(row|{'face_bbox':'null'},root,c)['state'],'requires_GPU_localization')
            self.assertEqual(p.prepare(row|{'width':'801'},root,c)['state'],'held_source_dimensions_changed')
            self.assertFalse((root/'prepared_native').exists())

    def test_matching_preparation_reused_only_after_GPU_verifier_accepts(self):
        class Detector:
            def detect(self,image):return np.empty((0,5)),None
        class Fake(c.LocalModels):
            def __init__(self):self.detector=Detector()
            def ask(self,images,prompt):
                return {k:'yes' for k in c.VALIDATION_KEYS}|{'view':'profile','certainty':'high'}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,source=self.fixture(root);key=c.output_key(row['photo_id'])
            item=p.prepare(row,root,c);pending=root/item['output_path'];self.register(root,key,item)
            accepted=Fake().process({'path':str(source),'photo_id':row['photo_id'],
                                    'patient_key':row['patient_key'],'width':800,'height':600},LABEL,root,key)
            self.assertEqual(accepted['status'],'validated_provisional')
            self.assertTrue(accepted['CPU_preparation_reused'])
            self.assertFalse(accepted['training_admitted'])
            self.assertFalse(pending.exists())
            self.assertTrue((root/accepted['output_path']).exists())
            self.assertEqual(accepted['output']['sha256'],item['output']['sha256'])

    def test_wrong_provenance_coordinates_or_corrupt_pixels_are_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,source=self.fixture(root);key=c.output_key(row['photo_id'])
            item=p.prepare(row,root,c);self.register(root,key,item)
            args=(root,key,row['patient_key'],item['source_sha256'],(800,600),item['crop_box'])
            self.assertIsNotNone(c.prepared_crop(*args))
            self.assertIsNone(c.prepared_crop(root,key,'other',item['source_sha256'],(800,600),item['crop_box']))
            self.assertIsNone(c.prepared_crop(root,key,row['patient_key'],'wrong-hash',(800,600),item['crop_box']))
            self.assertIsNone(c.prepared_crop(root,key,row['patient_key'],item['source_sha256'],(800,600),[0,0,800,600]))
            (root/item['output_path']).write_bytes(b'corrupted-synthetic-pixels')
            self.assertIsNone(c.prepared_crop(*args))


if __name__=='__main__':unittest.main(verbosity=2)

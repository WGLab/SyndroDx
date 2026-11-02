import fcntl
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import crop_worker as c

LABEL = dict(content='human_face_photo', face_count='one', completeness='complete',
             quality='usable', orientation='upright', context='ordinary',
             prominence='adequate', certainty='high', view='profile', face_bbox=[250,250,750,750])

class CropTests(unittest.TestCase):
    def test_bbox_rejects_invalid_coordinates(self):
        for box in (None, [0,0,0,100], [-1,0,100,100], [0,0,1001,1000],
                    [0,0,float('nan'),100], [0,0,True,100], [0,0,1]):
            self.assertIsNone(c.normalized_box(box, (800,600)))
        self.assertEqual(c.normalized_box([250,250,750,750], (800,600)), (200,150,600,450))

    def test_padding_keeps_native_coordinates_and_clamps(self):
        box = c.padded_box((200,150,600,450), (800,600), 'profile')
        self.assertEqual(box, (0,0,800,525))
        self.assertEqual(c.padded_box((0,0,800,600), (800,600), 'frontal'), (0,0,800,600))
        self.assertEqual(c.union_box((10,30,90,95), (5,40,110,100)), (5,30,110,100))

    def test_pixels_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            image = Image.frombytes('RGB', (11,9), bytes(i%256 for i in range(11*9*3)))
            image.info['comment'] = 'synthetic metadata must not survive'
            image.info['exif'] = b'not copied'
            box = (2,1,9,8)
            output = Path(d)/'crop.png'
            saved = c.save_crop(image, box, output)
            with Image.open(output) as crop:
                self.assertEqual(crop.tobytes(), image.crop(box).tobytes())
                self.assertEqual(crop.size, (7,7))
                self.assertNotIn('exif', crop.info)
                self.assertNotIn('comment', crop.info)
            self.assertEqual((saved['width'],saved['height']), (7,7))
            self.assertFalse(output.with_suffix('.tmp.png').exists())

    def test_validator_fails_closed(self):
        good = {k:'yes' for k in c.VALIDATION_KEYS} | {'view':'profile','certainty':'high'}
        self.assertTrue(c.validated(good, 'profile'))
        self.assertFalse(c.validated(good, 'frontal'))
        for key in c.VALIDATION_KEYS:
            for value in ('no','uncertain',True,None):
                self.assertFalse(c.validated(good | {key:value}, 'profile'))
        self.assertFalse(c.validated(good | {'certainty':'medium'}, 'profile'))
        self.assertFalse(c.validated(good | {'extra':'yes'}, 'profile'))
        self.assertEqual(c.parse_json('```json\n{"x":1}\n```'), {'x':1})
        for text in ('[]', '{bad}', '{"x":1} trailing', '```json\n{}'):
            with self.assertRaises(ValueError): c.parse_json(text)

    def fixture(self, root, shard, state, rows):
        folder = root/'direct_workers'
        folder.mkdir(exist_ok=True)
        (folder/f'shard_{shard:02d}.status.json').write_text(json.dumps({'state':state}))
        db = sqlite3.connect(folder/f'shard_{shard:02d}.sqlite')
        db.execute('CREATE TABLE labels(photo_id TEXT,patient_key TEXT,path TEXT,patient_known INTEGER,status TEXT,label TEXT,width INTEGER,height INTEGER,duplicate_of TEXT,raw TEXT,updated_at REAL)')
        for photo,known,status,duplicate,label in rows:
            db.execute('INSERT INTO labels VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (photo,'synthetic-patient','synthetic.png',known,status,json.dumps(label),800,600,duplicate,'',0))
        db.commit();db.close()

    def test_completed_review_gating_shard_ownership_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'records').mkdir()
            self.fixture(root, 0, 'completed', [('keep',1,'ok',None,LABEL),
                ('duplicate',1,'ok','keep',LABEL), ('unknown',0,'ok',None,LABEL),
                ('failed',1,'failed',None,LABEL), ('partial',1,'ok',None,LABEL|{'completeness':'partial'})])
            self.fixture(root, 2, 'qwen_patient_duplicate_review', [('not-ready',1,'ok',None,LABEL)])
            self.fixture(root, 1, 'completed', [('other-slot',1,'ok',None,LABEL)])
            db=c.open_records(root,0)
            chosen=list(c.candidates(root,0,db))
            self.assertEqual([row[2]['photo_id'] for row in chosen], ['keep'])
            c.record(db, chosen[0][1], 0, {'status':'hold'})
            self.assertEqual(list(c.candidates(root,0,db)), [])
            db.close()
            db=c.open_records(root,0)
            self.assertEqual(list(c.candidates(root,0,db)), [])
            with self.assertRaises(sqlite3.IntegrityError):
                c.record(db, chosen[0][1], 0, {'status':'hold'})
            db.close()
            other=c.open_records(root,1)
            self.assertEqual([row[2]['photo_id'] for row in c.candidates(root,1,other)], ['other-slot'])
            other.close()

    def test_exclusive_worker_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with (Path(d)/'lock').open('a') as first, (Path(d)/'lock').open('a') as second:
                fcntl.flock(first, fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_missing_detection_uses_existing_qwen_box_and_saves_native(self):
        import numpy as np
        class Detector:
            def detect(self, image): return np.empty((0,5)), None
        class Fake(c.LocalModels):
            def __init__(self): self.detector=Detector()
            def ask(self, images, prompt):
                return {k:'yes' for k in c.VALIDATION_KEYS}|{'view':'profile','certainty':'high'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'synthetic.png'
            Image.new('RGB',(800,600),'blue').save(source)
            before=source.read_bytes()
            row=dict(path=str(source),photo_id='synthetic',patient_key='synthetic-patient',width=800,height=600)
            item=Fake().process(row,LABEL,root,c.output_key('synthetic'))
            self.assertEqual(item['status'],'validated_provisional')
            self.assertEqual(item['locator'],'Qwen_annotation_bbox_fallback')
            self.assertFalse(item['training_admitted'])
            self.assertEqual(source.read_bytes(),before)
            self.assertTrue((root/item['output_path']).exists())

    def test_multiface_held_without_crop(self):
        import numpy as np
        class Detector:
            def detect(self, image): return np.array([[0,0,200,200,.99],[400,300,600,500,.99]]), None
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'synthetic.png'
            Image.new('RGB',(800,600)).save(source)
            model=object.__new__(c.LocalModels);model.detector=Detector()
            row=dict(path=str(source),photo_id='synthetic',patient_key='synthetic-patient',width=800,height=600)
            item=model.process(row,LABEL,root,'test')
            self.assertEqual(item['status'],'hold')
            self.assertFalse((root/'images_native').exists())

    def test_interrupted_pilot_resumes_from_durable_records(self):
        class FakeModels:
            def __init__(self, root): pass
            def ask(self, images, prompt): return {'single_face':'no'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'crop_pipeline';root.mkdir()
            for folder in ('records','pilot','workers'): (root/folder).mkdir()
            (root/'READY.json').write_text(json.dumps({'sha256':{}}))
            statuses=root.parent/'direct_workers';statuses.mkdir()
            for i in range(0,64,2):
                (statuses/f'shard_{i:02d}.status.json').write_text(json.dumps({'state':'completed'}))
            db=c.open_records(root,0)
            for view in c.VIEWS:
                for i in range(8):
                    c.record(db,f'{view}-{i}',0,{'status':'validated_provisional','pilot':True,
                             'patient_key':f'synthetic-{i}','view':view})
            db.close()
            # Simulate interruption after committing the last pilot crop but before its marker.
            self.assertFalse((root/'pilot/slot_00.json').exists())
            with patch.object(c,'ROOT',root),patch.object(c,'LocalModels',FakeModels),patch.object(c,'candidates',return_value=iter(())):
                self.assertEqual(c.serve(0),0)
            self.assertTrue(json.loads((root/'pilot/slot_00.json').read_text())['passed'])
            self.assertEqual(json.loads((root/'workers/slot_00.json').read_text())['state'],'finished_provisional')

if __name__ == '__main__': unittest.main(verbosity=2)

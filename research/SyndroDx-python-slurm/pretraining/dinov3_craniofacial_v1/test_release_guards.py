"""Clinical-free tests for the release boundary and independent QC schema."""
import csv,hashlib,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import release_data as release
from independent_qc import COUNT,parse,approved
from runtime import digest,save

class GuardTests(unittest.TestCase):
 def test_pseudopatient_links_transitive(self):
  g=release.Union();g.join('p:a','i:known');g.join('p:other_view','i:known');g.join('i:known','i:protected_alias')
  self.assertEqual(g.find('p:other_view'),g.find('i:protected_alias'))
 def test_qc_requires_boolean_and_expected_view(self):
  value={k:True for k in ['human_face_photo','single_face','upright','complete_visible_face','usable_quality','ordinary_photograph']};value.update(view='profile',certainty='high')
  self.assertTrue(approved(parse(json.dumps(value)),'profile'));self.assertFalse(approved(value,'frontal'))
  value['single_face']='true'
  with self.assertRaises(ValueError):parse(json.dumps(value))
 def test_release_excludes_protected_patient_and_detects_tampering(self):
  from PIL import Image
  with tempfile.TemporaryDirectory(prefix='synthetic_ssl_release_') as tmp:
   root=Path(tmp);run=root/'run';out=root/'dataset_release';qc=root/'independent_qc';run.mkdir();out.mkdir();qc.mkdir()
   metadata=root/'approved.tsv';metadata.write_text('synthetic metadata only\n')
   save(run/'crop_completion.json',{'state':'crop_complete'})
   identity={'state':'patient_identity_plan_ready','metadata_sha256':digest(metadata),'control_sha256':{},'blocked_hashes':[],'patients':{'allowed':{'group_id':'synthetic_group_0','split':'train','source':'approved_internal'},'protected':{'group_id':'synthetic_group_1','split':'hold','source':'approved_internal'}}}
   save(out/'patient_identity_plan.private.json',identity)
   fields=['photo_id','patient_key','crop_path','view','crop_sha256'];candidates=[]
   for i,patient in enumerate(['allowed','protected']):
    image=root/('abstract_'+str(i)+'.png');Image.new('RGB',(72,80),(30+i*20,40,50)).save(image)
    row=dict(photo_id=str(i),patient_key=patient,crop_path=str(image),view='frontal',crop_sha256=digest(image));candidates.append(row)
   with (run/'cropped_candidates.tsv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(candidates)
   for i in range(COUNT):
    save(qc/f'slot_{i:02d}.json',{'state':'finished'})
    db=sqlite3.connect(qc/f'slot_{i:02d}.sqlite');db.execute('CREATE TABLE reviews(record TEXT)')
    if i<2:
     v=dict(candidates[i],state='pass',source_sha256='source'+str(i),pixel_sha256='pixel'+str(i),integrity_passed=True)
     db.execute('INSERT INTO reviews VALUES (?)',(json.dumps(v),))
    db.commit();db.close()
   reference=root/'assets/fairface';reference.mkdir(parents=True)
   reference_image=root/'abstract_reference.png';Image.new('RGB',(80,80),(20,80,90)).save(reference_image)
   ref=reference/'reference_train.tsv'
   with ref.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['photo_id','patient_key','crop_path','source','view','crop_sha256','training_admitted','split'],delimiter='\t');w.writeheader();w.writerow(dict(photo_id='reference',patient_key='reference',crop_path=str(reference_image),source='fairface_reference',view='unspecified',crop_sha256=digest(reference_image),training_admitted='true',split='train'))
   save(reference/'status.json',{'state':'public_reference_ready','manifest':str(ref),'manifest_sha256':digest(ref)})
   with patch.multiple(release,ROOT=root,RUN=run,OUT=out,METADATA=metadata):
    self.assertEqual(release.final_release(),'ssl_data_ready')
    with (out/'train.tsv').open() as f:admitted=list(csv.DictReader(f,delimiter='\t'))
    self.assertEqual([r['patient_key'] for r in admitted],['allowed'])
    self.assertTrue(json.loads((out/'release.json').read_text())['protected_pools_excluded'])
    Path(candidates[0]['crop_path']).write_bytes(b'changed after review')
    with self.assertRaises(RuntimeError):release.final_release()

if __name__=='__main__':unittest.main()

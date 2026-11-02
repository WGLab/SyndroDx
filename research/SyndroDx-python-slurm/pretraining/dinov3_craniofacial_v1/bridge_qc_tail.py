"""Merge temporary reviewers on the original canonical review writer node."""
import argparse,csv,json,socket,sqlite3,time,hashlib
from runtime import ROOT,RUN,load,save,now
p=argparse.ArgumentParser();p.add_argument('--expected-host',required=True);a=p.parse_args();TAIL=ROOT/'qc_tail'
if socket.gethostname().split('.')[0]!=a.expected_host.split('.')[0]:raise RuntimeError('original_review_writer_node_required')
plan=load(TAIL/'plan.json');assignments=load(TAIL/'task_assignments.private.json')['assignments'];progress=load(TAIL/'bridge.json');watermarks=progress.get('watermarks',{});copied=progress.get('copied',0)
with (RUN/'accepted_faces.tsv').open(newline='') as f:identities={r['photo_id']:r['patient_key'] for r in csv.DictReader(f,delimiter='\t')}
db=sqlite3.connect(ROOT/'independent_qc/slot_01.sqlite',timeout=30)
try:
 while True:
  for slot in plan['extra_slots']:
   path=TAIL/f'slot_{slot:02d}.sqlite'
   if not path.exists():continue
   src=None
   try:
    src=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
    rows=src.execute('SELECT rowid,key,state,record,updated_at FROM reviews WHERE rowid>? ORDER BY rowid',(watermarks.get(str(slot),0),)).fetchall()
   except (sqlite3.Error,OSError):continue
   finally:
    if src is not None:src.close()
   for rowid,key,state,encoded,updated in rows:
    if assignments.get(key)!=slot or state not in ['pass','hold']:raise RuntimeError('extra_review_ownership_conflict')
    value=json.loads(encoded)
    if value.get('state')!=state or not value.get('photo_id') or not value.get('patient_key'):raise RuntimeError('extra_review_schema_invalid')
    expected_key=hashlib.sha256(('provisional-crop-v2'+'\0'+value['photo_id']).encode()).hexdigest()
    if expected_key!=key or identities.get(value['photo_id'])!=value['patient_key']:raise RuntimeError('extra_review_identity_conflict')
    db.execute('BEGIN IMMEDIATE');old=db.execute('SELECT record FROM reviews WHERE key=?',(key,)).fetchone()
    if old and json.loads(old[0])!=value:raise RuntimeError('extra_review_overwrite_refused')
    if not old:db.execute('INSERT INTO reviews VALUES (?,?,?,?)',(key,state,encoded,updated))
    db.commit();watermarks[str(slot)]=rowid;copied+=1
   save(TAIL/'bridge.json',{'state':'merging','copied':copied,'watermarks':watermarks,'updated_at_utc':now()})
  if all(load(TAIL/f'slot_{s:02d}.json').get('state')=='finished' for s in plan['extra_slots']):
   expected=sum(plan['task_counts'][str(s)] for s in plan['extra_slots'])
   if copied!=expected:raise RuntimeError('extra_review_merge_coverage_incomplete')
   save(TAIL/'bridge.json',{'state':'complete','copied':copied,'watermarks':watermarks,'updated_at_utc':now()});break
  time.sleep(3)
finally:db.close()

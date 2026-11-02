"""Copy helper results on the original shared-result writer node only."""
import argparse,fcntl,json,socket,sqlite3,time
from runtime import load,save,now
from crop_delegation import ACCEL
p=argparse.ArgumentParser();p.add_argument('--extra',type=int,required=True);p.add_argument('--expected-host',required=True);a=p.parse_args()
if socket.gethostname().split('.')[0]!=a.expected_host.split('.')[0]:raise RuntimeError('original_result_writer_node_required')
lock=(ACCEL/f'extra_{a.extra:02d}.bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=load(ACCEL/f'extra_{a.extra:02d}.transfer.json');target=receipt['source_slot']
with (ACCEL/f'worker_{a.extra:02d}.tasks.jsonl').open() as f:tasks={t['key']:t for t in map(json.loads,f)}
progress=load(ACCEL/f'extra_{a.extra:02d}.bridge.json');cursor=progress.get('cursor',0);copied=progress.get('copied',0)
db=sqlite3.connect(ACCEL/f'results_{target:02d}.sqlite',timeout=30)
try:
 while copied<len(tasks):
  path=ACCEL/f'results_{a.extra:02d}.sqlite';source=None
  try:
   source=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=3) if path.exists() else None
   rows=source.execute('SELECT sequence,key,owner,record,updated_at FROM results WHERE sequence>? ORDER BY sequence',(cursor,)).fetchall() if source else []
  except (sqlite3.Error,OSError):time.sleep(3);continue
  finally:
   if source is not None:source.close()
  for sequence,key,owner,encoded,updated in rows:
   task=tasks.get(key);value=json.loads(encoded)
   if not task or task['owner']!=owner or task['delegation_id']!=value.get('delegation_id') or task['row']['photo_id']!=value.get('photo_id') or task['row']['patient_key']!=value.get('patient_key'):raise RuntimeError('extra_result_identity_conflict')
   db.execute('BEGIN IMMEDIATE')
   previous=db.execute('SELECT record FROM results WHERE key=?',(key,)).fetchone()
   if previous and json.loads(previous[0])!=value:raise RuntimeError('extra_result_overwrite_refused')
   if not previous:db.execute('INSERT INTO results(key,owner,record,updated_at) VALUES (?,?,?,?)',(key,owner,encoded,updated))
   db.commit();cursor=sequence;copied+=1
   save(ACCEL/f'extra_{a.extra:02d}.bridge.json',{'state':'complete' if copied==len(tasks) else 'merging','cursor':cursor,'copied':copied,'total':len(tasks),'updated_at_utc':now()})
  if copied<len(tasks):time.sleep(5)
finally:db.close();lock.close()

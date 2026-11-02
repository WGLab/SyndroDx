"""Delegate future primary-crop work without interrupting the six legacy workers.

All clinical task manifests, labels, images and records remain local to HPC.
Only the original writer node updates each legacy WAL database.
"""
import csv,fcntl,hashlib,importlib.util,json,os,socket,sqlite3,time
from pathlib import Path
from runtime import ROOT,RUN,load,save,now
PIPELINE=RUN/'crop_pipeline'
ACCEL=ROOT/'crop_acceleration'
os.umask(0o077)

def crop_module():
 spec=importlib.util.spec_from_file_location('frozen_crop_worker_for_delegation',PIPELINE/'crop_worker.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def original_node(expected):
 if socket.gethostname().split('.')[0]!=expected.split('.')[0]:raise RuntimeError('legacy_database_write_requires_original_node')

def claim_owner(owner,expected_host):
 original_node(expected_host);ACCEL.mkdir(parents=True,exist_ok=True)
 plan=load(ACCEL/'plan.json');delegation=plan['delegation_id'];mod=crop_module()
 state=load(PIPELINE/'workers'/f'slot_{owner:02d}.json')
 current=state.get('shard')
 manifest=ACCEL/f'owner_{owner:02d}.tasks.jsonl'
 if manifest.exists() and load(ACCEL/f'owner_{owner:02d}.claimed.json').get('state')=='claimed':return
 with (RUN/'accepted_faces.tsv').open(newline='') as f:accepted={r['photo_id'] for r in csv.DictReader(f,delimiter='\t')}
 selected=[]
 db=sqlite3.connect(PIPELINE/'records'/f'slot_{owner:02d}.sqlite',timeout=30)
 try:
  if state.get('state')=='cropping' and isinstance(current,int):
   # Current shards remain entirely with their existing workers. Future shards
   # are claimed in a short same-node transaction before they are reached.
   for shard in range(owner,64,6):
    latest=load(PIPELINE/'workers'/f'slot_{owner:02d}.json').get('shard',64)
    if shard<=current or shard<=latest:continue
    src=sqlite3.connect((RUN/'direct_workers'/f'shard_{shard:02d}.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=5);src.row_factory=sqlite3.Row
    try:
     candidates=[]
     for row in src.execute("SELECT photo_id,patient_key,path,label,width,height FROM labels WHERE status='ok' AND patient_known=1 AND (duplicate_of IS NULL OR duplicate_of='') ORDER BY photo_id"):
      value=dict(row);label=json.loads(value['label'])
      if value['photo_id'] in accepted and mod.eligible(label):candidates.append((value,label))
    finally:src.close()
    db.execute('BEGIN IMMEDIATE')
    try:
     latest=load(PIPELINE/'workers'/f'slot_{owner:02d}.json').get('shard',64)
     if shard<=latest:db.rollback();continue
     for row,label in candidates:
      key=mod.output_key(row['photo_id'])
      if db.execute('SELECT 1 FROM records WHERE key=?',(key,)).fetchone():continue
      value={'photo_id':row['photo_id'],'patient_key':row['patient_key'],'view':label['view'],'status':'delegated_pending','delegation_id':delegation,'training_admitted':False,'pilot':False,'version':mod.VERSION}
      db.execute('INSERT INTO records VALUES (?,?,?,?,?)',(key,shard,'delegated_pending',json.dumps(value),time.time()))
      selected.append({'key':key,'owner':owner,'shard':shard,'row':row,'label':label,'delegation_id':delegation})
     db.commit()
    except BaseException:db.rollback();raise
  selected=[]
  pending=db.execute("SELECT key,shard,record FROM records WHERE status='delegated_pending'").fetchall()
  for shard in sorted({r[1] for r in pending}):
   src=sqlite3.connect((RUN/'direct_workers'/f'shard_{shard:02d}.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=5);src.row_factory=sqlite3.Row
   try:rows={r['photo_id']:dict(r) for r in src.execute('SELECT photo_id,patient_key,path,label,width,height FROM labels')}
   finally:src.close()
   for key,task_shard,encoded in pending:
    if task_shard!=shard:continue
    previous=json.loads(encoded)
    if previous.get('delegation_id')!=delegation:raise RuntimeError('unexpected_delegation_identity')
    row=rows.get(previous['photo_id'])
    if row is None:raise RuntimeError('delegated_source_missing')
    label=json.loads(row['label'])
    if row['patient_key']!=previous['patient_key'] or mod.output_key(row['photo_id'])!=key:raise RuntimeError('delegated_source_identity_changed')
    selected.append({'key':key,'owner':owner,'shard':shard,'row':row,'label':label,'delegation_id':delegation})
  # Durable manifest publication is followed by an explicit claim receipt.
  # No helper starts before every owner's claim receipt is present.
  temporary=manifest.with_suffix('.tmp')
  with temporary.open('w') as f:
   for v in selected:f.write(json.dumps(v)+'\n')
   f.flush();os.fsync(f.fileno())
  temporary.replace(manifest)
  save(ACCEL/f'owner_{owner:02d}.claimed.json',{'state':'claimed','owner':owner,'task_count':len(selected),'delegation_id':delegation,'completed_at_utc':now()})
 finally:db.close()

def distribute(worker_slots):
 plan=load(ACCEL/'plan.json');tasks=[]
 for owner in range(6):
  receipt=load(ACCEL/f'owner_{owner:02d}.claimed.json')
  if receipt.get('state')!='claimed':raise RuntimeError('all_claim_receipts_required')
  with (ACCEL/f'owner_{owner:02d}.tasks.jsonl').open() as f:values=[json.loads(line) for line in f if line.strip()]
  if len(values)!=receipt['task_count']:raise RuntimeError('claim_manifest_count_mismatch')
  tasks.extend(values)
 keys=[v['key'] for v in tasks]
 if len(keys)!=len(set(keys)):raise RuntimeError('duplicate_delegated_task_ownership')
 if not worker_slots:raise RuntimeError('accelerator_workers_required')
 for rank,slot in enumerate(worker_slots):
  path=ACCEL/f'worker_{slot:02d}.tasks.jsonl';temporary=path.with_suffix('.tmp')
  with temporary.open('w') as f:
   for task in tasks:
    if int(task['key'][:16],16)%len(worker_slots)==rank:f.write(json.dumps(task)+'\n')
  temporary.replace(path)
 plan.update(state='running',task_count=len(tasks),worker_slots=worker_slots,claims_sealed=True,updated_at_utc=now());save(ACCEL/'plan.json',plan)

class CropWork:
 def __init__(self,slot,mod):
  self.slot=slot;self.mod=mod
  with (ACCEL/f'worker_{slot:02d}.tasks.jsonl').open() as f:self.tasks=[json.loads(line) for line in f if line.strip()]
  self.db=sqlite3.connect(ACCEL/f'results_{slot:02d}.sqlite',timeout=30)
  self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS results (sequence INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT UNIQUE,owner INTEGER,record TEXT,updated_at REAL)')
  self.db.execute('CREATE INDEX IF NOT EXISTS results_owner_sequence ON results(owner,sequence)');self.db.commit()
  self.done={r[0] for r in self.db.execute('SELECT key FROM results')} & {t['key'] for t in self.tasks};self.cursor=0;self.failures=0
 def pending(self):return len(self.done)<len(self.tasks)
 def step(self,models,limit=8):
  completed=0
  while self.cursor<len(self.tasks) and completed<limit:
   task=self.tasks[self.cursor];self.cursor+=1
   if task['key'] in self.done:continue
   try:
    value=models.process(task['row'],task['label'],PIPELINE,task['key']);self.failures=0
   except Exception as e:
    import traceback
    with (ACCEL/f'worker_{self.slot:02d}.private_errors.log').open('a') as f:traceback.print_exc(file=f)
    value={'status':'error','reason':type(e).__name__,'photo_id':task['row']['photo_id'],'patient_key':task['row']['patient_key'],'view':task['label']['view'],'training_admitted':False,'version':self.mod.VERSION};self.failures+=1
    models.torch.cuda.empty_cache()
   value.update(pilot=False,delegation_id=task['delegation_id'],assisted_crop_worker=self.slot)
   self.db.execute('INSERT INTO results(key,owner,record,updated_at) VALUES (?,?,?,?)',(task['key'],task['owner'],json.dumps(value),time.time()));self.db.commit();self.done.add(task['key']);completed+=1
   save(ACCEL/f'worker_{self.slot:02d}.json',{'state':'assisting_primary_crop' if self.pending() else 'primary_crop_partition_complete','completed':len(self.done),'total':len(self.tasks),'updated_at_utc':now()})
   if self.failures>=3:raise RuntimeError('assisted_crop_repeated_errors')
  return completed

def merge_owner(owner,expected_host,once=False):
 original_node(expected_host);plan=load(ACCEL/'plan.json');delegation=plan['delegation_id']
 lock=(ACCEL/f'owner_{owner:02d}.merge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=load(ACCEL/f'owner_{owner:02d}.merged.json');watermarks=status.get('watermarks',{})
 db=sqlite3.connect(PIPELINE/'records'/f'slot_{owner:02d}.sqlite',timeout=30)
 try:
  while True:
   for slot in plan['worker_slots']:
    path=ACCEL/f'results_{slot:02d}.sqlite'
    if not path.exists():continue
    source=None
    try:
     source=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
     values=source.execute('SELECT sequence,key,record FROM results WHERE owner=? AND sequence>? ORDER BY sequence',(owner,watermarks.get(str(slot),0))).fetchall()
    except (sqlite3.Error,OSError):continue
    finally:
     if source is not None:source.close()
    for sequence,key,encoded in values:
     value=json.loads(encoded)
     if value.get('delegation_id')!=delegation or value.get('status') not in ['validated_provisional','hold','error']:raise RuntimeError('assisted_result_invalid')
     db.execute('BEGIN IMMEDIATE')
     try:
      original=db.execute('SELECT status,record FROM records WHERE key=?',(key,)).fetchone()
      if not original:raise RuntimeError('delegated_record_missing')
      previous=json.loads(original[1])
      if previous.get('delegation_id')!=delegation or previous['photo_id']!=value['photo_id'] or previous['patient_key']!=value['patient_key']:raise RuntimeError('delegation_identity_conflict')
      if original[0]=='delegated_pending':db.execute('UPDATE records SET status=?,record=?,updated_at=? WHERE key=?',(value['status'],encoded,time.time(),key))
      elif previous!=value:raise RuntimeError('delegated_record_overwrite_refused')
      db.commit()
     except BaseException:db.rollback();raise
     watermarks[str(slot)]=sequence
   remaining=db.execute("SELECT count(*) FROM records WHERE status='delegated_pending'").fetchone()[0]
   save(ACCEL/f'owner_{owner:02d}.merged.json',{'state':'complete' if remaining==0 else 'merging','owner':owner,'remaining':remaining,'watermarks':watermarks,'updated_at_utc':now()})
   if remaining==0 or once:return
   time.sleep(10)
 finally:
  db.close();lock.close()

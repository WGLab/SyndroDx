"""Atomically split uncompleted tasks at a durable shared-worker checkpoint."""
import argparse,fcntl,json,os,signal,socket,sqlite3,time
from pathlib import Path
from runtime import ROOT,load,save,now
from crop_delegation import ACCEL
def partition_pending(tasks,done):
 pending=[t for t in tasks if t['key'] not in done]
 moved=pending[len(pending)//2:];moved_keys={t['key'] for t in moved}
 return [t for t in tasks if t['key'] not in moved_keys],moved

p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);p.add_argument('--extra',type=int,required=True);p.add_argument('--expected-host',required=True);a=p.parse_args()
if socket.gethostname().split('.')[0]!=a.expected_host.split('.')[0]:raise RuntimeError('original_result_writer_node_required')
lock=(ACCEL/f'slot_{a.source:02d}.split.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=ACCEL/f'extra_{a.extra:02d}.transfer.json'
if load(receipt).get('state')=='sealed':raise SystemExit(0)
manifest=ACCEL/f'worker_{a.source:02d}.tasks.jsonl'
db=sqlite3.connect(ACCEL/f'results_{a.source:02d}.sqlite',timeout=30)
try:
 db.execute('BEGIN IMMEDIATE')
 tasks=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()];keys={t['key'] for t in tasks}
 done={r[0] for r in db.execute('SELECT key FROM results')} & keys
 pending=[t for t in tasks if t['key'] not in done]
 if len(pending)<300:db.rollback();raise SystemExit(3)
 matches=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   if proc.stat().st_uid==os.getuid() and len(args)==4 and args[1]==str(ROOT/'dual_crop_qc_worker.py') and args[2:]==['--slot',str(a.source)]:matches.append(int(proc.name))
  except (OSError,UnicodeError):continue
 if len(matches)!=1:raise RuntimeError('exact_shared_worker_required')
 # Holding the result writer transaction ensures the worker cannot commit an
 # in-flight task during transfer. Committed records are never reassigned.
 os.kill(matches[0],signal.SIGTERM)
 for _ in range(100):
  if not Path('/proc',str(matches[0])).exists():break
  time.sleep(.1)
 else:raise RuntimeError('shared_worker_did_not_exit')
 retained,moved=partition_pending(tasks,done)
 backup=ACCEL/f'extra_{a.extra:02d}.source_manifest.backup.jsonl';backup.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
 extra=ACCEL/f'worker_{a.extra:02d}.tasks.jsonl'
 def publish(path,values):
  temp=path.with_name(path.name+'.split.tmp');temp.write_text(''.join(json.dumps(t)+'\n' for t in values));temp.replace(path)
 publish(extra,moved);publish(manifest,retained)
 save(receipt,{'state':'sealed','source_slot':a.source,'extra_slot':a.extra,'task_count':len(moved),'retained_task_count':len(retained),'source_host':a.expected_host,'at_utc':now(),'committed_tasks_reassigned':0})
 db.commit()
 # Keep this CPU step visible until its replacement GPU process starts, so
 # the older watchdog cannot race the intentional checkpoint handoff.
 for attempt in range(600):
  found=False
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==matches[0]:continue
   try:
    args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if proc.stat().st_uid==os.getuid() and len(args)==4 and args[1]==str(ROOT/'dual_crop_qc_worker.py') and args[2:]==['--slot',str(a.source)]:found=True;break
   except (OSError,UnicodeError):continue
  if found:break
  time.sleep(.2)
 else:raise RuntimeError('shared_worker_restart_not_observed')
finally:db.close();lock.close()

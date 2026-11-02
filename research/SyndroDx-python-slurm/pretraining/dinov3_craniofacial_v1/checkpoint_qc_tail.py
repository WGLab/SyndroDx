"""Stop one exact reviewer at a durable checkpoint; assign pending work locally."""
import argparse,json,os,signal,socket,sqlite3,time
from pathlib import Path
from runtime import ROOT,RUN,load,save,now
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);p.add_argument('--expected-host',required=True);a=p.parse_args();TAIL=ROOT/'qc_tail'
if socket.gethostname().split('.')[0]!=a.expected_host.split('.')[0]:raise RuntimeError('original_review_writer_node_required')
def workers():
 result=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   if proc.stat().st_uid==os.getuid() and len(args)==4 and args[1]==str(ROOT/'dual_crop_qc_worker.py') and args[2:]==['--slot',str(a.slot)]:result.append(int(proc.name))
  except (OSError,UnicodeError):continue
 return result
matches=workers()
if len(matches)>1 or (not matches and a.slot==1):raise RuntimeError('exact_shared_reviewer_required')
db=sqlite3.connect(ROOT/'independent_qc'/f'slot_{a.slot:02d}.sqlite',timeout=30)
try:
 db.execute('BEGIN IMMEDIATE');done={r[0] for r in db.execute('SELECT key FROM reviews')}
 if matches:
  os.kill(matches[0],signal.SIGTERM)
  for _ in range(100):
   if not Path('/proc',str(matches[0])).exists():break
   time.sleep(.1)
  else:raise RuntimeError('reviewer_did_not_exit')
 if a.slot==1:
  keys=[]
  for owner in range(6):
   for attempt in range(5):
    source=None
    try:
     source=sqlite3.connect((RUN/'crop_pipeline/records'/f'slot_{owner:02d}.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
     values=[key for key, in source.execute("SELECT key FROM records WHERE status='validated_provisional'") if int(key[:16],16)%11==1 and key not in done];keys.extend(values);break
    except (sqlite3.Error,OSError):
     if attempt==4:raise RuntimeError('review_assignment_snapshot_unavailable')
     time.sleep(1)
    finally:
     if source is not None:source.close()
  plan=load(TAIL/'plan.json');recipients=plan['recipients'];keys=sorted(set(keys));assignments={key:recipients[i%len(recipients)] for i,key in enumerate(keys)}
  save(TAIL/'task_assignments.private.json',{'assignments':assignments,'committed_reviews_reassigned':0,'created_at_utc':now()});plan.update(state='running',task_count=len(keys),task_counts={str(s):sum(v==s for v in assignments.values()) for s in recipients},updated_at_utc=now());save(TAIL/'plan.json',plan)
 db.commit();save(TAIL/f'base_{a.slot:02d}.checkpoint.json',{'state':'checkpoint_saved','slot':a.slot,'at_utc':now()})
 # Keep the CPU step active while its new reviewer starts, preventing a race
 # with the older idle-only watchdog. No completed review is deleted.
 for _ in range(1200):
  if any(pid not in matches for pid in workers()):break
  time.sleep(.2)
 else:raise RuntimeError('reviewer_restart_not_observed')
finally:db.close()

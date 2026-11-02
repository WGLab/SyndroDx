"""Resume a lagging QC worker from durable records using the shared worker."""
import argparse,os,signal,time
from pathlib import Path
from runtime import ROOT,load,save,now
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args()
path=ROOT/'independent_qc'/f'slot_{a.slot:02d}.json'
for attempt in range(60):
 state=load(path)
 if state.get('state') not in ['reviewing','waiting_for_crops'] or time.time()-state.get('updated_at',0)>10:time.sleep(1);continue
 matches=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   if proc.stat().st_uid==os.getuid() and len(args)==4 and args[1]==str(ROOT/'independent_qc.py') and args[2:]==['--slot',str(a.slot)]:matches.append(int(proc.name))
  except (OSError,UnicodeError):continue
 if len(matches)!=1:raise RuntimeError('exact_QC_worker_required')
 os.kill(matches[0],signal.SIGTERM)
 for _ in range(100):
  if not Path('/proc',str(matches[0])).exists():break
  time.sleep(.1)
 else:raise RuntimeError('QC_worker_did_not_exit')
 # SQLite DELETE journals recover unfinished transactions on open. Every
 # committed review stays intact; an unsaved batch may be recomputed once.
 save(ROOT/'crop_acceleration'/f'slot_{a.slot:02d}.switched.json',{'state':'idle_QC_process_replaced','method':'durable_QC_checkpoint_handoff','prior_state':state['state'],'slot':a.slot,'at_utc':now()})
 raise SystemExit(0)
raise RuntimeError('recent_QC_checkpoint_unavailable')

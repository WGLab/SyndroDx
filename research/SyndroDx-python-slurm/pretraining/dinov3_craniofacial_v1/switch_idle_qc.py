"""Replace only the exact idle QC Python process; retain its Slurm allocation."""
import argparse,os,signal,time
from pathlib import Path
from runtime import ROOT,load,save,now
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args()
state_path=ROOT/'independent_qc'/f'slot_{a.slot:02d}.json'
for attempt in range(180):
 state=load(state_path)
 if state.get('state')=='finished':raise SystemExit(3)
 if state.get('state')=='waiting_for_crops' and 0<=time.time()-state.get('updated_at',0)<10:
  matches=[]
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    if proc.stat().st_uid!=os.getuid():continue
    args=(proc/'cmdline').read_bytes().split(b'\0');args=[x.decode() for x in args if x]
    if len(args)==4 and args[1]==str(ROOT/'independent_qc.py') and args[2:]==['--slot',str(a.slot)]:matches.append(int(proc.name))
   except (OSError,UnicodeError):continue
  if len(matches)!=1:raise SystemExit(4)
  # Read immediately before signalling; waiting state is emitted before a 45s sleep.
  check=load(state_path)
  if check.get('state')!='waiting_for_crops' or time.time()-check.get('updated_at',0)>=10:continue
  os.kill(matches[0],signal.SIGTERM)
  save(ROOT/'crop_acceleration'/f'slot_{a.slot:02d}.switched.json',{'state':'idle_QC_process_replaced','slot':a.slot,'prior_pid':matches[0],'at_utc':now()})
  raise SystemExit(0)
 time.sleep(2)
raise SystemExit(5)

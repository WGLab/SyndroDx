"""Read only this experiment's Slurm allocation receipts."""
import subprocess
from common import ROOT,SLURM,load,environment

def active_allocations():
 r=subprocess.run([str(SLURM/'squeue'),'-h','-r','-u','research_user','-o','%i|%T|%N'],env=environment(),capture_output=True,text=True,timeout=25,check=True)
 states={}
 for line in r.stdout.splitlines():
  fields=line.split('|')
  if len(fields)==3:states[fields[0]]=(fields[1],fields[2])
 active=[]
 for slot in range(20):
  item=load(ROOT/'allocations'/f'slot_{slot:02d}.json')
  key=item.get('lookup_id')
  if key in states and states[key][0]=='RUNNING':active.append(item)
 return active,states

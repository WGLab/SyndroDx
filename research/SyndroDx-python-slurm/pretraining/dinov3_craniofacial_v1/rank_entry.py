"""Map each allocated Slurm task to exactly one global distributed rank."""
import argparse,os
from runtime import ROOT
p=argparse.ArgumentParser();p.add_argument('--offset',type=int,required=True);p.add_argument('--world',type=int,required=True);p.add_argument('--master',required=True);p.add_argument('--port',required=True);p.add_argument('--run-id',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
rank=a.offset+int(os.environ.get('SLURM_PROCID','0'))
os.environ.update(RANK=str(rank),WORLD_SIZE=str(a.world),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',MASTER_ADDR=a.master,MASTER_PORT=a.port,TORCHELASTIC_RUN_ID=a.run_id)
arguments=a.arguments
if arguments and arguments[0]=='--':arguments=arguments[1:]
os.execv(str(ROOT/'venv/bin/python'),[str(ROOT/'venv/bin/python'),*arguments])

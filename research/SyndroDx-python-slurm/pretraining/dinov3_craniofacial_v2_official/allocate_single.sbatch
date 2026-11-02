#!/bin/bash
set -euo pipefail
umask 077
/usr/bin/python3 - <<'ALLOCATION'
import os,socket
from common import ROOT,save,now
slot=int(os.environ['SLURM_ARRAY_TASK_ID'])
save(ROOT/'allocations'/f'slot_{slot:02d}.json',dict(state='allocated',slot=slot,job_id=os.environ['SLURM_JOB_ID'],lookup_id=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'],node=socket.gethostname(),allocated_at_utc=now()))
ALLOCATION
exec /bin/sleep infinity

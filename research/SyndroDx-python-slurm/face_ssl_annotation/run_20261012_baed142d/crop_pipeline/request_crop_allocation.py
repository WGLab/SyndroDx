"""Request persistent single-GPU allocations; scheduler logs stay local."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

os.umask(0o077)
RUN=Path(Path('/tmp/local_face_label_setup/active_run.txt').read_text().strip())


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def request(slot,partition='gpu-xe9680q'):
    with (RUN/'crop_allocation_requests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=RUN/'crop_allocations.json'
        data=json.loads(registry.read_text()) if registry.exists() else {'allocations':[]}
        prior=next((x for x in data['allocations'] if x['worker']==slot),None)
        if prior:
            print(json.dumps({'slot':slot,'state':prior['state'],'already_requested':True,
                              'allocation_job_id':prior.get('allocation_job_id')}))
            return
        stderr=RUN/f'crop_salloc_{slot:02d}.stderr.log'
        stdout=RUN/f'crop_salloc_{slot:02d}.stdout.log'
        walltime='5-00:00:00' if partition=='gpuq' else '10-00:00:00'
        seconds='432000' if partition=='gpuq' else '864000'
        command=['/opt/site/slurm/bin/salloc','--job-name=facecrop-local-'+str(slot),
                 '--partition='+partition,'--nodes=1','--ntasks=1','--cpus-per-task=4',
                 '--gres=gpu:a100:1' if partition=='gpuq' else '--gres=gpu:1',
                 '--mem=56G','--time='+walltime]
        if partition=='gpuq':
            # This node was verified with an 80GB allocation in the prior run.
            command.append('--nodelist=g-10-01')
        command.extend(['/usr/bin/sleep',seconds])
        with stdout.open('a') as out,stderr.open('a') as err:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        row={'worker':slot,'state':'awaiting_scheduler_reply','requested_at':time.time(),
             'requester_pid':process.pid,'kind':'crop_keepalive','partition':partition,
             'walltime':walltime,'allocation_body':'sleep_'+seconds}
        data['allocations'].append(row);save(registry,data)
        deadline=time.time()+40
        while time.time()<deadline:
            text=stderr.read_text(errors='replace')
            matches=re.findall(r'(?:Pending job allocation|Granted job allocation) (\d+)',text)
            if matches:
                row['allocation_job_id']=matches[-1];row['state']='submitted'
                save(registry,data)
                print(json.dumps({'slot':slot,'state':'submitted','allocation_job_id':row['allocation_job_id']}))
                return
            if process.poll() is not None:
                row['state']='request_failed';row['returncode']=process.returncode;save(registry,data)
                print(json.dumps({'slot':slot,'state':'request_failed','returncode':process.returncode}))
                return
            time.sleep(0.5)
        print(json.dumps({'slot':slot,'state':'awaiting_scheduler_reply'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--slot',type=int,required=True,choices=range(6))
    parser.add_argument('--partition',choices=('gpu-xe9680q','gpuq'),default='gpu-xe9680q')
    args=parser.parse_args()
    request(args.slot,args.partition)

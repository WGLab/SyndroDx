"""Resume only the identified missing comparison-token helper failure."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

os.umask(0o077)
RUN = Path(__file__).resolve().parent


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    with (RUN/'known_handler_recovery.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        allocations=[]
        for name in ('single_gpu_allocations.json','a10080_allocations.json'):
            path=RUN/name
            if path.exists():allocations.extend(json.loads(path.read_text()).get('allocations',[]))
        archive=RUN/'direct_workers'/'claims'/'recovered'
        registry_path=RUN/'known_handler_recoveries.json'
        registry=json.loads(registry_path.read_text()) if registry_path.exists() else {}
        for item in allocations:
            slot=int(item['worker'])
            replica=RUN/'replicas'/('slot_%02d.json'%slot)
            if not replica.exists():continue
            state=json.loads(replica.read_text())
            if state.get('state')!='failed' or str(slot) in registry:continue
            error=RUN/'replicas'/('slot_%02d.error.log'%slot)
            if not error.exists() or "NameError: name 'token' is not defined" not in error.read_text():continue
            job=str(item['allocation_job_id'])
            status=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=15)
            if status.returncode or status.stdout.strip()!='RUNNING':continue
            archive.mkdir(exist_ok=True)
            for path in (RUN/'direct_workers').glob('shard_*.status.json'):
                shard=json.loads(path.read_text())
                if shard.get('state')!='failed' or shard.get('slot')!=slot:continue
                claim=RUN/'direct_workers'/'claims'/path.name.replace('.status.json','')
                if not claim.exists():continue
                owner=json.loads((claim/'owner.json').read_text())
                if owner.get('slot')!=slot:continue
                claim.rename(archive/(claim.name+'_'+str(time.time_ns())))
                save(path,{**shard,'state':'waiting_for_resume','updated_at':time.time()})
            save(replica,{**state,'state':'retry_scheduled','updated_at':time.time()})
            settings=json.loads((RUN/'config.json').read_text())
            command=['srun','--jobid='+job,'--nodes=1','--ntasks=1','--cpus-per-task=4',
                '--gpus-per-task=1','--gpu-bind=single:1','--exclusive',
                settings['python'],str(RUN/'qwen_replica_service.py'),'--slot='+str(slot)]
            with (RUN/'logs'/('known_error_recovery_slot_%02d.log'%slot)).open('ab') as log:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True,cwd=RUN)
            registry[str(slot)]={'state':'resume_launched','pid':process.pid,'updated_at':time.time(),
                                 'reason':'fixed_missing_comparison_token_helper'}
            save(registry_path,registry)


if __name__=='__main__':
    main()

"""Reversibly park only named CPU coordinators; never signal GPU allocations."""
import os,sys,signal,json
from pathlib import Path
root=Path(__file__).resolve().parent
targets=[root.parent/'arcface_transfer_evaluation_20261028/continuous_seeds.py',root.parent/'arcface_transfer_evaluation_20261028/extra_seeds.py',root.parent/'arcface_transfer_evaluation_20261028/fill_seeds.py',root/'controller.py']
found=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        args=(p/'cmdline').read_bytes().split(b'\0')
        for target in targets:
            if str(target).encode() in args and b'python' in args[0]:
                os.kill(int(p.name),signal.SIGSTOP);found.append(dict(pid=int(p.name),script=target.name))
    except (OSError,IndexError):pass
name=__import__('socket').gethostname()
(root/('parked_cpu_'+name+'.json')).write_text(json.dumps(dict(host=name,processes=found),indent=2))
print(json.dumps(dict(host=name,parked=found,gpu_workers_signalled=False,allocations_cancelled=False)))

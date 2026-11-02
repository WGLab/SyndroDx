"""Keep early validation, recent recovery, milestone and best full checkpoints."""
import os
from control import WORK,load,save,now

def protected_steps(steps,protocol,best_steps=()):
    steps=sorted(set(steps));keep=set(steps[-3:])|set(best_steps)|{0}
    validation=[n for n in steps if n in protocol['validation_first_steps'] or n%protocol['validation_every_steps']==0]
    keep.update(validation[-4:]);keep.update(n for n in validation if n<=2000)
    keep.update(n for n in steps if n%5000==0)
    return keep

def rotate():
    files=sorted((WORK/'checkpoints').glob('step_*.pt'))
    steps={int(p.stem.split('_')[-1]):p for p in files}
    best=load(WORK/'best/registry.json')
    keep=protected_steps(steps,load(WORK/'protocol.json'),[v['step'] for v in best.values()])
    for n,path in steps.items():
        if n not in keep:path.unlink();path.with_suffix('.json').unlink(missing_ok=True)

def record_validation(step):
    report=load(WORK/'validation'/f'step_{step:07d}.json')
    path=WORK/'checkpoints'/f'step_{step:07d}.pt';receipt=load(path.with_suffix('.json'))
    if report.get('state')!='complete' or receipt.get('state')!='complete' or not path.exists():
        raise RuntimeError('complete_validation_and_checkpoint_required_for_best_selection')
    registry=load(WORK/'best/registry.json')
    for category in ('disease','structural'):
        for endpoint in ('retrieval','ridge'):
            key=category+'_'+endpoint;value=report['metrics'][category][endpoint]['top1']
            if key in registry and value<=registry[key]['macro_top1']:continue
            target=WORK/'best'/(key+'.pt');tmp=target.with_name(target.name+'.tmp.'+str(os.getpid()))
            tmp.unlink(missing_ok=True);os.link(path,tmp);tmp.replace(target)
            registry[key]=dict(step=step,macro_top1=value,path=str(target),sha256=receipt['sha256'],selected_at_utc=now())
    save(WORK/'best/registry.json',registry)


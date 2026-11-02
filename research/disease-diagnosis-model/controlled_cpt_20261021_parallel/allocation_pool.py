"""Exact disjoint, reviewed allocation pools. No automatic lease adoption."""
import subprocess
from configuration import WORK, WORLD, ARMS
from common import SLURM, environment, load


class SchedulerUnavailable(RuntimeError):
    pass


def query(arguments):
    try:
        result = subprocess.run([str(SLURM/arguments[0]), *arguments[1:]], env=environment(),
                                capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise SchedulerUnavailable('scheduler_query_timeout') from error
    if result.returncode:
        if 'Invalid job id' in result.stderr:
            raise RuntimeError('selected_allocation_no_longer_available')
        raise SchedulerUnavailable('scheduler_query_failed')
    return result.stdout


def select_pools(specs, records, steps):
    if set(specs) != set(ARMS) or any(len(specs[a]) != WORLD for a in ARMS):
        raise RuntimeError('allocation_count_invalid')
    identities = [s['job_id'] for a in ARMS for s in specs[a]]
    if len(set(identities)) != WORLD*len(ARMS):
        raise RuntimeError('overlapping_allocations')
    result = {}
    for arm in ARMS:
        if len({s['node'] for s in specs[arm]}) != 1:
            raise RuntimeError('arm_not_on_single_node')
        result[arm] = []
        for slot, spec in enumerate(specs[arm]):
            record = records[spec['lookup_id']]
            if record.get('JobId') != spec['job_id'] or not record.get('UserId', '').startswith('research_user('):
                raise RuntimeError('allocation_identity_mismatch')
            if record.get('JobState') in ('PENDING', 'CONFIGURING'):
                continue
            if record.get('JobState') != 'RUNNING':
                raise RuntimeError('selected_allocation_ended')
            tres = dict(x.split('=', 1) for x in record.get('AllocTRES','').split(',') if '=' in x)
            memory = tres.get('mem', '')
            if (record.get('NodeList') != spec['node'] or record.get('Partition') != 'gpu-xe9680q'
                    or tres.get('gres/gpu') != '1' or int(tres.get('cpu', 0)) < 4
                    or not memory.endswith('G') or float(memory[:-1]) < spec['host_memory_gib']):
                raise RuntimeError('allocation_resources_mismatch')
            busy, overlap = False, False
            for step, name in steps:
                if step.split('.')[0] not in (spec['job_id'], spec['lookup_id']) or step.endswith(('.batch','.extern')):
                    continue
                if spec.get('allowed_idle_steps', {}).get(step) == name:
                    overlap = True
                else:
                    busy = True
            if not busy:
                result[arm].append(dict(spec, slot=slot, step_overlap=overlap))
    return result


def idle_pools():
    specs = load(WORK/'selected_allocations.json')['arms']
    records = {}
    for arm in ARMS:
        for spec in specs[arm]:
            output = query(['scontrol', 'show', 'job', spec['lookup_id'], '-o'])
            records[spec['lookup_id']] = dict(x.split('=', 1) for x in output.split() if '=' in x)
    steps = []
    for line in query(['squeue','-h','--steps','-u','research_user','-o','%i|%j']).splitlines():
        fields = line.strip().split('|')
        if len(fields) != 2:
            raise RuntimeError('invalid_scheduler_step_format')
        steps.append(fields)
    return select_pools(specs, records, steps)


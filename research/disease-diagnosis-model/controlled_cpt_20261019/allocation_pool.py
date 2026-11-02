"""Admit only reviewed GPU leases; transient scheduler failures never launch work."""
import subprocess
from configuration import WORK, WORLD
from common import SLURM, environment, load


class SchedulerUnavailable(RuntimeError):
    pass


def query(arguments):
    try:
        result = subprocess.run([str(SLURM / arguments[0]), *arguments[1:]],
                                env=environment(), capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise SchedulerUnavailable('scheduler_query_timeout') from error
    if result.returncode:
        if 'Invalid job id' in result.stderr:
            raise RuntimeError('selected_allocation_no_longer_available')
        raise SchedulerUnavailable('scheduler_query_failed')
    return result.stdout


def select_pool(specifications, records, steps):
    if len(specifications) != WORLD or len({s['job_id'] for s in specifications}) != WORLD:
        raise RuntimeError('selected_allocation_count_or_identity_invalid')
    active = []
    for slot, spec in enumerate(specifications):
        record = records[spec['lookup_id']]
        if record.get('JobId') != spec['job_id'] or not record.get('UserId', '').startswith('research_user('):
            raise RuntimeError('allocation_owner_or_identity_mismatch')
        state = record.get('JobState')
        if state in ('PENDING', 'CONFIGURING'):
            continue
        if state != 'RUNNING':
            raise RuntimeError('selected_allocation_ended')
        tres = dict(field.split('=', 1) for field in record.get('AllocTRES', '').split(',') if '=' in field)
        if (record.get('Partition') != 'gpu-xe9680q' or record.get('NodeList') != spec['node']
                or tres.get('gres/gpu') != '1' or int(tres.get('cpu', '0')) < 4):
            raise RuntimeError('allocation_resources_mismatch')
        memory = tres.get('mem', '')
        if not memory.endswith('G') or float(memory[:-1]) < 68:
            raise RuntimeError('allocation_host_memory_insufficient')
        aliases = {spec['job_id'], spec['lookup_id']}
        busy, overlap = False, False
        for step, name in steps:
            if step.split('.')[0] not in aliases or step.endswith(('.batch', '.extern')):
                continue
            if spec.get('allowed_idle_steps', {}).get(step) == name:
                overlap = True  # Existing verified sleep holds this salloc alive.
            else:
                busy = True
        if not busy:
            active.append(dict(spec, slot=slot, step_overlap=overlap))
    return active


def idle_pool():
    specs = load(WORK/'selected_allocations.json')['allocations']
    records = {}
    for spec in specs:
        text = query(['scontrol', 'show', 'job', spec['lookup_id'], '-o'])
        records[spec['lookup_id']] = dict(item.split('=', 1) for item in text.split() if '=' in item)
    lines = query(['squeue', '-h', '--steps', '-u', 'research_user', '-o', '%i|%j']).splitlines()
    steps = []
    for line in lines:
        fields = line.strip().split('|')
        if len(fields) != 2:
            raise RuntimeError('unrecognized_scheduler_step_format')
        steps.append(fields)
    return select_pool(specs, records, steps)

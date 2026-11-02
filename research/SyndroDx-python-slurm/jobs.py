# -*- coding: utf-8 -*-
"""Read-only job navigation. Does not submit, install, or execute scripts."""
import argparse
import hashlib
from pathlib import Path
from jobs_catalog import JOBS
from catalog import CATALOG
BASE = Path(__file__).resolve().parent

def select(name):
    if name in JOBS:
        return name
    matches = [k for k in JOBS if Path(k).name == name or Path(k).stem == name]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous job: ' + name)
    return matches[0]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list')
    show = sub.add_parser('show'); show.add_argument('job')
    check = sub.add_parser('check')
    check.add_argument('--workspace', type=Path)
    args = parser.parse_args()
    if args.action == 'list':
        for name, e in JOBS.items():
            print(f"{name} | {e['group']} | {e['purpose']}")
        return 0
    if args.action == 'show':
        name = select(args.job); e = JOBS[name]
        print(f"原路径: {name}\n整理位置: {e['source']}\n来源: {e['origin']}\n用途: {e['purpose']}")
        print('Python: ' + ', '.join(e['python_entries']))
        for key, value in e['resources'].items():
            print(f"  {key}: {value}")
        print(e['note'])
        print('\n' + (BASE / e['source']).read_text())
        return 0
    workspace = args.workspace.expanduser().resolve() if args.workspace else None
    if workspace is not None and not workspace.is_dir():
        parser.error('Workspace must be an existing directory')
    failures = []
    for name, e in JOBS.items():
        path = workspace / name if workspace else BASE / e['source']
        if not path.is_file() or digest(path) != e['sha256']:
            failures.append(name + ': missing or source differs')
        for target in e['python_entries']:
            meta = CATALOG[target]
            p = workspace / target if workspace else BASE / meta['source']
            if not p.is_file() or digest(p) != meta['sha256']:
                failures.append(name + ' -> ' + target + ': missing or source differs')
    for failure in failures:
        print(failure)
    print(f"Checked {len(JOBS)} jobs and their direct Python entries; {len(failures)} problems.")
    print('Code-only check; data, environments, Slurm availability, runtime state and transitive dependencies are not verified.')
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Navigate organized sources and run them only in their canonical HPC layout.

The catalog is static: list/show/check/install never import processing scripts.
Code is kept byte-identical so __file__, sibling imports and frozen hashes retain
their original meaning after installation. No data are bundled or fabricated.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from catalog import CATALOG

BASE = Path(__file__).resolve().parent
FLOW = """
A. Candidate inventory and evidence (historical frontal bootstrap)
   build_labels -> curate_available_data -> assemble_candidate_evidence
   normalize_mondo + manual view_qc_sample -> finalize_data_inventory
   -> audit_duplicate_and_patient_conflicts
   expert audit -> split proposal -> reservation guards
   candidate concepts -> diagnosis strata -> four subsets

B. Conservative SSL preparation (subset 4)
   face_qc/batch -> build_ssl_admission_inventory -> clean_ssl_image_lineage
   -> controlled visual review / original-preserving crop pilot
   Preparation ledgers and split proposals are NOT training admission.

C. Full original-image release (all indexed views)
   face_qc/full_audit -> face_qc/stream_qwen_review_v5
   -> merge + coverage checks -> face_qc/pretrain_release
   -> face_qc/pretrain_crop
   This is not wired into legacy ssl_dataset/ssl_train.

The arrows describe major artifact relationships, not an automatic schedule.
Manual review, local prompts, model files, ontology files and frozen manifests
are required at several boundaries. Use show/check before choosing a stage.
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(name):
    if name in CATALOG:
        return name
    stem = name.removesuffix('.py')
    matches = [k for k in CATALOG if Path(k).stem == stem or k.removesuffix('.py') == stem]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous script: ' + name)
    return matches[0]


def workspace(value):
    if not value:
        raise ValueError('Provide --workspace or set SYNDRODX_WORKSPACE to the HPC project directory')
    p = Path(value).expanduser().resolve()
    if not p.is_dir():
        raise ValueError('Workspace must already exist: ' + str(p))
    if p == BASE or BASE in p.parents or p in BASE.parents:
        raise ValueError('Keep the organized source directory separate from the HPC workspace')
    return p


def target_path(root, canonical):
    target = root / canonical
    # Never follow code symlinks into another directory or overwrite a data target.
    for p in [target, *target.parents]:
        if p == root:
            break
        if p.is_symlink():
            raise ValueError('Code target contains a symlink: ' + canonical)
    if target.exists() and not target.is_file():
        raise ValueError('Code target is not a regular file: ' + canonical)
    return target


def verified_source(canonical):
    entry = CATALOG[canonical]
    p = BASE / entry['source']
    if digest(p) != entry['sha256']:
        raise ValueError('Organized source has changed: ' + entry['source'])
    return p


def install(root, apply=False, replace=False):
    plans = []
    # Preflight every source/target before any mutation.
    for canonical in sorted(CATALOG):
        src = verified_source(canonical)
        dst = target_path(root, canonical)
        state = 'missing' if not dst.exists() else 'same' if digest(dst) == CATALOG[canonical]['sha256'] else 'changed'
        plans.append((canonical, src, dst, state))
        print(state.ljust(8), canonical)
    changed = [item for item in plans if item[3] == 'changed']
    if not apply:
        print('\nPreview only. No files written. Use --apply after reviewing the list.')
        return 0
    if changed and not replace:
        raise ValueError('Different HPC versions found. Review them before using --replace-changed.')
    backup = None
    if changed:
        backup = Path(tempfile.mkdtemp(prefix='syndrodx_code_backup_', dir=root))
        for canonical, _, dst, _ in changed:
            saved = backup / canonical
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(dst.read_bytes())
    for canonical, src, dst, state in plans:
        if state == 'same':
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace for existing files; exclusive create for new files.
        if state == 'missing':
            with dst.open('xb') as stream:
                stream.write(src.read_bytes())
        else:
            if digest(dst) != digest(backup / canonical):
                raise ValueError('HPC source changed during installation: ' + canonical)
            fd, tmp = tempfile.mkstemp(prefix='.syndrodx_', suffix='.py', dir=dst.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(src.read_bytes())
                os.chmod(tmp, dst.stat().st_mode & 0o777)
                os.replace(tmp, dst)
            finally:
                if Path(tmp).exists():
                    Path(tmp).unlink()
    print('Installed Python source only. Data, configurations and manifests were not changed.')
    if backup:
        print('Previous code saved in:', backup)
    return 0


def check(root, name=None):
    keys = [select(name)] if name else sorted(CATALOG)
    missing_code = changed_code = missing_inputs = 0
    for canonical in keys:
        entry = CATALOG[canonical]
        verified_source(canonical)
        target = target_path(root, canonical)
        if not target.exists():
            print('MISSING CODE', canonical)
            missing_code += 1
        elif digest(target) != entry['sha256']:
            print('DIFFERENT CODE', canonical)
            changed_code += 1
        for item in entry['primary_inputs']:
            p = Path(item)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                print('MISSING INPUT', canonical, item)
                missing_inputs += 1
    print('Missing code:', missing_code, '| Different code:', changed_code, '| Missing primary inputs:', missing_inputs)
    print('Primary-input checks are partial and use default arguments. They do not validate data, approvals, models or frozen state.')
    return int(bool(missing_code or changed_code or missing_inputs))


def run(root, name, arguments, execute=False):
    canonical = select(name)
    entry = CATALOG[canonical]
    if entry['status'] in {'library', 'operations'}:
        raise ValueError('This is a library or operations tool, not a workflow entry. Inspect it separately.')
    if entry['status'] == 'historical':
        print('Historical/fixed-cohort script: inspect its assumptions before use.')
    # Validate the selected script and all transitive local imports.
    pending = [canonical]
    seen = set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        verified_source(key)
        target = target_path(root, key)
        if not target.exists() or digest(target) != CATALOG[key]['sha256']:
            raise ValueError('Install/review the matching HPC source first: ' + key)
        pending.extend(CATALOG[key]['code_dependencies'])
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    command = [sys.executable, str(root / canonical), *arguments]
    print('Working directory:', root)
    print('Command:', shlex.join(command), flush=True)
    if not execute:
        print('Preview only. Add --execute before the script name to run it.')
        return 0
    # No shell, implicit dependency runs, scheduler submissions or email tools.
    return subprocess.run(command, cwd=root).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('list', help='Show the categorized script index without imports')
    p.add_argument('--group')
    sub.add_parser('flow', help='Explain the three processing branches')
    p = sub.add_parser('show', help='Show purpose, source origin, imports and primary inputs')
    p.add_argument('script')
    for action in ('check', 'install', 'run'):
        p = sub.add_parser(action)
        p.add_argument('--workspace', default=os.environ.get('SYNDRODX_WORKSPACE'))
        if action == 'check':
            p.add_argument('script', nargs='?')
        elif action == 'install':
            p.add_argument('--apply', action='store_true')
            p.add_argument('--replace-changed', action='store_true')
        else:
            p.add_argument('--execute', action='store_true')
            p.add_argument('script')
            p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'list':
        for group in sorted({e['group'] for e in CATALOG.values()}):
            if args.group and args.group != group:
                continue
            print('\n[' + group + ']')
            for key, entry in sorted(CATALOG.items()):
                if entry['group'] == group:
                    print(key, '|', entry['status'], '|', entry['purpose'])
        return 0
    if args.command == 'flow':
        print(FLOW)
        return 0
    if args.command == 'show':
        key = select(args.script)
        print(key)
        for field in ('purpose', 'group', 'status', 'origin', 'source', 'has_main_guard', 'code_dependencies', 'primary_inputs', 'imports', 'doc'):
            print(field + ':', CATALOG[key][field])
        return 0
    root = workspace(args.workspace)
    if args.command == 'check':
        return check(root, args.script)
    if args.command == 'install':
        return install(root, args.apply, args.replace_changed)
    return run(root, args.script, args.arguments, args.execute)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)

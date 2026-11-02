"""Local-only I/O. Do not print patient records or exception messages."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PRIVATE = ROOT / 'private'
os.umask(0o077)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def tsv(path):
    with Path(path).open() as f:
        yield from csv.DictReader(f, delimiter='\t')


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        for row in records:
            f.write(json.dumps(row) + '\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def guarded_main(fn):
    try:
        fn()
    except Exception as error:
        # A traceback from a parser can include private values. Save only type.
        save(ROOT / (fn.__module__.replace('.', '_') + '_failure.json'),
             {'error_type': type(error).__name__, 'updated_at_utc': now()})
        print(json.dumps({'state': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1) from None

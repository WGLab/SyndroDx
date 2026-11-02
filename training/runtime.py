"""Private workspace I/O. This module never downloads or uploads data."""
import os, sys, json, hashlib, datetime, importlib.util
from pathlib import Path
CODE = Path(__file__).resolve().parent
PROJECT = Path(os.environ['CRL_WORKSPACE']).expanduser().resolve()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return {} if default is None else default
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2)+'\n'); os.chmod(tmp,0o600); tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def upstream(name, env):
    p=Path(os.environ[env]).resolve()
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

"""Check the research archive without importing or executing research programs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'research'


def main():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    checked=0;python_files=0;json_files=0
    for row in manifest['files']:
        path=ROOT/row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('invalid_source_path')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('source_hash_mismatch: '+row['path'])
        if path.suffix=='.py':
            ast.parse(data.decode());python_files+=1
        if path.suffix=='.json':
            json.loads(data);json_files+=1
        checked+=1
    print(json.dumps({'state':'passed','source_files':checked,'Python_syntax_checks':python_files,
        'JSON_parse_checks':json_files,'training_executed':False,'patient_data_loaded':False}))


if __name__=='__main__':main()

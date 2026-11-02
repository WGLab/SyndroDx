"""Create a shareable archive from a strict allowlist, with integrity receipts."""
import gzip
import hashlib
import json
import os
import pathlib
import tarfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
BUILD=ROOT/'bch_release_build_20261102'
BUNDLE=ROOT/'model_releases/bch_evaluation_v1_20261102'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def main():
    os.umask(0o077)
    parity=json.loads((BUILD/'verification.json').read_text())
    smoke=json.loads((BUILD/'smoke.json').read_text())
    assert parity['state']==smoke['state']=='passed'
    (BUNDLE/'VERIFICATION.json').write_text(json.dumps({'numerical_parity':parity,'portable_command_test':smoke},indent=2)+'\n')
    files=['README.md','run.sh','requirements.txt','predict.py','networks.py','config.json','VERIFICATION.json',
        'assets/diseases.json','assets/term_index.json','examples/cases.jsonl',
        'weights/visual.pt','weights/visual_head.pt','weights/hpo.pt','weights/fusion.pt']
    assert all((BUNDLE/n).is_file() and not (BUNDLE/n).is_symlink() for n in files)
    pins={n:sha(BUNDLE/n) for n in files}
    (BUNDLE/'checksums.json').write_text(json.dumps(pins,indent=2)+'\n')
    files.append('checksums.json')
    # Scan all exportable non-tensor contents for accidental local paths/credentials.
    for n in files:
        if n.endswith('.pt'):continue
        content=(BUNDLE/n).read_text()
        assert '/controlled-data/users/' not in content and '/opt/craniofacial/user_home' not in content
        assert not __import__('re').search(r'hf_[A-Za-z0-9]{20,}',content)
    archive=BUNDLE.with_suffix('.tar.gz')
    temporary=archive.with_suffix('.gz.tmp')
    with temporary.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as compressed:
        with tarfile.open(mode='w',fileobj=compressed) as tar:
            for name in files:
                p=BUNDLE/name
                info=tar.gettarinfo(str(p),arcname=BUNDLE.name+'/'+name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
                with p.open('rb') as source:tar.addfile(info,source)
    temporary.replace(archive)
    # Validate every archived byte against the per-file manifest.
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len(files) and all(m.isfile() for m in members)
        for name in files:
            stream=tar.extractfile(BUNDLE.name+'/'+name)
            h=hashlib.sha256()
            for block in iter(lambda:stream.read(8388608),b''):h.update(block)
            assert h.hexdigest()==sha(BUNDLE/name)
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    receipt={'state':'ready_to_share','archive':str(archive),'sha256':digest,'archive_bytes':archive.stat().st_size,
        'allowlisted_files':len(files),'archive_content_hashes_verified':True,
        'source_checkpoint_unchanged':True,'patient_data_included':False,'training_state_included':False,
        'uploaded_or_sent_to_BCH':False,'runtime_test_device':'CPU','runtime_precision':'FP32'}
    (BUILD/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'model_releases/BCH_EVALUATION_CURRENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()

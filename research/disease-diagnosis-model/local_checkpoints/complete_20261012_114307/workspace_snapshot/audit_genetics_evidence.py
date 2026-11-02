#!/usr/bin/env python3
"""Audit a possible MRN-to-patient-ID bridge to clinician HPO; no note text emitted."""
import collections
import csv
import gzip
import json
from pathlib import Path

from audit_metadata_sources import FRONT, clean, mrnkey

ROOT=Path('/controlled-data/craniofacial_data')
OUT=Path(__file__).resolve().parent/'metadata_audit'


def open_csv(path):
    with path.open('rb') as f:magic=f.read(2)
    return gzip.open(path,'rt',newline='') if magic==b'\x1f\x8b' else path.open(newline='')


def main():
    csv.field_size_limit(10**8)
    with (ROOT/'genetics_photos_metdata_top10dx.csv').open() as f:
        g=[r for r in csv.DictReader(f,delimiter='\t') if r['EVA02_Organ']==FRONT]
    target={mrnkey(r['pat_mrn_id']) for r in g}
    direct=collections.defaultdict(set)
    for r in g:direct[mrnkey(r['pat_id'])].add(mrnkey(r['pat_mrn_id']))
    bridge=collections.defaultdict(set)
    patient_files=sorted((ROOT/'arcus_export/arcus_2026_06_29_full_ctakes').glob('patient-*.csv'))
    n=0
    for i,p in enumerate(patient_files):
        with open_csv(p) as f:
            for r in csv.DictReader(f):
                n+=1;m=mrnkey(r['pat_mrn_id'])
                if m in target:bridge[mrnkey(r['pat_id'])].add(m)
        print('patient_shard',i+1,'/',len(patient_files),'matched_mrns',len(set().union(*bridge.values())) if bridge else 0,flush=True)
    # Exclude ambiguous patient-ID keys; do not silently resolve identity conflicts.
    usable={pid:next(iter(ms)) for pid,ms in bridge.items() if len(ms)==1}
    hpo_files=list((ROOT/'arcus_export/arcus_2026_06_29').glob('clinician_hpo_element*'))
    pats=set();codes=set();matched=set();matched_codes=set();direct_matched=set();n_hpo=0;matched_rows=0;notes=0
    for p in hpo_files:
        with open_csv(p) as f:
            for r in csv.DictReader(f):
                n_hpo+=1;pid=mrnkey(r['pat_id']);pats.add(pid)
                if clean(r.get('hpo_code')):codes.add(r['hpo_code'])
                if pid in direct:direct_matched.update(direct[pid])
                if pid in usable:
                    matched.add(usable[pid]);matched_rows+=1
                    if clean(r.get('hpo_code')):matched_codes.add(r['hpo_code'])
                    if clean(r.get('notes')):notes+=1
    raw=ROOT/'Genetics-Patient-Photos'
    import os
    exts=collections.Counter();nd=0;links=0
    for dp,dn,fn in os.walk(raw):
        nd+=1
        links+=sum(Path(dp,n).is_symlink() for n in dn)
        exts.update(Path(n).suffix.lower() for n in fn)
    result=dict(patient_bridge=dict(shards=len(patient_files),rows=n,
                                    matched_genetics_frontal_mrns=len(set(usable.values())),
                                    ambiguous_patient_id_keys=sum(len(v)>1 for v in bridge.values())),
                clinician_hpo=dict(rows=n_hpo,patient_ids=len(pats),unique_hpo_codes=len(codes),
                                   direct_normalized_patient_id_overlap=len(direct_matched),
                                   matched_genetics_frontal_patients=len(matched),matched_rows=matched_rows,
                                   matched_hpo_codes=len(matched_codes),matched_rows_with_notes=notes),
                raw_genetics_archive=dict(directories=nd,symlink_directories=links,file_extensions=dict(exts)),
                scope='Clinician HPO and patient bridge were scanned fully. Large cTAKES annotation exports were inspected only for schemas; no clinical note bodies were exported or printed.',
                notes=['HPO overlap does not imply a confirmed diagnosis; timestamp and diagnostic-information leakage require review.',
                       'The clinician_hpo notes field has not been reviewed for clinical content or availability at prediction time.'])
    (OUT/'genetics_evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

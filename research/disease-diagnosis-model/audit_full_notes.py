#!/usr/bin/env python3
"""Streaming coverage audit of full note archive; never retain or print note text."""
import collections
import csv
import json
from pathlib import Path
from audit_metadata_sources import FRONT, clean, mrnkey

ROOT=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images')
HERE=Path(__file__).resolve().parent


def main():
    csv.field_size_limit(10**8)
    cf=set();gf=set()
    with (HERE/'labels.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if mrnkey(r['mrn']):cf.add(mrnkey(r['mrn']))
    with open('/controlled-data/craniofacial_data/genetics_photos_metdata_top10dx.csv') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['EVA02_Organ']==FRONT:gf.add(mrnkey(r['pat_mrn_id']))
    stats=collections.Counter();pats=set();textpats=set();noteids=set();perpat=collections.Counter()
    source=ROOT/'all_clarity_notes_merged.csv'
    with source.open(newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            stats['rows']+=1;m=mrnkey(r.get('MRN'));nid=clean(r.get('NOTE_ID'))
            if m:pats.add(m);perpat[m]+=1
            if nid:noteids.add(nid)
            if clean(r.get('TEXT')):
                stats['rows_with_text']+=1
                if m:textpats.add(m)
                if m in cf:stats['text_rows_craniofacial_frontal']+=1
                if m in gf:stats['text_rows_genetics_frontal']+=1
            if clean(r.get('DATE')):stats['rows_with_date']+=1
            if stats['rows']%50000==0:print('scanned_note_rows',stats['rows'],flush=True)
    result=dict(source=str(source),size_bytes=source.stat().st_size,counts=dict(stats),
                unique_mrns=len(pats),unique_note_ids=len(noteids),patients_with_text=len(textpats),
                patients_with_multiple_rows=sum(v>1 for v in perpat.values()),
                craniofacial_frontal_patients_with_text=len(textpats&cf),
                genetics_frontal_patients_with_text=len(textpats&gf),
                scope='Complete streaming scan for coverage only. No diagnostic assertions adjudicated; no clinical text copied or printed. MRN links remain provisional until identity reconciliation.')
    (HERE/'metadata_audit/full_notes_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

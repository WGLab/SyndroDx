#!/usr/bin/env python3
"""Aggregate-only genetics image/diagnosis audit; never changes clinical labels."""
import ast
import collections
import csv
import json
import os
import random
from pathlib import Path

from audit_metadata_sources import FRONT, clean, mrnkey
from ontology import parse_obo

HERE = Path(__file__).resolve().parent
ROOT = Path('/controlled-data/craniofacial_data')
OUT = HERE / 'metadata_audit'


def rows(name):
    with (ROOT / name).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def image_stats(rr):
    pats = {mrnkey(r['pat_mrn_id']) for r in rr if mrnkey(r['pat_mrn_id'])}
    paths = {r['image_path'] for r in rr if r['image_path']}
    perpat = collections.defaultdict(set)
    pathpat = collections.defaultdict(set)
    pathlabels = collections.defaultdict(set)
    for r in rr:
        m = mrnkey(r['pat_mrn_id'])
        perpat[m].add(r['image_path'])
        pathpat[r['image_path']].add(m)
        pathlabels[r['image_path']].add(r['unified_syndrome'])
    return dict(rows=len(rr), unique_images=len(paths), unique_patients=len(pats),
                patients_with_multiple_images=sum(len(x)>1 for x in perpat.values()),
                images_with_multiple_patient_ids=sum(len(x)>1 for x in pathpat.values()),
                images_with_multiple_label_strings=sum(len(x)>1 for x in pathlabels.values()),
                patients=pats, paths=paths)


def public_stats(stats):
    return {k:v for k,v in stats.items() if k not in ('patients','paths')}


def main():
    csv.field_size_limit(10**8)
    g = rows('genetics_photos_metdata_top10dx.csv')
    front = [r for r in g if r['EVA02_Organ'] == FRONT]
    allstats, fs = image_stats(g), image_stats(front)
    print('genetics_metadata',public_stats(allstats),'frontal',public_stats(fs),flush=True)
    cf_old=set(); cf_paths=set()
    with (HERE/'labels.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if mrnkey(r['mrn']):cf_old.add(mrnkey(r['mrn']))
            cf_paths.add(r['image_path'])
    cf_latest=set()
    with (ROOT/'complete_full_craniofacial_metadata_updated_03152026.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['EVA02_Organ']==FRONT and mrnkey(r['norm_mrn']):
                cf_latest.add(mrnkey(r['norm_mrn']))
    expert=rows('expert_chop_disease_mondo_labels.tsv')
    exby=collections.defaultdict(list)
    for r in expert:exby[mrnkey(r['MRN'])].append(r)
    matched=fs['patients'] & set(exby)
    mondo=parse_obo(HERE/'ontology/mondo.obo')
    patient_mondo=collections.defaultdict(set)
    parse_stats=collections.Counter()
    genes=set();with_genes=set();source_counts=collections.Counter()
    for m in matched:
        for r in exby[m]:
            source_counts[clean(r['data'])]+=1
            if clean(r.get('Gene')):
                with_genes.add(m)
                genes.add(clean(r['Gene']))
            raw=clean(r.get('mondo'))
            if not raw:
                parse_stats['missing_mondo']+=1
                continue
            try:
                ids=ast.literal_eval(raw)
                if not isinstance(ids,list) or not all(isinstance(i,str) for i in ids):
                    raise ValueError('not a list of strings')
            except (ValueError,SyntaxError):
                parse_stats['invalid_list']+=1
                continue
            for mid in ids:
                if mid not in mondo:
                    parse_stats['unknown_mondo_ids']+=1
                elif mondo[mid].obsolete:
                    parse_stats['obsolete_mondo_ids']+=1
                else:
                    patient_mondo[m].add(mid)
    concept_pats=collections.defaultdict(set)
    for m,ids in patient_mondo.items():
        for mid in ids:concept_pats[mid].add(m)
    by_label=collections.defaultdict(lambda:dict(pats=set(),images=set(),expert=set(),genes=set()))
    for r in front:
        m=mrnkey(r['pat_mrn_id'])
        for label in r['unified_syndrome'].split('|'):
            label=label.strip()
            v=by_label[label];v['pats'].add(m);v['images'].add(r['image_path'])
            if m in matched:v['expert'].add(m)
            if m in with_genes:v['genes'].add(m)
    labels=[dict(label=k,patients=len(v['pats']),images=len(v['images']),
                 patients_in_expert_table=len(v['expert']),patients_with_gene_field=len(v['genes']))
            for k,v in sorted(by_label.items(),key=lambda kv:(-len(kv[1]['pats']),kv[0]))]
    OUT.mkdir(exist_ok=True)
    with (OUT/'genetics_frontal_labels.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(labels[0]),delimiter='\t');w.writeheader();w.writerows(labels)
    with (OUT/'genetics_frontal_expert_mondo.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['mondo_id','mondo_name','patients'])
        for mid,pats in sorted(concept_pats.items(),key=lambda kv:(-len(kv[1]),kv[0])):
            w.writerow([mid,mondo[mid].name,len(pats)])
    conf=rows('testing_data_for_MAI_121225_withConfirmation.csv')
    confirmed_pats={mrnkey(r['pat_mrn_id']) for r in conf if r['confirmed']=='True'}
    confrontal=fs['patients'] & confirmed_pats
    # Inventory existing cropped images; crop existence alone does not establish frontal view.
    crop_root=ROOT/'cropped_faces_genetics_photos'
    crop_pats=set();extensions=collections.Counter();crop_files=[];empty=0
    for entry in os.scandir(crop_root):
        if not entry.is_dir():continue
        n=0
        for f in os.scandir(entry.path):
            if f.is_file():
                extensions[Path(f.name).suffix.lower()]+=1
                if Path(f.name).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                    n+=1;crop_files.append(f.path)
        if n:crop_pats.add(mrnkey(entry.name))
        else:empty+=1
    samples=collections.Counter()
    for p in random.Random(20260928).sample(sorted(fs['paths']),min(100,len(fs['paths']))):
        samples['exists' if Path(p).is_file() else 'missing']+=1
    confidence=[]
    for r in front:
        try:confidence.append(float(r['EVA02_Confidence']))
        except ValueError:pass
    confidence.sort()
    overlap=fs['patients'] & (cf_old|cf_latest)
    raw_22q_deletion={mrnkey(r['pat_mrn_id']) for r in front
                      if '22q11.2 deletion syndrome' in r['unified_syndrome'].split('|')}
    result=dict(
        scope='Audited genetics_photos_metdata_top10dx.csv, not the entire raw Genetics-Patient-Photos archive. Frontal means classifier label, not manual verification.',
        all_metadata=public_stats(allstats),frontal=public_stats(fs),
        frontal_label_count=len(labels),frontal_label_patient_thresholds={str(n):sum(x['patients']>=n for x in labels) for n in [5,10,20,50,100]},
        organ_counts=dict(collections.Counter(r['EVA02_Organ'] for r in g)),
        frontal_confidence=dict(n=len(confidence),min=confidence[0],median=confidence[len(confidence)//2],max=confidence[-1]),
        overlap=dict(frontal_patients_in_old_craniofacial=len(fs['patients']&cf_old),
                     frontal_patients_in_latest_craniofacial=len(fs['patients']&cf_latest),
                     frontal_patients_in_either_craniofacial=len(overlap),
                     frontal_patients_absent_from_both=len(fs['patients']-overlap),
                     exact_frontal_image_path_overlap=len(fs['paths']&cf_paths)),
        expert_link=dict(expert_table_rows=len(expert),unique_expert_mrns=len(exby),
                         matched_frontal_patients=len(matched),with_gene_field=len(with_genes),
                         with_valid_mondo=len(patient_mondo),unique_mondo_ids=len(concept_pats),
                         patients_with_multiple_mondo_ids=sum(len(v)>1 for v in patient_mondo.values()),
                         parse_status=dict(parse_stats),source_values=dict(source_counts),
                         duplicated_expert_mrns=sum(len(v)>1 for v in exby.values()),
                         fields=list(expert[0]),
                         warning='Gene names, source labels and testing recommendations do not establish molecular confirmation. No variant-level result or pathogenicity field in this table.'),
        prior_confirmed_table=dict(rows=len(conf),unique_confirmed_patients=len(confirmed_pats),
                                   source_counts=dict(collections.Counter(r['source'] for r in conf)),
                                   overlap_genetics_frontal_patients=len(confrontal),
                                   exact_genetics_frontal_image_overlap=len({r['image_path'] for r in conf}&fs['paths']),
                                   warning='Existing test/confirmation subset: preserve provenance and quarantine pending prior-use review; not a newly untouched test set.'),
        crops=dict(image_files=len(crop_files),patients_with_images=len(crop_pats),empty_patient_dirs=empty,
                   extensions=dict(extensions),frontal_patients_with_crop_directory=len(fs['patients']&crop_pats),
                   warning='Counts are for the entire crop directory, not a verified frontal subset or one-to-one crop linkage.'),
        frontal_original_path_sample=dict(samples),
        additional_review=dict(
            genetics_frontal_expert_linked_absent_from_both_cf_versions=len(matched-overlap),
            raw_22q_deletion_vs_expert_duplication_patients=sum(
                'MONDO:0012020' in patient_mondo.get(m,set()) for m in raw_22q_deletion)),
        notes=['Both image cohorts are from CHOP. Different departments do not constitute independent institutions.',
               'Patient overlap must be excluded across every training, pretraining, gallery and test partition as appropriate.',
               'Label and MONDO counts allow multiple diagnoses per patient; not mutually exclusive confirmed disease counts.'])
    (OUT/'genetics_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    print('TOP_FRONTAL_LABELS',json.dumps(labels[:15]),flush=True)


if __name__=='__main__':main()

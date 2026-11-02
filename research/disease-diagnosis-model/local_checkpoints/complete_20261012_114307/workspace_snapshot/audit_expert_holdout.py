"""Read-only expert-label audit and conservative patient/image reservation."""
import ast, collections, csv, hashlib, json, re
from pathlib import Path
from audit_metadata_sources import mrnkey, clean
from ontology import parse_obo, normstr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'expert_holdout_audit'
SOURCE = Path('/controlled-data/craniofacial_data/expert_chop_disease_mondo_labels.tsv')
MED = Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs')

def read(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))

def write(name, rows, fields):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader(); w.writerows(rows)

def mids(x):
    return set(re.findall(r'\bMONDO:\d{7}(?!\d)', x or ''))

def main():
    OUT.mkdir(exist_ok=True)
    raw = read(SOURCE)
    terms = parse_obo(ROOT / 'ontology/mondo.obo')
    seed = {mrnkey(r['MRN']) for r in raw if clean(r['MRN'])}
    manual_path = MED / 'CHOP_Diagnosis/expert_labels_manual_fix_20260511.tsv'
    manual = collections.defaultdict(set)
    for r in read(manual_path): manual[mrnkey(r['MRN'])].update(mids(r.get('mondo')))
    counts = collections.Counter(); sources = collections.Counter(); audit = []
    disease = collections.defaultdict(set)
    for i, r in enumerate(raw, 2):
        m = mrnkey(r['MRN']); ids = mids(r.get('mondo')); flags = []
        # Preserve malformed source rows; do not silently shift a diagnosis ID.
        if any(v is None for v in r.values()) or None in r: flags.append('ragged_source_row')
        if not ids: flags.append('missing_mondo')
        if not clean(r.get('Diagnosis')): flags.append('missing_diagnosis_text')
        if len(ids) > 1: flags.append('multiple_mondo_requires_adjudication')
        try:
            parsed = ast.literal_eval(r.get('mondo') or '[]')
            if not isinstance(parsed, list) or any(not isinstance(v, str) or not re.fullmatch(r'MONDO:\d{7}', v) for v in parsed):
                flags.append('invalid_mondo_cell')
        except (SyntaxError, ValueError): flags.append('invalid_mondo_cell')
        if any(d not in terms for d in ids): flags.append('id_missing_from_local_ontology')
        if any(d in terms and terms[d].obsolete for d in ids): flags.append('obsolete_mondo')
        exact = any(normstr(r.get('Diagnosis') or '') in {normstr(terms[d].name or '')} | {normstr(s) for s, scope in terms[d].synonyms if scope == 'EXACT'} for d in ids if d in terms)
        if ids and not exact: flags.append('no_whole_text_exact_name_match_review_only')
        if m in manual and ids != manual[m]: flags.append('manual_fix_mondo_disagreement')
        src = [s.strip() for s in re.split('[,;]', r.get('data') or '') if s.strip()]
        if 'genetest' in src: flags.append('genetest_source_recommended_phase_needs_confirmation')
        if re.search(r'\b(suspect\w*|possible|probable|rule out|carrier|vus)\b', r.get('Diagnosis') or '', re.I): flags.append('diagnosis_assertion_review')
        counts.update(flags); sources.update(src)
        for d in ids: disease[d].add(m)
        audit.append(dict(source_row=i, mrn=m, source_tags=r.get('data') or '', gene=r.get('Gene') or '', diagnosis_raw=r.get('Diagnosis') or '', mondo_raw=r.get('mondo') or '', mondo_ids='|'.join(sorted(ids)), mondo_names='|'.join(terms[d].name or '' for d in sorted(ids) if d in terms), issues='|'.join(flags), clinical_adjudication='pending', reservation='holdout_not_approved_test'))
    write('label_audit.tsv', audit, list(audit[0]))
    write('label_review_queue.tsv', [r for r in audit if r['issues']], list(audit[0]))
    write('disease_counts.tsv', [dict(mondo_id=d, name=terms[d].name if d in terms else '', patients=len(ms)) for d, ms in sorted(disease.items(), key=lambda x: -len(x[1]))], ['mondo_id','name','patients'])
    images = read(ROOT / 'data_release_audit/image_manifest.tsv')
    def identities(r):
        if not r['source'].startswith('CHOP'): return set()
        return {mrnkey(x) for k in ['mrn','original_mrn','latest_norm_mrn','identity_candidates'] for x in (r.get(k) or '').split('|') if clean(x)}
    idsets = [identities(r) for r in images]
    protected = set(seed); hashes = set(); paths = set()
    while True:
        before = (len(protected), len(hashes), len(paths))
        for r, ids in zip(images, idsets):
            h = r.get('crop_sha256') or ''
            rp = {str(Path(r[k])) for k in ['image_path','cropping_path'] if r.get(k)}
            if ids & protected or (h and h in hashes) or rp & paths:
                protected.update(ids); paths.update(rp)
                if h: hashes.add(h)
        if before == (len(protected), len(hashes), len(paths)): break
    reserved = [r for r, ids in zip(images, idsets) if ids & protected or r.get('crop_sha256') in hashes or any(str(Path(r[k])) in paths for k in ['image_path','cropping_path'] if r.get(k))]
    strict = read(ROOT / 'data_release_audit/chop_patient_consistent_candidates.tsv')
    held = [r for r in strict if mrnkey(r['mrn']) in protected or r['crop_sha256'] in hashes]
    rest = [r for r in strict if mrnkey(r['mrn']) not in protected and r['crop_sha256'] not in hashes]
    write('reserved_image_manifest.tsv', reserved, list(images[0]))
    write('reserved_conservative_candidates.tsv', held, list(strict[0]))
    write('remaining_ssl_candidates_pending_qc.tsv', rest, list(strict[0]))
    write('protected_mrns.tsv', [dict(mrn=m, reason='source_label_patient' if m in seed else 'identity_or_exact_copy_link') for m in sorted(protected)], ['mrn','reason'])
    write('blocked_paths.tsv', [dict(path=p) for p in sorted(paths)], ['path'])
    write('blocked_crop_hashes.tsv', [dict(sha256=h) for h in sorted(hashes)], ['sha256'])
    historical = collections.Counter(); old_patients = set(); old_rows = 0
    for r in read(ROOT / 'labels.tsv'):
        if mrnkey(r['mrn']) in protected or any(str(Path(r[k])) in paths for k in ['image_path','cropping_path'] if r.get(k)):
            old_rows += 1; old_patients.add(mrnkey(r['mrn'])); historical[r['split']] += 1
    bridge = read(ROOT / 'data_release_audit/patient_manifest.tsv')
    selected = [r for r in bridge if r['mrn'] in seed]
    held_ids = {r['mrn'] for r in held if r['mrn'] in seed}
    held_diseases = collections.Counter(d for r in audit if r['mrn'] in held_ids for d in r['mondo_ids'].split('|') if d)
    combined = MED / 'CHOP_Diagnosis/expert_chop_disease_labels_combined_20260511.tsv'
    inputs = [SOURCE, combined, manual_path, ROOT/'ontology/mondo.obo', ROOT/'labels.tsv', ROOT/'data_release_audit/image_manifest.tsv', ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv', ROOT/'data_release_audit/patient_manifest.tsv', MED/'Arcus_old/Processing.ipynb', MED/'CHOP_Diagnosis/Untitled.ipynb', MED/'CHOP_Diagnosis/Diagnosis_Prep.ipynb']
    versions = {str(p): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs}
    (OUT/'input_versions.json').write_text(json.dumps(versions, indent=2)+'\n')
    summary = dict(source_rows=len(raw), unique_source_mrns=len(seed), duplicate_mrn_rows=len(raw)-len(seed), source_identical_to_combined=SOURCE.read_bytes()==combined.read_bytes(), source_tags=dict(sources), label_flags=dict(counts), rows_with_mondo=sum(bool(mids(r.get('mondo'))) for r in raw), unique_mondo_ids=len(disease), known_obsolete_ids=sorted(d for d in disease if d in terms and terms[d].obsolete), unknown_ids=sorted(d for d in disease if d not in terms), protected_mrns=len(protected), extra_identity_linked_mrns=len(protected-seed), reserved_image_rows=len(reserved), reserved_by_source=dict(collections.Counter(r['source'] for r in reserved)), source_patients_in_broad_candidates=len(selected), source_patients_in_conservative_candidates=len({r['mrn'] for r in held if r['mrn'] in seed}), reserved_conservative_crops=len(held), reserved_conservative_patients=len({r['mrn'] for r in held}), remaining_conservative_crops=len(rest), remaining_conservative_patients=len({r['mrn'] for r in rest}), overlap_ian_patients=sum(r['expert_review_record']=='True' for r in selected), historical_label_rows_reserved=old_rows, historical_label_patients_reserved=len(old_patients), historical_split_rows=dict(historical), final_approved_test_patients=None)
    assert not ({r['mrn'] for r in rest} & protected)
    assert not ({r['crop_sha256'] for r in rest} & hashes)
    assert len(rest)+len(held)==len(strict)
    assert all(not any(str(Path(r[k])) in paths for k in ['image_path','cropping_path']) for r in rest)
    summary['partition_checks_passed']=True
    summary['reserved_source_mondo_ids']=len(held_diseases)
    summary['reserved_source_diseases_at_least_5_patients']=sum(n>=5 for n in held_diseases.values())
    summary['reserved_source_top_diseases']=[dict(mondo_id=d,name=terms[d].name if d in terms else '',patients=n) for d,n in held_diseases.most_common(10)]
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

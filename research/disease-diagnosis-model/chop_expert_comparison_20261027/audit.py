"""Local-only expert identity, current crop provenance, and training exposure audit."""
import csv,collections,hashlib,hmac,json,os,re,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
V1=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
RUN=Path('/opt/craniofacial/SyndroDx-python-slurm/face_ssl_annotation/run_20261012_baed142d')
def load(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def save(p,d):
    p.write_text(json.dumps(d,indent=2)+'\n');os.chmod(p,0o600)
def norm(x):
    x=(x or '').strip();x=re.sub(r'\.0$','',x);return (x.lstrip('0') or '0') if x.isdigit() else x
def main():
    os.umask(0o077);sys.path.insert(0,str(RUN))
    from annotation_pipeline import references,canonical_path
    config=load(RUN/'config.json');salt=(RUN/'direct_identity_salt.bin').read_bytes()
    identity=load(V1/'dataset_release/patient_identity_plan.private.json')
    labels=collections.defaultdict(set);bad=set()
    fatal={'ragged_source_row','missing_mondo','invalid_mondo_cell','id_missing_from_local_ontology','obsolete_mondo','manual_fix_mondo_disagreement','diagnosis_assertion_review','genetest_source_recommended_phase_needs_confirmation'}
    for r in rows(PROJECT/'expert_holdout_audit/label_audit.tsv'):
        m=norm(r['mrn']);labels[m].update(x for x in r['mondo_ids'].split('|') if x)
        if fatal&set(r['issues'].split('|')):bad.add(m)
    path_ids=collections.defaultdict(set)
    for r in rows(PROJECT/'expert_holdout_audit/reserved_image_manifest.tsv'):
        ids={norm(x) for k in ['mrn','original_mrn','latest_norm_mrn','identity_candidates'] for x in r.get(k,'').split('|') if x}
        expert=ids&labels.keys()
        if not expert:continue
        for k in ['image_path','cropping_path']:
            if r.get(k):path_ids[str(canonical_path(r[k],config))].update(expert)
    key_ids=collections.defaultdict(set)
    for r in rows(Path('/controlled-data/chop_data_metadata/metadata_internal.tsv')):
        pseudo=(r.get('pseudo_mrn') or r.get('psedudo_mrn') or '').strip()
        if not pseudo:continue
        key=hmac.new(salt,('patient:'+pseudo).encode(),hashlib.sha256).hexdigest()
        for p in references(r.get('image_path')):key_ids[key].update(path_ids.get(str(canonical_path(p,config)),set()))
    release=load(V1/'dataset_release/release.json');training=[]
    for split,spec in release['manifests'].items():
        if split=='train':training+=rows(spec['path'])
    groups={r['group_id'] for r in training};keys={r['patient_key'] for r in training}
    hashes={r.get(k) for r in training for k in ['crop_sha256','pixel_sha256','source_sha256'] if r.get(k)}
    paths={r.get(k) for r in training for k in ['crop_path','original_path'] if r.get(k)}
    qc={}
    for p in (V1/'independent_qc').glob('*.sqlite'):
        db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        for encoded, in db.execute('SELECT record FROM reviews'):
            q=json.loads(encoded)
            if q.get('photo_id') in qc:raise RuntimeError('duplicate_QC_photo')
            qc[q['photo_id']]=q
        db.close()
    admitted=[];counts=collections.Counter();exposed=set()
    for r in rows(RUN/'cropped_candidates.tsv'):
        ids=key_ids.get(r['patient_key'],set())
        if not ids:continue
        counts['linked_candidate_images']+=1
        if len(ids)!=1:counts['ambiguous_identity_images']+=1;continue
        m=next(iter(ids));plan=identity['patients'].get(r['patient_key'],{});group=plan.get('group_id')
        q=qc.get(r['photo_id'],{})
        leak=r['patient_key'] in keys or group in groups or any(q.get(k) in hashes for k in ['source_sha256','crop_sha256','pixel_sha256'] if q.get(k)) or r['original_path'] in paths or r['crop_path'] in paths
        if leak:exposed.add(m);counts['known_training_exposure_images']+=1;continue
        if q.get('state')!='pass' or r['view'] not in ['frontal','profile','oblique']:counts['QC_not_passed_images']+=1;continue
        if m in bad or len(labels[m])!=1:counts['label_assertion_or_mapping_review_images']+=1;continue
        p=Path(r['crop_path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['crop_sha256']:counts['crop_integrity_failures']+=1;continue
        admitted.append(dict(patient=m,group_id=group,patient_key=r['patient_key'],label=next(iter(labels[m])),crop_path=str(p),crop_sha256=r['crop_sha256'],pixel_sha256=q.get('pixel_sha256'),source_sha256=q.get('source_sha256'),view=r['view']))
    admitted=[r for r in admitted if r['patient'] not in exposed]
    save(ROOT/'private/eligible.json',dict(rows=admitted))
    save(ROOT/'audit_summary.json',dict(state='audited',counts=dict(counts),eligible_images=len(admitted),eligible_patients=len({r['patient'] for r in admitted}),eligible_diseases=len({r['label'] for r in admitted}),
        known_current_CHOP_training_exposed_patients=len(exposed),clinical_expert_labels_still_pending_adjudication=True,
        cross_GMDB_identity_and_near_duplicate_audit='pending; do not release independent test results yet',
        historical_unrelated_models_not_assumed_checkpoint_ancestors=True,patient_records_remain_on_HPC=True))
if __name__=='__main__':main()

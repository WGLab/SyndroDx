"""Lock genuine paired diagnosis SFT and matched HPO-only control inputs."""
import csv,os
from runtime import *
def main():
    os.umask(0o077)
    if not load(WORK/'training_complete.json').get('visual_conditioning_gate'):raise RuntimeError('visual_alignment_gate_required')
    patients=load(WORK/'private/training_index.json')['patients'];terms=load(PREVIOUS/'private/ontology.json')['terms']
    images=load(GMDB/'private/records.private.json')['rows']
    metadata=Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/image_metadata_v1.1.0.tsv')
    with metadata.open() as f:meta={r['image_id']:r for r in csv.DictReader(f,delimiter='\t')}
    names={}
    for r in patients:
        ms=[meta[images[i]['image_id']] for i in r['image_indices']]
        if any(m['patient_id']!=r['subject'] or m['internal_syndrome_id']!=r['label'] for m in ms):raise RuntimeError('diagnosis_pair_mismatch')
        choices={m['internal_syndrome_name'].strip() for m in ms}
        if len(choices)!=1 or not next(iter(choices)):raise RuntimeError('ambiguous_diagnosis_name')
        name=next(iter(choices))
        if r['label'] in names and names[r['label']]!=name:raise RuntimeError('inconsistent_disease_vocabulary')
        names[r['label']]=name
        r['diagnosis']=name
        r['clinical_input']='Present findings: '+'; '.join(terms[x]['name'] for x in r['all_present_hpo'])+'.'
        r['clinical_input']+=' Explicitly absent findings: '+('; '.join(terms[x]['name'] for x in r['absent_hpo']) if r['absent_hpo'] else 'none recorded')+'. Unrecorded findings are unknown.'
    classes=sorted({r['label'] for r in patients if r['split']=='train'})
    # Class vocabulary is derived from training patients only.
    vocab=[dict(label=x,name=names[x]) for x in classes]
    if len({r['name'] for r in vocab})!=len(vocab):raise RuntimeError('nonunique_disease_names')
    save(WORK/'private/sft_patients.json',dict(patients=patients,vocabulary=vocab))
    save(WORK/'sft_data_receipt.json',dict(state='complete',train_patients=sum(r['split']=='train' for r in patients),
        validation_patients=sum(r['split']=='validation' and r['label'] in classes for r in patients),
        validation_unseen_disease_patients=sum(r['split']=='validation' and r['label'] not in classes for r in patients),
        classes=len(classes),patient_sha256=digest(WORK/'private/sft_patients.json'),patient_split_unchanged=True,
        clinical_test_used=False,CHOP_candidate_HPO_admitted=False,diagnosis_targets='original GMDB patient diagnosis; no generated rationale',updated_at_utc=now()))
    save(WORK/'sft_protocol.json',dict(seed=20261027,arms=['fusion','hpo'],max_epochs=5,patience=2,min_epochs=3,
        per_gpu_batch=2,accumulation=4,lr=0.00002,connector_lr=0.00001,weight_decay=.01,
        lora_rank=8,lora_alpha=16,lora_modules=['q_proj','v_proj'],max_sequence=2048,
        checkpoint_selection='closed-set GMDB development target NLL',
        ranking='constrained beam search over training disease names; beam width 10; not exhaustive class likelihood',
        matched_patients=True,expert_test_used=False,vision_backbone_frozen=True,
        initialization_sha256=load(WORK/'best_checkpoint.json')['sha256'],
        next_step='compare disease macro top1/top5/top10 against matched HPO-only, frozen native Qwen and existing fusion; no clinical superiority claim from NLL'))
    print('Diagnosis SFT release and HPO-only control sealed; no patient records emitted.')
if __name__=='__main__':main()

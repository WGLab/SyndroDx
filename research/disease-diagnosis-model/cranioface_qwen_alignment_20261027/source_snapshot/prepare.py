"""Reuse the frozen patient-disjoint GMDB release; no clinical records are printed."""
import os
from runtime import *
def main():
    os.umask(0o077)
    receipt=load(PREVIOUS/'data_receipt.json')
    p=PREVIOUS/'private/patients.json'
    if receipt.get('state')!='complete' or digest(p)!=receipt['patient_file_sha256']:raise RuntimeError('unsealed_pair_release')
    old=load(p)['patients'];patients=[]
    for r in old:
        if not r['hpo'] or set(r['all_present_hpo'])&set(r['absent_hpo']):raise RuntimeError('invalid_patient_assertions')
        patients.append({k:r[k] for k in ['subject','split','image_indices','hpo','text','label','all_present_hpo','absent_hpo']})
    sets={s:{r['subject'] for r in patients if r['split']==s} for s in ['train','validation']}
    if sets['train']&sets['validation']:raise RuntimeError('patient_overlap')
    save(WORK/'private/patients.json',dict(patients=patients))
    cfg=load(MODEL/'config.json');idx=load(MODEL/'model.safetensors.index.json')
    files=set(idx['weight_map'].values())
    if not files or not all((MODEL/f).is_file() for f in files):raise RuntimeError('incomplete_Qwen_checkpoint')
    save(WORK/'model_receipt.json',dict(model_id='Qwen/Qwen3-VL-32B-Instruct',revision=MODEL.name,path=str(MODEL),
        config_sha256=digest(MODEL/'config.json'),index_sha256=digest(MODEL/'model.safetensors.index.json'),
        total_bytes=sum((MODEL/f).stat().st_size for f in files),language_hidden_size=cfg['text_config']['hidden_size'],
        native_Qwen_vision_branch_used_in_stage1=False,training='frozen_language_backbone_plus_new_TransFace_connector'))
    save(WORK/'data_receipt.json',dict(state='complete',source_release_sha256=digest(PREVIOUS/'data_receipt.json'),
        patients_sha256=digest(WORK/'private/patients.json'),counts=receipt['counts'],
        patient_split_disjoint=True,expert_CHOP_training=False,BCH_training=False,
        internal_CHOP_HPO_admission='separate_audit_pending',public_patient_cases_admitted=False,
        targets='documented_patient_craniofacial_HPO_names; association, not expert image-localized truth',
        diagnosis_in_prompt=False,HPO_in_prompt=False,clinical_test_used=False,updated_at_utc=now()))
    protocol=dict(seed=20261027,stage='frozen_vision_frozen_LLM_connector_alignment',
        language_model='Qwen/Qwen3-VL-32B-Instruct',vision='CHOP_identity_CPT_TransFace_L_step2500',
        vision_training_history='patient-pseudonym-supervised ArcFace CPT; not label-free SSL',
        connector='2-layer learned-query cross-attention resampler, 32 queries, genuine 144 patch tokens plus global embedding',
        queries=32,width=768,heads=8,layers=2,per_gpu_batch=2,accumulation=4,
        max_epochs=8,min_epochs=3,patience=3,learning_rate=0.0001,weight_decay=0.01,
        max_sequence=384,validation_every_epochs=1,gradient_checkpointing=True,
        checkpoint_selection='GMDB development patient-average target NLL',
        transfer_gate='paired-minus-shuffled NLL bootstrap upper 95% CI < 0 and improvement over initial NLL',
        expert_test_not_used_for_tuning=True,all_patient_processing_offline=True,
        followon='save connector and evaluate; do not automatically label clinical reasoning established')
    if (WORK/'protocol.json').exists() and load(WORK/'protocol.json')!=protocol:raise RuntimeError('protocol_already_frozen')
    save(WORK/'protocol.json',protocol)
    print('GMDB patient-disjoint alignment release sealed; all patient records remain local.')
if __name__=='__main__':main()

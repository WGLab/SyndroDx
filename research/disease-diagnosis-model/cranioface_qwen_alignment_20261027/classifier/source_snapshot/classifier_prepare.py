"""Lock classifier labels, inputs and a matched two-stage training protocol."""
from runtime import *
def main():
    r=load(WORK/'sft_data_receipt.json')
    if r.get('state')!='complete' or digest(WORK/'private/sft_patients.json')!=r['patient_sha256']:raise RuntimeError('sealed_patient_release_required')
    protocol=dict(seed=20261028,arms=['fusion','hpo'],classes=r['classes'],train_patients=r['train_patients'],validation_patients=r['validation_patients'],
        patient_sha256=r['patient_sha256'],feature_receipt_sha256=digest(WORK/'feature_receipt.json'),
        aligned_connector=load(WORK/'best_checkpoint.json'),language_initialization='original Qwen3-VL-32B-Instruct; no disease-generation SFT',
        HPO_control_sha256=digest(WORK/'hpo_control_protocol.json'),pooling='last non-padding token after input-only clinical prompt',
        head='LayerNorm(5120), Dropout(0.2), Linear(5120,216)',loss='cross entropy with sqrt inverse training-patient class-frequency weights',
        head_probe_epochs=50,head_probe_lr=.001,head_probe_global_batch=512,
        joint_max_epochs=50,joint_min_epochs=15,joint_patience=10,joint_global_batch=128,per_gpu_batch=2,
        lora_rank=8,lora_alpha=16,lora_modules=['q_proj','v_proj'],lora_lr=.000003,connector_lr=.0001,head_lr=.0001,
        scheduler='ReduceLROnPlateau on development cross entropy, factor 0.5, patience 3',
        checkpoint_selection='development disease-macro Top5, with Top1 tie-break',
        expert_test_used=False,BCH_used=False,CHOP_candidate_HPO_admitted=False,
        completion_email='aggregate Top1/5/10 for frozen probe and joint arms, paired differences and bootstrap intervals; no patient data',
        source='unchanged GMDB paired training/development release')
    path=WORK/'classifier/protocol.json'
    if path.exists() and load(path)!=protocol:raise RuntimeError('classifier_protocol_already_locked')
    save(path,protocol);print('Classifier protocol sealed; patient records remain local.')
if __name__=='__main__':main()

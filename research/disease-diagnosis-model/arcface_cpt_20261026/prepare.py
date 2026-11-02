"""Verify exact weight conversion and clinical split; print aggregates only."""
import os
import sys
import torch
import numpy as np
from control import WORK, V1, BASE, ONNX, load, save, digest, now
sys.path.insert(0, str(WORK/'dependencies'))
sys.path.append(str(WORK.parent/'arcface_glint360k_eval_20261026/dependencies'))
from model import backbone, fixed_batchnorm
from data import verify_manifest, validate_split_rows

def main():
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    expected = '4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf'
    if digest(ONNX) != expected:
        raise RuntimeError('public_model_checksum_mismatch')
    release = load(V1/'dataset_release/release.json')
    if release['state'] != 'ssl_data_ready' or not release['patient_split_disjoint'] or not release['protected_pools_excluded']:
        raise RuntimeError('approved_clinical_release_required')
    parts = [verify_manifest(release['manifests'][s]) for s in ('train','validation','test')]
    counts = validate_split_rows(*parts)
    # Verify development cohort is confined to the released train and validation pools.
    from data import evaluation_cohort
    records, _ = evaluation_cohort()
    permitted = {split: {r['group_id'] for r in rows} for split,rows in zip(('train','validation','test'),parts)}
    if any(r['group_id'] not in permitted[r['split']] for r in records):
        raise RuntimeError('development_cohort_outside_release')
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    options.log_severity_level=3
    session = ort.InferenceSession(str(ONNX), sess_options=options, providers=['CPUExecutionProvider'])
    model = backbone().eval()
    value = torch.rand(4,3,112,112)*2-1
    with torch.no_grad():
        actual = model(value).numpy()
    reference = session.run(None, {session.get_inputs()[0].name: value.numpy()})[0]
    cosine = np.sum(actual*reference,1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(reference,axis=1))
    relative = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    if cosine.min() < .99999 or relative > 1e-4:
        raise RuntimeError('converted_pretrained_weights_not_equivalent')
    model.train(); fixed_batchnorm(model)
    target = torch.randn(2,512)
    output = model(value[:2])
    (torch.nn.functional.normalize(output,dim=-1)*target).mean().backward()
    learned = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _,p in learned):
        raise RuntimeError('backbone_gradient_coverage_failed')
    receipt = dict(state='passed', model_sha256=expected, source='official InsightFace Glint360K R100',
        backbone_parameters=sum(p.numel() for p in model.parameters()), output_dimension=512,
        minimum_conversion_cosine=float(cosine.min()), relative_l2_error=relative,
        all_trainable_parameters_have_finite_gradients=True, batchnorm_running_stats_frozen=True,
        prelu_slopes_restored_as_parameters=True, synthetic_inputs_only=True, checked_at_utc=now())
    save(WORK/'conversion_receipt.json', receipt)
    validation_plan=load(BASE/'validation/plan.json')
    protocol = dict(state='prepared', model='ArcFace iResNet-100 / Glint360K',
        model_selection='largest directly downloadable verified Glint360K model; larger R200/ViT-L official archive requires access',
        backbone_parameters=receipt['backbone_parameters'], model_sha256=expected,
        train_manifest=release['manifests']['train'], data=counts, clinical_only=True,
        reference_faces_used=False, gmdb_used_for_training_or_online_selection=False,
        hpo_or_disease_labels_used_in_ssl=False, training_patient_balanced=True, training_view_balanced=True,
        validation_inputs_sha256=validation_plan['inputs_sha256'], validation_labels_sha256=validation_plan['labels_sha256'],
        method='DINO global teacher/student self-distillation on pretrained ArcFace; no iBOT or patch loss',
        output_dim=65536, input_size=112, input_normalization='RGB [-1,1]',
        global_views=2, crop_scale=[.85,1.0], crop_ratio=[.95,1.05], horizontal_flip=False,
        teacher_momentum=.996, teacher_temperature_start=.04, teacher_temperature_end=.07,
        teacher_temperature_warmup_steps=1000, center_momentum=.9, student_temperature=.1,
        backbone_learning_rate=1e-5, head_learning_rate=1e-4, backbone_weight_decay=.01,
        head_weight_decay=.04, head_only_steps=100, backbone_ramp_steps=100,
        freeze_last_layer_steps=10, optimizer='AdamW', gradient_clip_norm=3.,
        batchnorm_running_stats='frozen; affine parameters trainable',
        checkpoint_every_steps=50, validation_initial=True, validation_first_steps=[100,250], validation_every_steps=500,
        validation_teacher_embedding=True, diagnostic_linear_probe_only=True, inference_classification_head=False,
        development_guard='pause for two consecutive statistically supported retrieval declines or feature collapse; preserve checkpoints',
        minimum_runtime_not_inherited_from_stopped_dino_run=True,
        training_duration='continue from checkpoints; first assessment at 100 and 250 steps; formal stop requires user decision or health guard',
        reshard_policy='same optimizer/teacher/center and cumulative stream; no LR scaling on resize',
        private_images_and_identifiers_remain_on_hpc=True, trackers_disabled=True, created_at_utc=now())
    save(WORK/'protocol.json', protocol)
    print({'state':'prepared','backbone_parameters':receipt['backbone_parameters'],
           'train_images':counts['train_images'],'train_patients':counts['train_patients'],
           'conversion_min_cosine':receipt['minimum_conversion_cosine']})

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save(WORK/'preparation_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise

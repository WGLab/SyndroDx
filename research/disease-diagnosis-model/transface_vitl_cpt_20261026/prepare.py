"""Check public weights and numerical/gradient compatibility using synthetic inputs."""
import gc
import os
import torch
from control import WORK, WEIGHTS, V1, load, save, digest, now
from model import author_backbone, fixed_batchnorm
from data import verify_manifest,validate_split_rows,evaluation_cohort

def main():
    os.umask(0o077);torch.set_num_threads(4);torch.manual_seed(20261026)
    p=load(WORK/'protocol.json')
    if digest(WEIGHTS)!=p['model_sha256']:raise RuntimeError('public_model_digest_mismatch')
    release=load(V1/'dataset_release/release.json')
    if release['state']!='ssl_data_ready' or not release['patient_split_disjoint'] or not release['protected_pools_excluded']:
        raise RuntimeError('approved_clinical_release_required')
    parts=[verify_manifest(release['manifests'][s]) for s in ('train','validation','test')]
    counts=validate_split_rows(*parts)
    records,_=evaluation_cohort()
    permitted={s:{r['group_id'] for r in rows} for s,rows in zip(('train','validation','test'),parts)}
    if any(r['group_id'] not in permitted[r['split']] for r in records):raise RuntimeError('development_outside_release')
    value=torch.rand(2,3,112,112)*2-1
    original=author_backbone(False).eval()
    with torch.no_grad():reference=original(value)[0]
    count=sum(v.numel() for v in original.parameters())
    del original;gc.collect()
    model=author_backbone(True).eval()
    with torch.no_grad():actual=model(value)[0]
    cosine=torch.nn.functional.cosine_similarity(actual,reference,dim=-1)
    relative=torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference)
    if float(cosine.min())<.99999 or float(relative)>1e-4:raise RuntimeError('sdpa_reference_equivalence_failed')
    model.train();fixed_batchnorm(model)
    output=model(value)[0]
    (torch.nn.functional.normalize(output,dim=-1)*torch.randn(2,512)).mean().backward()
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):
        raise RuntimeError('trainable_gradient_coverage_failed')
    p.update(backbone_parameters=count,data=counts)
    save(WORK/'protocol.json',p)
    receipt=dict(state='passed',model=p['model'],model_sha256=p['model_sha256'],
        backbone_parameters=count,output_dimension=512,strict_state_dict_load=True,
        minimum_conversion_cosine=float(cosine.min()),relative_l2_error=float(relative),
        all_trainable_parameters_have_finite_gradients=True,synthetic_inputs_only=True,checked_at_utc=now())
    save(WORK/'conversion_receipt.json',receipt);print(receipt)

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'preparation_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise

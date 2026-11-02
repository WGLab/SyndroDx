"""Retune ridge alpha using training patients only; reuse cached development features."""
import os
import sys
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent;TRAIN=WORK.parent/'transface_vitl_cpt_20261026'
sys.path[:0]=[str(TRAIN),str(WORK.parent/'training_diagnostics_20261017')]
from control import load,save,now
from data import evaluation_cohort
from frozen_probe import evaluate_pair,aggregate_patients

def main():
    os.umask(0o077)
    save(WORK/'probe_audit_status.json',dict(state='running',images_read=False,test_used=False,updated_at_utc=now()))
    records,labels=evaluation_cohort()
    baseline=np.load(TRAIN/'private/baseline_features.npy',allow_pickle=False)
    current=np.empty_like(baseline)
    for rank in range(3):
        shard=np.load(TRAIN/'private/validation_0004000_000'/f'rank_{rank:02d}.npy',allow_pickle=False)
        if shard.shape!=current[rank::3].shape:raise RuntimeError('incomplete_development_features')
        current[rank::3]=shard
    # The reused helper's legacy baseline key is renamed in the output below.
    partitions={name:{split:aggregate_patients(records,features,split) for split in ('train','validation')}
        for name,features in [('original_dinov3',baseline),('adapted',current)]}
    report=dict(state='running',checkpoint_step=4000,images_read=False,test_used=False,
        alpha_selection='five-fold CV on training patients only, separately for each encoder',
        metadata_labels_not_clinical_gold_standard=True,endpoints={})
    for category in ('disease_or_disorder','structural_diagnosis'):
        result=evaluate_pair(partitions,labels,category)
        result['models']['original_transface']=result['models'].pop('original_dinov3')
        result['models']['cpt4000']=result['models'].pop('adapted')
        report['endpoints'][category]=result
        save(WORK/'probe_audit.json',report)
    report.update(state='complete',updated_at_utc=now());save(WORK/'probe_audit.json',report)
    save(WORK/'probe_audit_status.json',dict(state='complete',images_read=False,test_used=False,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'probe_audit_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise

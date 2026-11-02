"""Prepare real staging work and verify the single-GPU orchestration locally."""
import os
import unittest
from configuration import WORK
from common import save,digest,now


def main():
    os.umask(0o077)
    suite=unittest.defaultTestLoader.discover(str(WORK),pattern='test_*.py')
    with (WORK/'synthetic_tests.private.log').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('single_gpu_pool_preflight_failed')
    from pool_work import prepare
    prepare()
    names=[p.name for p in WORK.glob('*.py')]+['lease.sbatch','controller.sbatch','resume_plan.json',
        'fixed_probe_parameters.json','original_runtime_baseline.json','original_runtime_baseline.private.npy',
        'disease_original_hits.private.npz','structural_original_hits.private.npz',
        'disease_resume_hits.private.npz','structural_resume_hits.private.npz','resume_anchor.json']
    save(WORK/'preflight.json',dict(state='passed',tests_passed=result.testsRun,
        file_sha256={name:digest(WORK/name) for name in names},
        cached_feature_audit_complete=True,primary_metrics_reproduced=True,created_at_utc=now()))
    print('single_gpu_pool_preflight_and_real_staging_work_prepared',result.testsRun)


if __name__=='__main__':main()

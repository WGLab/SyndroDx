"""Run synthetic preflight and cached-feature diagnostics before GPU admission."""
import os
import unittest
from pathlib import Path
from configuration import WORK,protocol
from common import save,load,digest,now,notify


def main():
    os.umask(0o077)
    suite=unittest.defaultTestLoader.discover(str(WORK),pattern='test_resume20.py')
    with (WORK/'synthetic_tests.private.log').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('resume20_synthetic_preflight_failed')
    from audit_probe import main as audit
    audit()
    names=[p.name for p in WORK.glob('*.py')]+['training.sbatch','diagnostic.sbatch','resume_plan.json',
        'fixed_probe_parameters.json','original_runtime_baseline.json','original_runtime_baseline.private.npy',
        'disease_original_hits.private.npz','structural_original_hits.private.npz',
        'disease_resume_hits.private.npz','structural_resume_hits.private.npz','resume_anchor.json']
    save(WORK/'protocol.json',protocol())
    save(WORK/'preflight.json',dict(state='passed',tests_passed=result.testsRun,
        file_sha256={name:digest(WORK/name) for name in names},cached_feature_audit_complete=True,
        primary_metrics_reproduced=True,created_at_utc=now()))
    notify('resume20 cached feature audit complete','Craniofacial probe diagnostic completed',
        'Cached-feature diagnostics reproduced the saved primary endpoints and recorded regularization sensitivity. Original reports are unchanged. The requested continuation remains queued behind the local preflight and resource checks. All patient data stay on HPC.')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(WORK/'cpu_job_status.json',dict(state='failed',error_type=type(error).__name__,updated_at_utc=now()))
        raise

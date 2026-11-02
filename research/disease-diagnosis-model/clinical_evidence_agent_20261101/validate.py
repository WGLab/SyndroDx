"""Record reproducible tests and a public-only local knowledge smoke check."""
import io
import unittest
from common import *
from knowledge import lookup_gene
from engine import run
from test_engine import case, observation


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('synthetic_tests_failed')
    # These are public gene symbols, not taken from any patient record.
    public_smoke = {symbol: len(lookup_gene(symbol)) for symbol in ['FGFR2', 'FBN1']}
    if not all(public_smoke.values()) or lookup_gene("' OR 1=1 --"):
        raise RuntimeError('offline_lookup_smoke_failed')
    synthetic = run(case(), lambda evidence: {'SYNTHETIC_A': 0., 'SYNTHETIC_B': 2.} if evidence else {'SYNTHETIC_A': 1., 'SYNTHETIC_B': 0.},
                    lambda action: observation(), {'HP:0000001': {'SYNTHETIC_A': .1, 'SYNTHETIC_B': .9}})
    save(ROOT / 'synthetic_replay_example.json', {'synthetic_only': True, 'not_a_clinical_example': True, 'result': synthetic})
    audit = load(ROOT / 'source_audit_receipt.json')
    pins = load(PROJECT / 'model_releases/CURRENT.json')
    if sha(pins['manifest']) != pins['sha256'] or audit['accepted_patient_observations'] != 0:
        raise RuntimeError('release_or_observation_gate_changed')
    adapter = load(ROOT / 'adapter_check_receipt.json')
    if adapter['state'] != 'passed' or adapter['adapter_code_sha256'] != sha(ROOT / 'frozen_v1.py'):
        raise RuntimeError('adapter_integration_check_missing_or_stale')
    from hpo_questions import selector
    choose = selector()
    profiles = load(ROOT / 'assets/class_hpo_associations.json')['associations']
    first = next(k for k, terms in profiles.items() if terms)
    if choose({first: 0.}, set()) is None:
        raise RuntimeError('public_hpo_question_lookup_failed')
    receipt = {'state': 'passed', 'synthetic_tests_passed': result.testsRun,
               'offline_public_gene_lookup_smoke_passed': True,
               'real_patient_readiness_replays': audit['patients'],
               'real_patient_diagnosis_metrics_available': False,
               'live_v1_inference_connected': True, 'LLM_controller_deployed': False,
               'existing_GMDB_development_adapter_check_patients': adapter['patients'],
               'public_HPO_question_lookup_passed': True,
               'source_code_sha256': {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))},
               'protocol_sha256': sha(ROOT / 'protocol.json'),
               'source_audit_sha256': sha(ROOT / 'source_audit_receipt.json'),
               'locked_v1_manifest_verified': True, 'updated_at_utc': now()}
    save(ROOT / 'validation_receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_code_sha256'}))


if __name__ == '__main__':
    guarded_main(main)

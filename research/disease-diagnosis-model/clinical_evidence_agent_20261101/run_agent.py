"""Local research agent CLI. No network or free-form text tool execution.

JSONL inputs and detailed results stay on HPC; stdout contains counts only.
"""
import argparse
from common import *
from frozen_v1 import FrozenV1
from local_pipeline import LocalPipeline
from learned_questions import current_selector


def exact_resolver(patient_key, records):
    # Inputs must be real reviewed observations. The engine checks timing,
    # provenance and polarity again; there is no public-knowledge answer fallback.
    by_term = {}
    for item in records:
        if item.get('patient_key') != patient_key:
            raise ValueError('cross_patient_answer_record')
        if item.get('kind') != 'hpo':
            continue
        code = item.get('concept_id')
        by_term.setdefault(code, []).append(item)
    def resolve(action):
        if action.get('tool') != 'request_hpo_observation': return None
        rows = by_term.get(action['concept_id'], [])
        # Ambiguous duplicate answers need source review, not arbitrary selection.
        return rows[0] if len(rows) == 1 else None
    return resolve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--purpose',choices=['development','evaluation'],default='development')
    parser.add_argument('--budget',type=int,default=5)
    args=parser.parse_args()
    if Path(args.input).resolve()==Path(args.output).resolve():raise ValueError('output_cannot_replace_input')
    if not 0<=args.budget<=5:raise ValueError('unvalidated_action_budget')
    adapter=FrozenV1(args.device); choose=current_selector(); pipeline=LocalPipeline(adapter,choose)
    policy=load(ROOT/'agent_releases/CURRENT.json')
    output=[];requests=0;variants=0
    for item in jsonl(args.input):
        case=item['case']
        resolver=exact_resolver(case['patient_key'],item.get('available_answer_records',[]))
        result=pipeline.assess(case,native512_feature=item.get('native512_feature'),
            visual_encoder_sha256=item.get('visual_encoder_sha256'),resolver=resolver,
            mode=item.get('mode','face_hpo'),max_actions=args.budget,purpose=args.purpose)
        result.update(policy_manifest_sha256=policy['sha256'],model_manifest_sha256=sha(adapter.manifest_path),
                      use_scope='research_only_not_independent_clinical_validation')
        output.append({'patient_key':case['patient_key'],'result':result})
        requests+=len(result['trace']);variants+=len(result['variant_evidence_cards'])
    write_jsonl(args.output,output)
    receipt={'state':'complete','cases':len(output),'actions':requests,'variant_review_cards':variants,
             'output_sha256':sha(args.output),'patient_records_in_stdout':False,'external_api_calls':0,
             'updated_at_utc':now()}
    save(str(args.output)+'.receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':guarded_main(main)

"""Export one genuine development replay for LOCAL viewing; no patient stdout."""
import csv
import html
import numpy as np
import torch
from common import *
from acquisition_replay import observations, initial_observations, answer, infer
from frozen_v1 import FrozenV1
from learned_questions import current_selector


def main():
    torch.set_num_threads(2)
    work=ROOT/'acquisition_policy_v2'
    source=PROJECT/'protected_visual_fusion_20261101/private/gmdb.npz'
    if sha(source)!=load(PROJECT/'protected_visual_fusion_20261101/input_receipt.json')['gmdb_sha256']:
        raise ValueError('source_mismatch')
    with np.load(source) as data:
        val=data['validation']
        # Deterministic choice independent of correctness or question outcomes.
        selected=None
        for local_row,idx in enumerate(val):
            recorded=observations(data['ids'][idx],data['states'][idx])
            if sum(v==0 for v in recorded.values())>=8:
                selected=(local_row,int(idx),recorded);break
        if selected is None:raise ValueError('no_eligible_example')
        local_row,index,recorded=selected
        feature=data['features'][index:index+1].copy();truth=int(data['labels'][index])
    adapter=FrozenV1('cpu');policy=current_selector()
    inverse={i:t for t,i in adapter.terms.items()}
    ontology=load(PROJECT/'arcface1000_hpo_alignment_20261028/private/ontology.json')['terms']
    dictionary=Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/gmdb_syndromes_v1.1.0.tsv')
    with dictionary.open() as f:
        names={str(r['syndrome_id']):r for r in csv.DictReader(f,delimiter='\t')}
    def disease(label):
        item=names.get(str(label),{})
        name=next((item[k] for k in ['syndrome_name','name','syndrome'] if item.get(k)),None)
        return {'GMDB_class':str(label),'name':name or str(label)}
    def term(term_id,state=None):
        code=inverse[int(term_id)]
        return {'HPO':code,'name':ontology.get(code,{}).get('name',code),
                'status':{0:'present',1:'explicitly absent',None:'unknown'}[state]}
    def ranking(prob):
        order=np.argsort(-prob,kind='stable')[:5]
        return [dict(rank=n+1,**disease(adapter.labels[int(i)]),model_score=float(prob[i])) for n,i in enumerate(order)]
    known=initial_observations(recorded,3,11,local_row);asked=set()
    initial=[term(i,s) for i,s in sorted(known.items())]
    prob=infer(adapter,feature,[known],inverse,'face_hpo')[0]
    before=ranking(prob);steps=[]
    for step in range(1,6):
        questions,details=policy.choose(prob[None],[set(known)|asked]);q=questions[0]
        outcome=answer(recorded,known,asked,q)
        prob=infer(adapter,feature,[known],inverse,'face_hpo')[0]
        steps.append({'step':step,'question':term(q) if q is not None else None,
            'answer':outcome,'source':'recorded patient HPO' if outcome in ['present','absent'] else 'not recorded; retained as unknown',
            'policy':policy.config['name'],'planning':details[0],'updated_top5':ranking(prob)})
    # Confirm that this example is exactly the trajectory used in the reported result.
    with np.load(work/'private/k3__seed11__face_hpo__fixed.npz') as replay:
        if not np.allclose(prob,replay['final_probabilities'][local_row],atol=1e-5,rtol=1e-4):
            raise ValueError('example_does_not_match_saved_replay')
        outcomes={'present':0,'absent':1,'unknown':2,'no_question':-1}
        for step,saved in zip(steps,replay['actions'][:,local_row]):
            if step['question'] is not None and policy.code_to_term[step['question']['HPO']]!=int(saved[0]):
                raise ValueError('example_question_drift')
            if outcomes[step['answer']]!=int(saved[1]):raise ValueError('example_answer_drift')
    example={'example_id':'LOCAL_EXAMPLE_001','real_development_case':True,
        'scope':'Retrospective HPO-masking simulation, not an actual clinical consultation',
        'selection_rule':'First development-array patient with at least eight recorded positive HPO terms; no filtering on outcome',
        'input':{'face':'Audited native512 facial feature used; image and embedding values omitted from this report',
                 'initial_HPO':initial,'initial_positive_HPO_limit':3,'question_budget':5,
                 'variant':'No verified patient variant input in this GMDB replay; CHOP variant evidence cards are a separate cohort'},
        'initial_top5':before,'steps':steps,'final_top5':ranking(prob),
        'reference_diagnosis_evaluation_only':disease(adapter.labels[truth]),
        'reference_diagnosis_seen_by_agent':False,'stop_reason':'action_budget_exhausted',
        'model_scores_are_calibrated_disease_probabilities':False,
        'model_manifest_sha256':sha(adapter.manifest_path),
        'policy_manifest_sha256':load(ROOT/'agent_releases/CURRENT.json')['sha256']}
    out=PRIVATE/'local_examples';out.mkdir(parents=True,exist_ok=True,mode=0o700)
    save(out/'example_001.json',example)
    # The replay row is kept in a separate private reproducibility record.
    save(out/'example_001_source.private.json',{'validation_row':local_row,'array_row':index,'input_sha256':sha(source)})
    esc=lambda x:html.escape(str(x))
    def table(rows,keys):
        return '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(row.get(k,''))+'</td>' for k in keys)+'</tr>' for row in rows)+'</table>'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'none\'; connect-src \'none\'; base-uri \'none\'; form-action \'none\'">',
        '<title>Local agent example 001</title><style>body{font:16px/1.6 Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#151515;background:white}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccc;padding:8px;text-align:left}th{background:#f3f3f3}.note{padding:14px;border-left:4px solid #444;background:#f5f5f5}</style>',
        '<body><h1>Agent: one real development example</h1>',
        '<p class="note">Private: view only within the authorized HPC environment. No images or real MRN are included. This is a retrospective masked-HPO replay, not an observed clinical conversation. Unknown is not negative.</p>',
        '<p>'+esc(example['selection_rule'])+'</p>',
        '<h2>Input</h2><p>'+esc(example['input']['face'])+'</p>',table(initial,['HPO','name','status']),
        '<p>Five questions maximum. '+esc(example['input']['variant'])+'</p>',
        '<h2>Initial output</h2><p>Scores are model outputs, not calibrated clinical probabilities.</p>',table(before,['rank','name','model_score'])]
    for step in steps:
        q=step['question']
        parts+=['<h2>Question '+str(step['step'])+'</h2>',
            '<p>'+esc(q['name']+' ('+q['HPO']+')')+'</p>' if q else '<p>No question available.</p>',
            '<p><b>Recorded answer:</b> '+esc(step['answer'])+'. '+esc(step['source'])+'</p>',
            '<p><b>Selection:</b> released fixed order learned from training-record frequency; excludes already observed or asked findings.</p>',
            table(step['updated_top5'],['rank','name','model_score'])]
    parts+=['<h2>Final output</h2>',table(example['final_top5'],['rank','name','model_score']),
        '<p>Stop: five-question budget exhausted.</p>',
        '<details><summary>Reference diagnosis — evaluation only, never supplied to the agent</summary><p>'+esc(example['reference_diagnosis_evaluation_only']['name'])+'</p></details>',
        '<p>Verified against the stored development replay. Patient-level outputs were not printed to the chat.</p></body></html>']
    (out/'example_001.html').write_text('\n'.join(parts));(out/'example_001.html').chmod(0o600)
    receipt={'state':'complete','examples':1,'real_case':True,'retrospective_simulation':True,
             'matches_saved_replay':True,'images_or_mrn_in_report':False,'patient_content_in_stdout':False,
             'html_path':str(out/'example_001.html'),'json_path':str(out/'example_001.json'),'updated_at_utc':now()}
    save(out/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':guarded_main(main)

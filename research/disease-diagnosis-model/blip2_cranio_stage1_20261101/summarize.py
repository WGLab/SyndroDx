"""Aggregate pilot results; do not promote models or consult clinical test sets."""
import numpy as np
from common import *

def main():
    p=load(WORK/'protocol.json');rows={}
    for condition in p['conditions']:
        rs=[load(WORK/'cases'/f'{condition}__seed{s}'/'result.json') for s in p['seeds']]
        assert all(r.get('state')=='complete' for r in rs)
        rows[condition]=dict(alignment={k:float(np.mean([r['alignment'][k] for r in rs])) for k in ['mean_R1','mean_R5']},
            image_only_disease_retrieval={k:float(np.mean([r['image_only_disease_retrieval'][k] for r in rs])) for k in ['top1','top5','top10']},
            native_visual_disease_retrieval=rs[0]['native_visual_disease_retrieval'],
            shuffled_validation_alignment={k:float(np.mean([r['shuffled_validation_alignment'][k] for r in rs])) for k in ['mean_R1','mean_R5']},
            selected_epochs={str(r['seed']):r['selected_epoch'] for r in rs})
    save(WORK/'results.json',dict(state='complete',conditions=rows,counts=load(WORK/'data_receipt.json')['counts'],
        development_only=True,two_seed_pilot=True,baseline_replaced=False,stage2_started=False,
        interpretation='Different filtered cohort from earlier 320-patient diagnosis experiments. Alignment is matching recorded patient-level facial captions, not expert localization. No clinical performance claim.',updated_at_utc=now()))

if __name__=='__main__':main()

"""Internal patient-isolated development checks; no GMDB test selection."""
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from control import WORK, V1, load, save, now, digest,notify
from data import evaluation_cohort, EvaluationImages
sys.path.insert(0, str(V1))
sys.path.insert(0, str(WORK.parent/'training_diagnostics_20261017'))
from evaluation_core import aggregate_patients, retrieval, macro_score, paired_interval, collapse_statistics
from frozen_probe import training_folds, select_alpha, ridge_predictions, label_hits
from guard_policy import advisory

def score(records, features, labels, step):
    parts={s:aggregate_patients(records,features,s) for s in ('train','validation')}
    gids,g=parts['train']; qids,q=parts['validation']
    if set(gids)&set(qids):
        raise RuntimeError('patient_leakage_in_development_scoring')
    fixed=load(WORK/'probe_parameters.json'); report={}; hits={}
    for category,short in [('disease_or_disorder','disease'),('structural_diagnosis','structural')]:
        lab=labels[category]
        gi=[i for i,k in enumerate(gids) if lab.get(k)]; qi=[i for i,k in enumerate(qids) if lab.get(k)]
        gy=[set(lab[gids[i]]) for i in gi]; qy=[set(lab[qids[i]]) for i in qi]
        rh,coverage=retrieval(q[qi],g[gi],[qids[i] for i in qi],[gids[i] for i in gi],qy,gy)
        alpha,cv=select_alpha(g[gi],gy,training_folds(gy))
        if step==0:
            fixed[short]=dict(alpha=alpha,cv=cv)
        vocabulary=sorted(set().union(*gy))
        predictions=ridge_predictions(g[gi],q[qi],gy,vocabulary,[alpha])[alpha]
        linear={k:label_hits(predictions,qy,vocabulary,k) for k in (1,5,10)}
        values=dict(training_patients=len(gi),validation_patients=len(qi),
            retrieval={f'top{k}':macro_score(rh[k],qy) for k in (1,5,10)},
            ridge={f'top{k}':macro_score(linear[k],qy) for k in (1,5,10)},ridge_alpha=alpha,
            ridge_alpha_selection='training_only_five_fold_cv_at_each_checkpoint')
        if step:
            reference=np.load(WORK/'private'/f'baseline_{short}_hits.npz',allow_pickle=False)
            values['retrieval_paired']=paired_interval(reference['retrieval'],rh[1],qy,repetitions=1000)
            values['ridge_paired']=paired_interval(reference['ridge'],linear[1],qy,repetitions=1000)
        else:
            np.savez(WORK/'private'/f'baseline_{short}_hits.npz',retrieval=rh[1],ridge=linear[1])
        report[short]=values
    if step==0:
        save(WORK/'probe_parameters.json',fixed)
    return report, collapse_statistics(q)

def validate(teacher, step, attempt):
    rank,world=dist.get_rank(),dist.get_world_size()
    rows,labels=evaluation_cohort()
    indices=list(range(rank,len(rows),world))
    folder=WORK/'private'/f'validation_{step:07d}_{attempt:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(EvaluationImages(rows,indices),batch_size=64,num_workers=2,pin_memory=True)
    pieces=[]; teacher.eval()
    with torch.no_grad():
        for images in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                values=teacher.embed(images.cuda(non_blocking=True))
            if not torch.isfinite(values).all():
                raise RuntimeError('nonfinite_validation_embedding')
            pieces.append(values.cpu().numpy())
    features=np.concatenate(pieces) if pieces else np.empty((0,512),np.float32)
    np.save(folder/f'rank_{rank:02d}.npy',features)
    dist.barrier()
    safe=True
    if rank==0:
        full=np.empty((len(rows),512),np.float32)
        for r in range(world):
            shard=np.load(folder/f'rank_{r:02d}.npy',allow_pickle=False)
            if shard.shape != full[r::world].shape:
                raise RuntimeError('incomplete_validation_shards')
            full[r::world]=shard
        metrics,collapse=score(rows,full,labels,step)
        report=dict(state='complete',completed_steps=step,metrics=metrics,feature_health=collapse,
            gmdb_test_used=False,patient_isolated=True,metadata_labels_not_clinical_gold_standard=True,
            exploratory_repeated_validation=True,updated_at_utc=now())
        baseline=load(WORK/'baseline_validation.json')
        monitor=load(WORK/'guard_state.json')
        if step==0:
            save(WORK/'baseline_validation.json',report)
            np.save(WORK/'private/baseline_features.npy',full)
        else:
            original=np.load(WORK/'private/baseline_features.npy',allow_pickle=False)
            report['mean_original_feature_cosine']=float((original*full).sum(1).mean())
            severe_collapse=collapse['total_coordinate_variance'] < .5*baseline['feature_health']['total_coordinate_variance']
            declines=[key for key,v in metrics.items() if v['retrieval_paired']['paired_patient_bootstrap_95_interval'][1] < 0]
            previous=monitor.get('retrieval_declines',[])
            repeated=set(declines)&set(previous)
            previous_lr=load(WORK/'adaptive_lr.json').get('multiplier',1.)
            decision=advisory(severe_collapse,repeated,previous_lr)
            adverse=decision['adverse_signal']
            safe=True
            report['health_guard']=dict(continue_training=True,feature_collapse=severe_collapse,
                adverse_signal=adverse,action='reduce_backbone_lr_and_continue' if adverse else 'continue',
                supported_retrieval_declines=declines,repeated_supported_retrieval_declines=sorted(repeated))
            save(WORK/'guard_state.json',dict(completed_steps=step,retrieval_declines=declines))
            if adverse:
                multiplier=decision['multiplier']
                save(WORK/'adaptive_lr.json',dict(multiplier=multiplier,trigger_step=step,
                    reason='development_signal_advisory_backoff',updated_at_utc=now()))
                save(WORK/'validation_alerts'/f'step_{step:07d}.json',dict(report['health_guard'],
                    completed_steps=step,backbone_lr_multiplier=multiplier,updated_at_utc=now()))
                if multiplier<previous_lr:
                    notify('advisory '+str(step),'TransFace validation alert: training continues with lower learning rate',
                        'An internal development signal triggered a predefined backbone learning-rate reduction. Training continues, and best full checkpoints are protected. No images or patient records are included.')
        save(WORK/'validation'/f'step_{step:07d}.json',report)
        save(WORK/'latest_validation.json',report)
    valid=torch.tensor(int(safe),device='cuda')
    dist.broadcast(valid,src=0)
    return bool(valid)

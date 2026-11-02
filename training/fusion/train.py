"""Single-GPU matched diagnosis training, with exact epoch-boundary resume."""
import fcntl
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from model import PhenotypeModel, profile_loss


@torch.no_grad()
def validate(model, data, indices, *, feature_key='features', term_limit=None, shuffled=False):
    model.eval()
    indices = np.asarray(indices)
    feature_indices = indices.copy()
    if shuffled:
        feature_indices = np.random.default_rng(20261031).permutation(feature_indices)
    scores = []
    for start in range(0, len(indices), 64):
        ix = indices[start:start + 64]
        fx = feature_indices[start:start + 64]
        ids, states = data['ids'][ix].clone(), data['states'][ix].clone()
        if term_limit is not None:
            # Explicit absences retained; sample a fixed subset of observed positive findings.
            for j, patient_index in enumerate(ix):
                positive = torch.where(ids[j].ne(0) & states[j].eq(0))[0]
                rng = np.random.default_rng(20261031 + int(patient_index))
                keep = set(rng.choice(len(positive), min(term_limit, len(positive)), replace=False).tolist())
                for k, pos in enumerate(positive):
                    if k not in keep:
                        ids[j, pos] = 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(data[feature_key][fx], ids, states)
        scores.append(logits.float())
    scores = torch.cat(scores)
    truth = data['labels'][indices]
    order = scores.argsort(dim=1, descending=True, stable=True)
    metrics = {}
    for k in (1, 5, 10):
        hit = (order[:, :min(k, scores.shape[1])] == truth[:, None]).any(1).float()
        metrics['macro_top' + str(k)] = float(torch.stack([hit[truth == c].mean() for c in truth.unique()]).mean())
    metrics['patients'] = len(indices)
    metrics['loss'] = float(F.cross_entropy(scores, truth))
    model.train()
    return metrics, scores.cpu().numpy()


def arrays(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: torch.from_numpy(np.array(d[k])).cuda() for k in d if k not in ('train', 'validation')}, d['train'].copy(), d['validation'].copy()


def main(name):
    os.umask(0o077)
    folder = WORK / 'cases' / name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if load(folder / 'result.json').get('state') == 'complete':
        return
    torch.set_num_threads(4)
    protocol = load(WORK / 'protocol.json')
    case = load(WORK / 'cases.json')['cases'][name]
    assert case['condition'] == 'no_chop' and case['mode'] == 'fusion_selected'
    receipt = load(WORK / 'data_receipt.json')
    for file, sha in receipt['files'].items():
        assert digest(WORK / 'private' / file) == sha
    ep = WORK / 'features/term_embeddings.npy'
    assert digest(ep) == load(WORK / 'term_embedding_receipt.json')['sha256']
    torch.manual_seed(case['seed'])
    np.random.seed(case['seed'])
    data, train, val = arrays(WORK / 'private/gmdb.npz')
    public, public_train, public_val = arrays(WORK / 'private/public_cases.npz')
    with np.load(WORK / 'private/profiles.npz') as d:
        profiles = {k: torch.from_numpy(np.array(d[k])).cuda() for k in d}
    counts = receipt['counts']
    model = PhenotypeModel(torch.from_numpy(np.load(ep)), counts['GMDB_classes'], counts['public_training_diseases'], case['mode']).cuda()
    pre = load(PROJECT / 'hpo_fusion_staged_20261031/pretrain' / ('public_pretrained_knowledge__seed' + str(case['seed'])) / 'result.json')
    assert pre['state'] == 'complete' and digest(pre['path']) == pre['sha256']
    initialization = torch.load(pre['path'], map_location='cpu', weights_only=False)
    initial_state = initialization['model']
    assertion = initial_state['assertion.weight']
    initial_state['assertion.weight'] = torch.cat([assertion, assertion.mean(0, keepdim=True)], 0)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4, min_lr=1e-6)
    feature_key = 'original_features' if case['mode'] == 'fusion_raw' else 'features'
    weights = torch.bincount(data['labels'][train], minlength=counts['GMDB_classes']).float().clamp_min(1).rsqrt()
    weights /= weights.mean()
    public_weights = torch.bincount(public['labels'][public_train], minlength=counts['public_training_diseases']).float().clamp_min(1).rsqrt()
    public_weights /= public_weights.mean()
    epoch, step, best, bad = 0, 0, (-1., -1.), 0
    if (folder / 'last.pt').exists():
        s = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == digest(WORK / 'protocol.json')
        model.load_state_dict(s['model'])
        optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        epoch, step, best, bad = s['epoch'], s['step'], tuple(s['best']), s['bad']
        torch.set_rng_state(s['torch_rng'])
        torch.cuda.set_rng_state(s['cuda_rng'])
    for epoch in range(epoch + 1, protocol['max_epochs'] + 1):
        model.train()
        started = time.monotonic()
        losses = []
        order = np.random.default_rng(case['seed'] + epoch * 1009).permutation(train)
        for start in range(0, len(order), protocol['batch_size']):
            ix = order[start:start + protocol['batch_size']]
            features = data[feature_key][ix].clone()
            ids, states = data['ids'][ix].clone(), data['states'][ix]
            # Term deletion means unknown, never a newly negated finding.
            ids[torch.rand(ids.shape, device='cuda') < .1] = 0
            if case['mode'] != 'hpo':
                drop = torch.rand(len(ix), device='cuda')
                features[drop < .1] = 0
                ids[(drop >= .1) & (drop < .2)] = 0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, representation = model(features, ids, states)
                loss = F.cross_entropy(logits.float(), data['labels'][ix], weight=weights)
                rng = np.random.default_rng(case['seed'] * 1000003 + step)
                j = rng.choice(train, size=min(len(ix), len(train)), replace=False)
                h, _, _ = model.phenotype(data['ids'][j], data['states'][j])
                auxiliary = F.cross_entropy(model.head(h).float(), data['labels'][j], weight=weights)
                loss = loss + protocol['public_auxiliary_weight'] * auxiliary
                if True:
                    knowledge = profile_loss(model, representation, data['labels'][ix], profiles)
                    loss = loss + protocol['knowledge_auxiliary_weight'] * knowledge
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite_training_loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
        metrics, _ = validate(model, data, val, feature_key=feature_key)
        score = (metrics['macro_top1'], metrics['macro_top5'])
        scheduler.step(score[0])
        improved = score > best
        if improved:
            best, bad = score, 0
        else:
            bad += 1
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          epoch=epoch, step=step, best=best, bad=bad, case=case,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                          protocol_sha256=digest(WORK / 'protocol.json'), data_sha256=digest(WORK / 'data_receipt.json'))
        temp = folder / 'last.tmp.pt'
        torch.save(checkpoint, temp)
        temp.replace(folder / 'last.pt')
        report = dict(state='training', epoch=epoch, step=step, loss=float(np.mean(losses)), validation=metrics,
                      best_validation_top1=best[0], epoch_seconds=time.monotonic() - started, updated_at_utc=now())
        save(folder / 'status.json', report)
        if improved:
            target = folder / 'best.pt'
            target.unlink(missing_ok=True)
            os.link(folder / 'last.pt', target)
            save(folder / 'best_validation.json', report)
        if epoch >= protocol['min_epochs'] and bad >= protocol['patience']:
            break
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    final, scores = validate(model, data, val, feature_key=feature_key)
    variants = {}
    if case['mode'] != 'hpo':
        variants['shuffled_face'], _ = validate(model, data, val, feature_key=feature_key, shuffled=True)
    for k in (1, 3, 5):
        variants['HPO_' + str(k)], _ = validate(model, data, val, feature_key=feature_key, term_limit=k)
    np.save(folder / 'validation_logits.private.npy', scores)
    save(folder / 'result.json', dict(state='complete', case=case, validation=final, controls=variants,
         selected_epoch=state['epoch'], completed_epoch=epoch, checkpoint_path=str(folder / 'best.pt'),
         checkpoint_sha256=digest(folder / 'best.pt'), clinical_test_used=False, test_used=False, updated_at_utc=now()))


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception as e:
        save(WORK / 'cases' / sys.argv[1] / 'failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))
        raise

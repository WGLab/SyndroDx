"""Public HPO pretraining precedes patient-paired GMDB adaptation."""
import fcntl
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from model import PhenotypeModel, profile_loss


@torch.no_grad()
def validate(model, data, indices):
    model.eval()
    scores = []
    for start in range(0, len(indices), 64):
        ix = indices[start:start + 64]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h, _, _ = model.phenotype(data['ids'][ix], data['states'][ix])
            logits = model.public_head(h)
        scores.append(logits.float())
    scores = torch.cat(scores)
    truth = data['labels'][indices]
    order = scores.argsort(1, descending=True)
    result = {}
    for k in (1, 5):
        hit = (order[:, :k] == truth[:, None]).any(1).float()
        result['macro_top' + str(k)] = float(torch.stack([hit[truth == c].mean() for c in truth.unique()]).mean())
    result['cases'] = len(indices)
    model.train()
    return result


def main(name):
    os.umask(0o077)
    folder = WORK / 'pretrain' / name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if load(folder / 'result.json').get('state') == 'complete':
        return
    condition, seed = name.split('__seed')
    seed = int(seed)
    assert condition == 'public_pretrained_knowledge'
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    protocol = load(WORK / 'protocol.json')
    receipt = load(WORK / 'data_receipt.json')
    counts = receipt['counts']
    for file in ['public_cases.npz', 'profiles.npz']:
        assert digest(WORK / 'private' / file) == receipt['files'][file]
    with np.load(WORK / 'private/public_cases.npz') as d:
        data = {k: torch.from_numpy(np.array(d[k])).cuda() for k in ['ids', 'states', 'labels']}
        train, val = d['train'].copy(), d['validation'].copy()
    with np.load(WORK / 'private/profiles.npz') as d:
        profiles = {k: torch.from_numpy(np.array(d[k])).cuda() for k in d}
    model = PhenotypeModel(torch.from_numpy(np.load(WORK / 'features/term_embeddings.npy')),
                           counts['GMDB_classes'], counts['public_training_diseases'], 'hpo').cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=protocol['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=.5, patience=4, min_lr=1e-6)
    weights = torch.bincount(data['labels'][train], minlength=counts['public_training_diseases']).float().clamp_min(1).rsqrt()
    weights /= weights.mean()
    epoch, best, bad = 0, (-1., -1.), 0
    if (folder / 'last.pt').exists():
        s = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == digest(WORK / 'protocol.json')
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        epoch, best, bad = s['epoch'], tuple(s['best']), s['bad']
        torch.set_rng_state(s['torch_rng'])
        torch.cuda.set_rng_state(s['cuda_rng'])
    for epoch in range(epoch + 1, protocol['public_pretrain_max_epochs'] + 1):
        began = time.monotonic()
        order = np.random.default_rng(seed + epoch * 1009).permutation(train)
        losses = []
        for start in range(0, len(order), 128):
            ix = order[start:start + 128]
            ids = data['ids'][ix].clone()
            ids[torch.rand(ids.shape, device='cuda') < .1] = 0
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h, _, _ = model.phenotype(ids, data['states'][ix])
                loss = F.cross_entropy(model.public_head(h).float(), data['labels'][ix], weight=weights)
                if condition == 'public_pretrained_knowledge':
                    loss = loss + .1 * profile_loss(model, h, data['labels'][ix], profiles, counts['GMDB_classes'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite_public_pretrain_loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        metrics = validate(model, data, val)
        score = (metrics['macro_top1'], metrics['macro_top5'])
        scheduler.step(score[0])
        improved = score > best
        best, bad = (score, 0) if improved else (best, bad + 1)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                     best=best, bad=bad, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                     protocol_sha256=digest(WORK / 'protocol.json'))
        temp = folder / 'last.tmp.pt'
        torch.save(state, temp)
        temp.replace(folder / 'last.pt')
        save(folder / 'status.json', dict(state='public_HPO_pretraining', epoch=epoch, loss=float(np.mean(losses)),
             validation=metrics, epoch_seconds=time.monotonic() - began, updated_at_utc=now()))
        if improved:
            target = folder / 'best.pt'
            target.unlink(missing_ok=True)
            os.link(folder / 'last.pt', target)
            save(folder / 'best_validation.json', dict(epoch=epoch, validation=metrics))
        if epoch >= protocol['public_pretrain_min_epochs'] and bad >= protocol['public_pretrain_patience']:
            break
    save(folder / 'result.json', dict(state='complete', condition=condition, seed=seed,
         selected=load(folder / 'best_validation.json'), path=str(folder / 'best.pt'), sha256=digest(folder / 'best.pt'),
         clinical_test_used=False, GMDB_cases_used_in_pretraining=False, updated_at_utc=now()))


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception as e:
        save(WORK / 'pretrain' / sys.argv[1] / 'failure.json', dict(error_type=type(e).__name__, updated_at_utc=now()))
        raise

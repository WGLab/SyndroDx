import os, sys, random, time, fcntl
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from data import verify, Images, classification
from network import Encoder

@torch.no_grad()
def validate(m, rows, ids):
    m.eval()
    values = []
    offsets = []
    loss = 0.0
    for (x, y, ix) in DataLoader(Images(rows, ids), batch_size=128, num_workers=2, pin_memory=True):
        z = m(x.cuda(non_blocking=True), margin=False)
        if not torch.isfinite(z).all():
            raise FloatingPointError('nonfinite_validation')
        loss += float(F.cross_entropy(z, y.cuda(), reduction='sum'))
        values.append(z.cpu().numpy())
        offsets.extend(ix.tolist())
    r = classification(np.concatenate(values), offsets, rows)
    r['loss'] = loss / len(ids)
    m.train()
    return r

def main():
    os.umask(63)
    case = load(WORK / 'cases.json')['cases'][sys.argv[1]]
    assert case['arm'] == 'transface_chop_cpt' and case['regime'] == 'arcface_full'
    folder = WORK / 'cases' / sys.argv[1]
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if load(folder / 'result.json').get('state') == 'complete':
        return
    p = load(WORK / 'protocol.json')
    rows = verify()
    seed = case['seed']
    regime = case['regime']
    arm = case['arm']
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tr = [i for (i, r) in enumerate(rows) if r['split'] == 'train']
    va = [i for (i, r) in enumerate(rows) if r['split'] == 'validation' and r['class_index'] >= 0]
    classes = max((rows[i]['class_index'] for i in tr)) + 1
    m = Encoder(arm, regime, classes).cuda()
    lr = case['lr']
    opt = torch.optim.AdamW([dict(params=[v for v in m.body.parameters() if v.requires_grad] + list(m.feature.parameters()), lr=lr), dict(params=m.head.parameters(), lr=0.001)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, threshold=0.0005, min_lr=1e-07)
    counts = np.bincount([rows[i]['class_index'] for i in tr], minlength=classes)
    weights = torch.tensor(0.5 + 0.5 * counts.min() / counts, dtype=torch.float32, device='cuda')
    start = 0
    if (folder / 'last.pt').exists():
        s = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == digest(WORK / 'protocol.json')
        m.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        sched.load_state_dict(s['scheduler'])
        start = s['epoch']
        torch.set_rng_state(s['rng']['torch'])
        torch.cuda.set_rng_state(s['rng']['cuda'])
        np.random.set_state(s['rng']['numpy'])
        random.setstate(s['rng']['python'])
    metrics = validate(m, rows, va) if start >= p['epochs'] else None
    for epoch in range(start, p['epochs']):
        m.train()
        began = time.monotonic()
        losses = []
        generator = torch.Generator().manual_seed(seed + epoch * 100003)
        loader = DataLoader(Images(rows, tr, seed, epoch, True), batch_size=p['batch_size'], shuffle=True, generator=generator, num_workers=2, pin_memory=True, drop_last=True)
        for (x, y, _) in loader:
            opt.zero_grad(set_to_none=True)
            z = m(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
            loss = F.cross_entropy(z, y.cuda(), weight=None)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite_training_loss')
            loss.backward()
            if epoch < 2:
                for v in list(m.body.parameters()) + list(m.feature.parameters()):
                    v.grad = None
            torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0, error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        metrics = validate(m, rows, va)
        sched.step(metrics['top5'])
        state = dict(model=m.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch + 1, case=case, validation=metrics, protocol_sha256=digest(WORK / 'protocol.json'), rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()))
        tmp = folder / 'last.tmp'
        torch.save(state, tmp)
        tmp.replace(folder / 'last.pt')
        save(folder / 'status.json', dict(state='training', epoch=epoch + 1, loss=float(np.mean(losses)), validation=metrics, epoch_seconds=time.monotonic() - began, updated_at_utc=now()))
    save(folder / 'result.json', dict(state='complete', case=case, epoch=50, validation=metrics, path=str(folder / 'last.pt'), sha256=digest(folder / 'last.pt'), test_used=False, updated_at_utc=now()))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(WORK / 'cases' / sys.argv[1] / 'failure.json', dict(error_type=type(e).__name__, oom=isinstance(e, torch.cuda.OutOfMemoryError), updated_at_utc=now()))
        raise

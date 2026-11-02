#!/usr/bin/env python
"""Is the linear probe converged, or just LR-annealed? Log train+val every epoch.

Runs three schedules so the two explanations separate:
  cos100  - what the reported 0.444 used
  cos300  - same shape, 3x longer: if loss keeps falling, cos100 was truncated
  const   - constant LR, no annealing: isolates schedule from convergence
"""
import os, csv, collections
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
csv.field_size_limit(10**9)

rows = list(csv.DictReader(open(os.path.join(HERE,'labels.tsv'), newline=''), delimiter='\t'))
X = np.load(os.path.join(HERE,'features.npy')).astype(np.float32)
mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
X = ((X - mu) / sd).astype(np.float32)

lab = np.array([r['primary_group'] for r in rows]); spl = np.array([r['split'] for r in rows])
mrn = np.array([r['mrn'] for r in rows])
use = (lab != '') & (spl != 'ssl')
classes = sorted(set(lab[use])); cidx = {c:i for i,c in enumerate(classes)}
y = np.array([cidx.get(l,-1) for l in lab]); nC = len(classes)
tr, va, te = use&(spl=='train'), use&(spl=='val'), use&(spl=='test')
y_pat = {r['mrn']: cidx[r['primary_group']] for r in rows if r['primary_group'] in cidx}
print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}  classes {nC}", flush=True)

Xtr = torch.from_numpy(X[tr]).to(DEV); ytr = torch.from_numpy(y[tr]).long().to(DEV)
Xva = torch.from_numpy(X[va]).to(DEV); yva = torch.from_numpy(y[va]).long().to(DEV)
Xte = torch.from_numpy(X[te]).to(DEV); yte = torch.from_numpy(y[te]).long().to(DEV)

def patient_top1(logits, mask):
    P = torch.softmax(logits,1).cpu().numpy()
    agg = collections.defaultdict(lambda: np.zeros(nC)); cnt = collections.Counter()
    for p,m in zip(P, mrn[mask]): agg[m]+=p; cnt[m]+=1
    ms = sorted(agg)
    pred = np.stack([agg[m]/cnt[m] for m in ms]).argmax(1)
    true = np.array([y_pat[m] for m in ms])
    return float((pred==true).mean())

def run(tag, epochs, sched_kind, lr=1e-2, bs=2048, wd=1e-4):
    torch.manual_seed(0)
    model = torch.nn.Linear(Xtr.shape[1], nC).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
           if sched_kind=='cos' else None)
    n = len(Xtr); best=(0,0)
    print(f"\n### {tag}  epochs={epochs} sched={sched_kind} lr={lr}", flush=True)
    print(f"{'ep':>4} {'lr':>9} {'train':>8} {'val':>8} {'trainAcc':>9} {'valAcc':>8} {'valPat':>8}", flush=True)
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(n, device=DEV); tot=0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = torch.nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())*len(idx)
        if sch: sch.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                lo_v = model(Xva); lo_t = model(Xtr)
                vl = float(torch.nn.functional.cross_entropy(lo_v, yva))
                va_acc = float((lo_v.argmax(1)==yva).float().mean())
                tr_acc = float((lo_t.argmax(1)==ytr).float().mean())
                vp = patient_top1(lo_v, va)
            if vp > best[0]: best = (vp, ep)
            print(f"{ep:>4} {opt.param_groups[0]['lr']:>9.2e} {tot/n:>8.4f} {vl:>8.4f} "
                  f"{tr_acc:>9.4f} {va_acc:>8.4f} {vp:>8.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        print(f"  -> best val patient top1={best[0]:.4f} @ep{best[1]};  "
              f"final TEST patient top1={patient_top1(model(Xte), te):.4f}", flush=True)

run('cos100',  100, 'cos')
run('cos300',  300, 'cos')
run('const300',300, 'none')

#!/usr/bin/env python
"""Probe frozen features and, with the same model, rank suspected mislabels.

Two jobs in one pass:
  1. Baseline. kNN + linear probe over patient-disjoint splits, scored at both
     image and patient level. Patient level is the clinically meaningful one.
  2. Label audit. Confident-learning style: cross-validated (patient-grouped)
     probabilities for every labelled image, aggregated per patient. Patients
     whose own folder label gets a low probability while a different class gets
     a high one are the review queue -- this is what triages the 510 conflict
     patients and the 6,812 unverifiable single-folder ones.

Runs on GPU when available; the linear probe is plain SGD rather than sklearn
because 73k x 1024 x 53 classes, fit six times, is not a CPU job.
"""
import os, sys, csv, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(10**9)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def metrics(y, pred, n_cls):
    y = np.asarray(y); pred = np.asarray(pred)
    acc = float((y == pred).mean())
    f1s, recs = [], []
    for c in range(n_cls):
        t = (y == c); p = (pred == c)
        if t.sum() == 0:
            continue
        tp = float((t & p).sum())
        prec = tp / max(p.sum(), 1)
        rec = tp / t.sum()
        recs.append(rec)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return acc, float(np.mean(f1s)), float(np.mean(recs))


def report(name, y, pred, n_cls):
    a, f, b = metrics(y, pred, n_cls)
    print(f"  {name:<22} top1={a:.4f}  macroF1={f:.4f}  balAcc={b:.4f}  n={len(y)}", flush=True)


def patient_agg(prob, mrns, y_pat):
    agg = collections.defaultdict(lambda: np.zeros(prob.shape[1], dtype=np.float64))
    cnt = collections.Counter()
    for p, m in zip(prob, mrns):
        agg[m] += p; cnt[m] += 1
    ms = sorted(agg)
    P = np.stack([agg[m] / cnt[m] for m in ms])
    return ms, P, np.array([y_pat[m] for m in ms]), P.argmax(1)


def train_linear(Xtr, ytr, n_cls, epochs=100, bs=2048, lr=1e-2, wd=1e-4, seed=0, quiet=False):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(ytr).long().to(DEV)
    model = torch.nn.Linear(Xt.shape[1], n_cls).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if not quiet and (ep + 1) % 25 == 0:
            print(f"    epoch {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, X, bs=8192):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def knn_prob(Xq, mq, Xtr, ytr, mtr, n_cls, k=20, bs=1024):
    Xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(ytr).long().to(DEV)
    # map MRN strings to ints so the same-patient mask is a cheap tensor compare
    uniq = {m: i for i, m in enumerate(sorted(set(mtr) | set(mq)))}
    mt = torch.tensor([uniq[m] for m in mtr], device=DEV)
    out = np.zeros((len(Xq), n_cls), dtype=np.float32)
    for i in range(0, len(Xq), bs):
        xb = torch.from_numpy(Xq[i:i+bs]).to(DEV)
        mb = torch.tensor([uniq[m] for m in mq[i:i+bs]], device=DEV)
        S = xb @ Xt.T
        S[mb[:, None] == mt[None, :]] = -2.0        # never vote with your own photos
        v, idx = S.topk(k, dim=1)
        w = v.clamp(min=0)
        lab = yt[idx]
        p = torch.zeros(len(xb), n_cls, device=DEV)
        p.scatter_add_(1, lab, w)
        p = p / p.sum(1, keepdim=True).clamp(min=1e-8)
        out[i:i+len(xb)] = p.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=os.path.join(HERE, 'features.npy'))
    ap.add_argument('--labels',   default=os.path.join(HERE, 'labels.tsv'))
    ap.add_argument('--out-audit',default=os.path.join(HERE, 'review_queue.tsv'))
    ap.add_argument('--k',        type=int, default=20)
    ap.add_argument('--epochs',   type=int, default=100)
    ap.add_argument('--skip-audit', action='store_true')
    args = ap.parse_args()

    print(f"device={DEV}", flush=True)
    rows = list(csv.DictReader(open(args.labels, newline=''), delimiter='\t'))
    if args.features.endswith('.npz'):
        z = np.load(args.features, allow_pickle=True); X = z['X'].astype(np.float32)
    else:
        X = np.load(args.features).astype(np.float32)
    assert len(rows) == len(X), f"{len(rows)} labels vs {len(X)} features"

    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)
    Xn = l2norm(X).astype(np.float32)          # cosine kNN wants unit norm
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)    # linear probe wants unit variance
    X = Xn

    lab = np.array([r['primary_group'] for r in rows])
    spl = np.array([r['split'] for r in rows])
    mrn = np.array([r['mrn'] for r in rows])
    use = (lab != '') & (spl != 'ssl')
    classes = sorted(set(lab[use]))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx.get(l, -1) for l in lab])
    nC = len(classes)
    print(f"{use.sum():,} labelled images, {nC} classes, dim={X.shape[1]}", flush=True)

    tr, va, te = use & (spl == 'train'), use & (spl == 'val'), use & (spl == 'test')
    y_pat = {r['mrn']: cidx[r['primary_group']] for r in rows if r['primary_group'] in cidx}
    Xtr, ytr, mtr = X[tr], y[tr], mrn[tr]

    print(f"\n=== kNN (k={args.k}, cosine, same-patient masked) ===", flush=True)
    for nm, mask in (('val', va), ('test', te)):
        P = knn_prob(X[mask], mrn[mask], Xtr, ytr, mtr, nC, k=args.k)
        report(f'{nm} image', y[mask], P.argmax(1), nC)
        _, _, yp, pp = patient_agg(P, mrn[mask], y_pat)
        report(f'{nm} patient', yp, pp, nC)

    print("\n=== linear probe ===", flush=True)
    model = train_linear(Xs[tr], ytr, nC, epochs=args.epochs)
    for nm, mask in (('val', va), ('test', te)):
        P = predict(model, Xs[mask])
        report(f'{nm} image', y[mask], P.argmax(1), nC)
        _, _, yp, pp = patient_agg(P, mrn[mask], y_pat)
        report(f'{nm} patient', yp, pp, nC)
        if nm == 'test':
            per = collections.defaultdict(lambda: [0, 0])
            for t, p in zip(yp, pp):
                per[classes[t]][1] += 1; per[classes[t]][0] += int(t == p)
            print("\n  per-class patient recall (test):", flush=True)
            for c, (ok, n) in sorted(per.items(), key=lambda x: -x[1][1]):
                print(f"    {c:<32} {ok:4d}/{n:<4d} {ok/n:6.2f}", flush=True)

    if args.skip_audit:
        return

    print("\n=== label audit (patient-grouped 5-fold OOF probabilities) ===", flush=True)
    idx = np.where(use)[0]
    Xa, ya, ma = Xs[idx], y[idx], mrn[idx]
    pats = sorted(set(ma))
    rng = np.random.RandomState(0); rng.shuffle(pats)
    fold_of = {m: i % 5 for i, m in enumerate(pats)}
    fa = np.array([fold_of[m] for m in ma])
    oof = np.zeros((len(idx), nC), dtype=np.float32)
    for f in range(5):
        a, b = fa != f, fa == f
        mdl = train_linear(Xa[a], ya[a], nC, epochs=args.epochs, seed=f, quiet=True)
        oof[b] = predict(mdl, Xa[b])
        print(f"  fold {f+1}/5 done", flush=True)

    _, P, _, _ = patient_agg(oof, ma, y_pat)
    ms = sorted(set(ma))
    meta = {r['mrn']: (r['tier'], r['primary_group'], r['patient_groups'], r['raw_label'])
            for r in rows if r['mrn'] and r['primary_group']}
    cnt = collections.Counter(ma)
    out = []
    for m, p in zip(ms, P):
        own = cidx[meta[m][1]]; alt = int(p.argmax())
        out.append((float(p[own]), float(p[alt]), classes[alt], m, cnt[m], *meta[m]))
    # class-conditional thresholds: t_c = mean self-confidence among patients
    # labelled c. Without this the queue just ranks rare classes first.
    own_p = collections.defaultdict(list)
    for m, p in zip(ms, P):
        own_p[meta[m][1]].append(float(p[cidx[meta[m][1]]]))
    thr = {c: float(np.mean(v)) for c, v in own_p.items()}
    out = [t + (thr[t[5+1]] if False else thr[t[6]],
                float(t[0] - thr[t[6]]),
                int(t[0] < thr[t[6]] and t[1] >= thr.get(t[2], 1.0))) for t in out]
    out.sort(key=lambda t: t[10])          # most negative margin first
    with open(args.out_audit, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['margin','flagged','p_own_label','class_threshold','p_top_alt',
                    'top_alt_class','mrn','n_images','tier','folder_label',
                    'patient_groups','raw_label'])
        for t in out:
            w.writerow([f"{t[10]:.4f}", t[11], f"{t[0]:.4f}", f"{t[9]:.4f}",
                        f"{t[1]:.4f}", t[2], t[3], t[4], t[5], t[6], t[7], t[8]])
    print(f"\nwrote {args.out_audit}  ({len(out):,} patients, worst first)", flush=True)

    by = collections.defaultdict(list)
    for t in out:
        by[t[5]].append((t[10], t[11]))
    print("\n  class-normalised margin on the patient's own folder label, by tier:", flush=True)
    print("  (if the tiers are real, C_conflict should be the most-flagged)", flush=True)
    for k in sorted(by):
        mg = np.array([x[0] for x in by[k]]); fl = np.array([x[1] for x in by[k]])
        print(f"    {k:<20} n={len(mg):5d}  mean_margin={mg.mean():+.3f}  flagged={fl.mean():.3f}", flush=True)


if __name__ == '__main__':
    main()

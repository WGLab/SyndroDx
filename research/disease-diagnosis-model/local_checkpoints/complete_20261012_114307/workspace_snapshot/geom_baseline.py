#!/usr/bin/env python
"""Gradient-boosted baseline on landmark geometry alone. No GPU, interpretable.

This is the number any deep model has to beat, and the one a clinical reviewer
will ask for. Scored at both image and patient level over the patient-disjoint
splits; val/test contain only corroborated (tier A/B) patients.
"""
import os, sys, csv, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=os.path.join(HERE, 'geom_features.npz'))
    ap.add_argument('--labels',   default=os.path.join(HERE, 'labels.tsv'))
    ap.add_argument('--iters',    type=int, default=400)
    ap.add_argument('--top-classes', type=int, default=0,
                    help='restrict to the N largest classes (0 = all)')
    args = ap.parse_args()

    z = np.load(args.features, allow_pickle=True)
    X, names = z['X'].astype(np.float32), list(z['names'])
    rows = list(csv.DictReader(open(args.labels, newline=''), delimiter='\t'))
    assert len(rows) == len(X)

    lab = np.array([r['primary_group'] for r in rows])
    spl = np.array([r['split'] for r in rows])
    mrn = np.array([r['mrn'] for r in rows])
    use = (lab != '') & (spl != 'ssl')

    if args.top_classes:
        big = [c for c, _ in collections.Counter(lab[use]).most_common(args.top_classes)]
        use &= np.isin(lab, big)
    classes = sorted(set(lab[use]))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx.get(l, -1) for l in lab])
    print(f"{use.sum():,} images, {len(classes)} classes, {X.shape[1]} features")

    tr, va, te = use & (spl == 'train'), use & (spl == 'val'), use & (spl == 'test')
    print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, balanced_accuracy_score
    clf = HistGradientBoostingClassifier(
        max_iter=args.iters, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=0, verbose=1)
    clf.fit(X[tr], y[tr])

    y_pat = {r['mrn']: cidx[r['primary_group']] for r in rows if r['primary_group'] in cidx}
    for nm, mask in (('val', va), ('test', te)):
        P = clf.predict_proba(X[mask])
        yi, pi = y[mask], P.argmax(1)
        print(f"\n{nm} image   top1={(yi==pi).mean():.4f} "
              f"macroF1={f1_score(yi,pi,average='macro',zero_division=0):.4f} "
              f"balAcc={balanced_accuracy_score(yi,pi):.4f}")
        agg = collections.defaultdict(lambda: np.zeros(len(classes)))
        cnt = collections.Counter()
        for p, m in zip(P, mrn[mask]):
            agg[m] += p; cnt[m] += 1
        ms = sorted(agg)
        yp = np.array([y_pat[m] for m in ms])
        pp = np.stack([agg[m]/cnt[m] for m in ms]).argmax(1)
        print(f"{nm} patient top1={(yp==pp).mean():.4f} "
              f"macroF1={f1_score(yp,pp,average='macro',zero_division=0):.4f} "
              f"balAcc={balanced_accuracy_score(yp,pp):.4f}  n={len(yp)}")
        if nm == 'test':
            per = collections.defaultdict(lambda: [0, 0])
            for t, p in zip(yp, pp):
                per[classes[t]][1] += 1; per[classes[t]][0] += int(t == p)
            print("\n  per-class patient recall (test):")
            for c, (ok, n) in sorted(per.items(), key=lambda x: -x[1][1]):
                print(f"    {c:<32} {ok:4d}/{n:<4d} {ok/n:6.2f}")

    # permutation importance on a val subsample -> which shape cues matter
    from sklearn.inspection import permutation_importance
    idx = np.where(va)[0]
    sub = np.random.RandomState(0).choice(idx, size=min(3000, len(idx)), replace=False)
    r = permutation_importance(clf, X[sub], y[sub], n_repeats=3, random_state=0, n_jobs=-1)
    print("\n  top 25 features by permutation importance:")
    for i in r.importances_mean.argsort()[::-1][:25]:
        print(f"    {names[i]:<34} {r.importances_mean[i]:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Turn the raw craniofacial metadata CSV into a curated per-image label table.

Emits labels.tsv (one row per image) and patients.tsv (one row per MRN).

Design decisions, per the label audit:
  * `disease_or_phenotypes` is the folder name, not a verified diagnosis, so
    every row carries a confidence tier derived from cross-folder agreement.
  * val/test are drawn ONLY from corroborated patients (tier A/B). Tier C
    (contradictory) and tier D (single folder, unverifiable) go to train, where
    noise is survivable, never to evaluation.
  * Splits are patient-level. 7.4 images/patient means a random image split
    leaks the same face across folds.
"""
import csv, sys, os, collections, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelmap import classify, resolve, assign_tier

CSV_PATH = "/controlled-data/craniofacial_data/craniofacial_metadata_frontal_updated_123025_deduplicated.csv"
csv.field_size_limit(10**9)

COLS = ['index','image_path','parent_path','norm_mrn','norm_name','date_taken',
        'disease_or_phenotypes','doctor_name','cropping_path']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=CSV_PATH)
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-patients', type=int, default=20,
                    help='drop DX groups with fewer than this many patients')
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline='', encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rec = {k: (r.get(k) or '').strip() for k in COLS}
            rec['bucket'], rec['dx_group'] = classify(rec['disease_or_phenotypes'])
            rows.append(rec)
    print(f"read {len(rows):,} rows", flush=True)

    # ---- per-patient aggregation -------------------------------------------
    folders = collections.defaultdict(set)
    groups  = collections.defaultdict(set)
    gcount  = collections.defaultdict(collections.Counter)
    for r in rows:
        m = r['norm_mrn']
        if not m or r['bucket'] != 'DX':
            continue
        folders[m].add(r['parent_path'])
        groups[m].add(r['dx_group'])
        gcount[m][r['dx_group']] += 1

    tier, primary, resolved = {}, {}, {}
    for m in groups:
        tier[m] = assign_tier(folders[m], groups[m])
        res = resolve(groups[m])
        resolved[m] = res
        # primary = most-photographed of the resolved groups
        primary[m] = max(res, key=lambda g: (gcount[m][g], g))

    # ---- drop rare groups ---------------------------------------------------
    pats_per_group = collections.Counter()
    for m, res in resolved.items():
        for g in res:
            pats_per_group[g] += 1
    keep = {g for g, n in pats_per_group.items() if n >= args.min_patients}
    print(f"keeping {len(keep)} / {len(pats_per_group)} DX groups "
          f"(>= {args.min_patients} patients)", flush=True)

    # ---- patient-level splits ----------------------------------------------
    # Only corroborated patients are eligible for evaluation.
    eligible = [m for m in tier if tier[m] in ('A_corroborated', 'B_multi_compatible')
                and primary[m] in keep]
    by_primary = collections.defaultdict(list)
    for m in eligible:
        by_primary[primary[m]].append(m)

    rng = random.Random(args.seed)
    split = {}
    for g, ms in by_primary.items():
        ms = sorted(ms)
        rng.shuffle(ms)
        n = len(ms)
        n_test = max(1, int(round(0.15 * n)))
        n_val  = max(1, int(round(0.15 * n)))
        if n_test + n_val >= n:              # tiny group: keep at least 1 in train
            n_test = n_val = max(0, (n - 1) // 3)
        for i, m in enumerate(ms):
            split[m] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
    for m in tier:                            # tier C/D -> train only
        split.setdefault(m, 'train')

    # ---- write per-image table ---------------------------------------------
    out_img = os.path.join(args.out_dir, 'labels.tsv')
    n_by_split = collections.Counter()
    with open(out_img, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['img_id','cropping_path','image_path','mrn','date_taken','doctor',
                    'raw_label','bucket','dx_group','patient_groups','tier','primary_group','split'])
        for r in rows:
            m = r['norm_mrn']
            t  = tier.get(m, '')
            pg = '|'.join(sorted(resolved.get(m, set()) & keep))
            pr = primary.get(m, '')
            if pr not in keep:
                pr = ''
            # anything without a usable DX label is pretraining data only
            sp = split.get(m, 'ssl') if (r['bucket'] == 'DX' and pr) else 'ssl'
            g = r['dx_group'] if (r['bucket'] == 'DX' and r['dx_group'] in keep) else ''
            n_by_split[sp] += 1
            w.writerow([r['index'], r['cropping_path'], r['image_path'], m, r['date_taken'],
                        r['doctor_name'], r['disease_or_phenotypes'], r['bucket'], g,
                        pg, t, pr, sp])

    # ---- write per-patient table -------------------------------------------
    out_pat = os.path.join(args.out_dir, 'patients.tsv')
    with open(out_pat, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['mrn','n_images','n_folders','tier','primary_group','patient_groups','split'])
        n_img = collections.Counter()
        for r in rows:
            if r['norm_mrn']:
                n_img[r['norm_mrn']] += 1
        for m in sorted(tier):
            w.writerow([m, n_img[m], len(folders[m]), tier[m], primary.get(m,''),
                        '|'.join(sorted(resolved[m] & keep)), split.get(m,'train')])

    print(f"\nwrote {out_img} and {out_pat}")
    print("images per split:", dict(n_by_split))
    tc = collections.Counter(tier.values())
    print("patients per tier:", dict(tc))
    sc = collections.Counter(split[m] for m in split)
    print("patients per split:", dict(sc))
    print("\nper-group patient counts (test-eligible groups):")
    print("%-32s %6s %6s %6s %6s" % ("group","train","val","test","imgs"))
    img_g = collections.Counter()
    for r in rows:
        if r['bucket'] == 'DX' and r['dx_group'] in keep:
            img_g[r['dx_group']] += 1
    per = collections.defaultdict(collections.Counter)
    for m in eligible:
        per[primary[m]][split[m]] += 1
    for g in sorted(keep, key=lambda x: -img_g[x]):
        c = per[g]
        print("%-32s %6d %6d %6d %6d" % (g, c['train'], c['val'], c['test'], img_g[g]))


if __name__ == '__main__':
    main()

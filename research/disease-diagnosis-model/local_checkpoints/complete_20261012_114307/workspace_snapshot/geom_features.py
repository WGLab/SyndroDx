#!/usr/bin/env python
"""Extract craniofacial shape features from the precomputed organ boxes/geometry.

Motivation: several of the largest classes are *defined* by head shape --
sagittal synostosis is scaphocephaly (long, narrow), metopic is trigonocephaly
(triangular forehead), hemifacial microsomia is left/right asymmetry. These
should be partly recoverable from landmark geometry alone, with no GPU and full
interpretability. This is the baseline any deep model has to beat.

Features per image:
  * 45 normalised inter-organ distances (already scale-normalised upstream)
  * 90 unit-vector components (cos/sin of each pairwise direction)
  * 15 left/right asymmetry magnitudes
  * 30 per-organ box shape (aspect ratio, normalised w, normalised h)
  *  6 global head-shape ratios (cephalic index proxies)
Invalid pairs become NaN; HistGradientBoosting consumes NaN natively.
"""
import csv, ast, os, sys, math, argparse
import numpy as np

csv.field_size_limit(10**9)
ORGANS = ['forehead','jaw','left_cheek','left_ear','left_eye','mouth','nose',
          'right_cheek','right_ear','right_eye']
PAIRS = [(a, b) for i, a in enumerate(ORGANS) for b in ORGANS[i+1:]]
LR = [('left_eye','right_eye'), ('left_ear','right_ear'), ('left_cheek','right_cheek')]
OTHERS = ['nose','mouth','forehead','jaw','left_eye']


def feature_names():
    n  = [f'd|{a}|{b}' for a, b in PAIRS]
    n += [f'{t}|{a}|{b}' for a, b in PAIRS for t in ('cos','sin')]
    n += [f'asym|{l}|{o}' for l, r in LR for o in OTHERS]
    n += [f'{s}|{o}' for o in ORGANS for s in ('ar','w','h')]
    n += ['cephalic_index','face_height_ratio','fh_width_ratio','jaw_width_ratio',
          'eye_sep_norm','midface_ratio']
    return n


def row_features(oc, og):
    f = []
    # normalise everything by interocular distance in pixels
    def ctr(o):
        x0, y0, x1, y1 = oc[o]
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    le, re_ = ctr('left_eye'), ctr('right_eye')
    iod = math.hypot(le[0] - re_[0], le[1] - re_[1]) or 1.0

    lut = {}
    for (a, b), v in og.items():
        lut[(a, b)] = lut[(b, a)] = v
    dist = {}
    for a, b in PAIRS:
        v = lut.get((a, b))
        if v is None or not v.get('valid'):
            f.append(np.nan); dist[(a, b)] = np.nan
        else:
            f.append(v['distance']); dist[(a, b)] = v['distance']
    for a, b in PAIRS:
        v = lut.get((a, b))
        if v is None or not v.get('valid'):
            f += [np.nan, np.nan]
        else:
            dx, dy = v['vector']
            m = math.hypot(dx, dy) or 1.0
            f += [dx / m, dy / m]
    # left/right asymmetry: does organ O sit equidistant from the two sides?
    for l, r in LR:
        for o in OTHERS:
            dl = dist.get(tuple(sorted((l, o))), np.nan)
            dr = dist.get(tuple(sorted((r, o))), np.nan)
            f.append(abs(dl - dr) if (dl == dl and dr == dr) else np.nan)
    # per-organ box shape
    for o in ORGANS:
        x0, y0, x1, y1 = oc[o]
        w, h = (x1 - x0), (y1 - y0)
        f += [w / h if h else np.nan, w / iod, h / iod]
    # global head-shape ratios
    fh = oc['forehead']; jw = oc['jaw']
    lc, rc = oc['left_cheek'], oc['right_cheek']
    head_w = abs(rc[2] - lc[0]); head_h = abs(jw[3] - fh[1])
    f += [head_w / head_h if head_h else np.nan,          # cephalic index proxy
          head_h / iod,
          (fh[2] - fh[0]) / iod,                           # forehead width
          (jw[2] - jw[0]) / iod,                           # jaw width
          iod / head_w if head_w else np.nan,
          (fh[2] - fh[0]) / (jw[2] - jw[0]) if (jw[2] - jw[0]) else np.nan]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default="/controlled-data/craniofacial_data/"
                    "craniofacial_metadata_frontal_updated_123025_deduplicated.csv")
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'geom_features.npz'))
    args = ap.parse_args()

    ids, X, bad = [], [], 0
    with open(args.csv, newline='', encoding='utf-8', errors='replace') as f:
        for i, r in enumerate(csv.DictReader(f, delimiter='\t')):
            try:
                oc = ast.literal_eval(r['organ_coordinates'])
                og = ast.literal_eval(r['organ_geometry'])
                X.append(row_features(oc, og))
            except Exception:
                bad += 1
                X.append([np.nan] * len(feature_names()))
            ids.append(r['index'])
            if (i + 1) % 25000 == 0:
                print(f"  {i+1:,}", flush=True)
    X = np.asarray(X, dtype=np.float32)
    np.savez_compressed(args.out, X=X, ids=np.array(ids), names=np.array(feature_names()))
    print(f"wrote {args.out}  X={X.shape}  parse failures={bad}  "
          f"NaN fraction={np.isnan(X).mean():.3f}")


if __name__ == '__main__':
    main()

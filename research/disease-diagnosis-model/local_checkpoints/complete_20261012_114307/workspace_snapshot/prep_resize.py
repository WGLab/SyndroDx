#!/usr/bin/env python
"""Pre-resize the cropped faces so the SSL dataloader is not JPEG-decode bound.

The crops are median 1403x1484 (up to 4416x4320), ~96 GB total, and every one is
decoded in full just to produce a 224px crop. At 384px short side the set is
~6 GB -- small enough to sit in page cache -- and decode cost drops ~25x, which
is what makes a 2-CPU-per-GPU allocation viable on a busy cluster.

384 (not 256) keeps headroom: the global crop scale floor of 0.45 takes a 258px
region, so 224 is still a downscale rather than an upscale.
"""
import csv, os, sys, argparse
from multiprocessing import Pool
from PIL import Image

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PREFIX = '/controlled-data/craniofacial_data/cropped_faces/'


def dest_for(src, out_root):
    rel = src[len(SRC_PREFIX):] if src.startswith(SRC_PREFIX) else \
          os.path.join('other', src.lstrip('/').replace('/', '_'))
    return os.path.join(out_root, rel)


def one(job):
    src, dst, size, quality = job
    try:
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return 'skip'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im = Image.open(src)
        im.draft('RGB', (size * 2, size * 2))      # fast DCT-domain downscale
        im = im.convert('RGB')
        w, h = im.size
        s = size / min(w, h)
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(dst, 'JPEG', quality=quality, optimize=True)
        return 'ok'
    except Exception as e:
        return f'fail:{src}:{type(e).__name__}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(HERE, 'labels.tsv'))
    ap.add_argument('--out-root', default='/opt/craniofacial/disease-diagnosis-model/faces_384')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--quality', type=int, default=92)
    ap.add_argument('--procs', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--manifest', default=os.path.join(HERE, 'resized_manifest.tsv'))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels, newline=''), delimiter='\t'))
    jobs, man = [], []
    for r in rows:
        d = dest_for(r['cropping_path'], args.out_root)
        jobs.append((r['cropping_path'], d, args.size, args.quality))
        man.append((r['img_id'], d))
    print(f"{len(jobs):,} images -> {args.out_root} at {args.size}px, "
          f"{args.procs} procs", flush=True)

    ok = skip = 0
    fails = []
    with Pool(args.procs) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs, chunksize=64), 1):
            if res == 'ok': ok += 1
            elif res == 'skip': skip += 1
            else: fails.append(res)
            if i % 10000 == 0:
                print(f"  {i:,}/{len(jobs):,}  ok={ok:,} skip={skip:,} fail={len(fails)}", flush=True)

    with open(args.manifest, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['img_id', 'resized_path'])
        w.writerows(man)
    print(f"\ndone: ok={ok:,} skip={skip:,} fail={len(fails)}")
    for x in fails[:10]:
        print("  ", x)
    print(f"manifest -> {args.manifest}")


if __name__ == '__main__':
    main()

"""Conservative local exact/perceptual duplicate quarantine before expert scoring."""
import os,sys,json,hashlib,csv,collections
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
V1=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
def load(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n');os.chmod(p,0o600)
def phash(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);small=cv2.resize(gray,(32,32)).astype(np.float32)
    dct=cv2.dct(small)[:8,:8].flatten()[1:];bits=dct>np.median(dct)
    return np.packbits(bits).view(np.uint64)[0]
def main():
    os.umask(0o077);cv2.setNumThreads(1)
    records=load(ROOT/'private/eligible.json')['rows'];qhash=[]
    for r in records:
        im=cv2.imread(r['crop_path'])
        if im is None:raise RuntimeError('expert_crop_decode_failure')
        qhash.append(phash(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
    qhash=np.asarray(qhash,dtype=np.uint64);close=set();table=np.asarray([bin(i).count('1') for i in range(256)],dtype=np.uint8)
    def check(values):
        for h in values:
            dist=table[np.bitwise_xor(qhash,h).view(np.uint8).reshape(-1,8)].sum(1)
            close.update(records[i]['patient'] for i in np.flatnonzero(dist<=4))
    cache=PROJECT/'transface_arcface_identity_cpt_20261026/cache_packed'
    idx=load(cache/'index.private.json')['images'];raw=np.memmap(cache/'train_rgb.private.bin',dtype=np.uint8,mode='r')
    done=0;buffer=[]
    for spec in idx.values():
        image=np.asarray(raw[spec['offset']:spec['offset']+spec['bytes']]).reshape(spec['height'],spec['width'],3)
        buffer.append(phash(image));buffer.append(phash(image[:,::-1].copy()));done+=1
        if len(buffer)>=256:
            check(buffer);buffer=[];save(ROOT/'near_audit_progress.json',dict(state='checking_CHOP_training_duplicates',completed=done,total=len(idx)))
    if buffer:check(buffer)
    gmdb=PROJECT/'gestaltmatcher_arc_comparison_20261027';train=load(gmdb/'private/budget_100.private.json')['indices']
    images=np.load(gmdb/'private/images.uint8.npy',mmap_mode='r');buffer=[]
    for i in train:
        image=np.asarray(images[i]);buffer.extend([phash(image),phash(image[:,::-1].copy())])
        if len(buffer)>=256:check(buffer);buffer=[]
    if buffer:check(buffer)
    kept=[r for r in records if r['patient'] not in close]
    owners=collections.defaultdict(set)
    for r in kept:owners[r['crop_sha256']].add(r['patient'])
    crosscopy=set().union(*(v for v in owners.values() if len(v)>1)) if owners else set()
    kept=[r for r in kept if r['patient'] not in crosscopy]
    save(ROOT/'private/released.json',dict(rows=kept))
    counts=collections.defaultdict(set)
    for r in kept:counts[r['label']].add(r['patient'])
    eligible_labels={k for k,v in counts.items() if len(v)>=2}
    report=dict(state='released_for_exploratory_expert_source_evaluation',images=len(kept),patients=len({r['patient'] for r in kept}),diseases=len(counts),
        retrieval_evaluable_diseases=len(eligible_labels),retrieval_evaluable_patients=len({r['patient'] for r in kept if r['label'] in eligible_labels}),
        near_duplicate_patient_quarantine=len(close),cross_patient_exact_copy_quarantine=len(crosscopy),
        known_patient_alias_and_exact_hash_exposure_checked=True,perceptual_duplicate_check='63-bit DCT hash distance <=4, both flips, all CHOP CPT and filtered GMDB training images',
        release_sha256=hashlib.sha256((ROOT/'private/released.json').read_bytes()).hexdigest(),
        limitations=['Perceptual hashing does not prove absence of all near duplicates or unknown identities.','Expert-source labels have not undergone new clinical adjudication.'],
        protocol='patient-mean embeddings; leave-one-patient-out expert gallery; same fixed candidate diseases for all four models; singleton diseases excluded from retrieval query denominator')
    save(ROOT/'release.json',report)
if __name__=='__main__':main()

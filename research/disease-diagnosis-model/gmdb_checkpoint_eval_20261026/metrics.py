"""Official syndrome-macro rank metric, without transmitting any records."""
import numpy as np
from sklearn.metrics import pairwise_distances

TOP_K=(1,5,10,30)

def evaluate(gallery,queries,features,index,strict=False):
    if strict:
        subjects={r['subject'] for r in gallery}
        queries=[r for r in queries if r['subject'] not in subjects]
    if not queries:raise RuntimeError('empty_query_set')
    gallery_ids=[r['image_id'] for r in gallery];query_ids=[r['image_id'] for r in queries]
    if len(set(gallery_ids))!=len(gallery_ids):raise RuntimeError('duplicate_gallery_image')
    g=features[[index[x] for x in gallery_ids]]
    labels=np.asarray([r['label'] for r in gallery]);targets=np.asarray([r['label'] for r in queries])
    hits=np.zeros((len(queries),len(TOP_K)),dtype=np.float64)
    for start in range(0,len(queries),128):
        q=features[[index[x] for x in query_ids[start:start+128]]]
        distances=pairwise_distances(q,g,metric='cosine')
        if not np.isfinite(distances).all():raise RuntimeError('nonfinite_retrieval_distances')
        ranks=np.argsort(distances,axis=1)
        for local,rank in enumerate(ranks):
            ordered=labels[rank]
            ordered=ordered[np.sort(np.unique(ordered,return_index=True)[1])]
            for j,k in enumerate(TOP_K):hits[start+local,j]=targets[start+local] in ordered[:k]
    classes=np.unique(targets)
    macro=np.stack([hits[targets==label].mean(0) for label in classes]).mean(0)
    coverage=np.mean(np.isin(targets,np.unique(labels)))
    return dict(query_images=len(queries),query_patients=len({r['subject'] for r in queries}),
        query_diseases=len(classes),gallery_images=len(gallery),gallery_diseases=len(set(labels)),
        gallery_coverage=float(coverage),macro={f'top{k}':float(v) for k,v in zip(TOP_K,macro)},
        micro={f'top{k}':float(v) for k,v in zip(TOP_K,hits.mean(0))})

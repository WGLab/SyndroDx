"""Local-only facial phenotype captions. No disease names or inferred observations."""
import numpy as np
from common import *
from transformers import BertTokenizer

SOURCE=PROJECT/'hpo_query_fusion_20261101'
STAGED=PROJECT/'hpo_fusion_staged_20261031'
# Public ontology IDs: conservative external facial morphology, not functional/systemic findings.
CODES='0000154 0000215 0000219 0000272 0000276 0000278 0000286 0000289 0000308 0000311 0000316 0000319 0000322 0000325 0000343 0000347 0000358 0000369 0000426 0000431 0000437 0000446 0000451 0000455 0000456 0000494 0000506 0000537 0000582 0000601 0000637 0000664 0002002 0002007 0003194 0005274 0005278 0005280 0010803 0010804 0010806 0011220 0011825 0011829 0011831 0011832 0011833 0012745 0033142 0045025'.split()

def main():
    os.umask(0o077)
    source=load(SOURCE/'data_receipt.json');features=load(SOURCE/'features/receipt.json')
    assert source['state']=='passed' and features['state']=='complete'
    assert digest(source['gmdb_path'])==source['gmdb_sha256']
    assert digest(features['tokens_path'])==features['tokens_sha256']
    assert digest(SOURCE/'private/local_mask.npy')==features['mask_sha256']
    oldreceipt=load(STAGED/'data_receipt.json')
    assert digest(STAGED/'private/term_index.json')==oldreceipt['files']['term_index.json']
    terms=load(STAGED/'private/term_index.json')['term_ids']
    ontology=load(OLD/'private/ontology.json')['terms']
    whitelist={'HP:'+c:ontology['HP:'+c]['name'] for c in CODES}
    save(WORK/'facial_term_whitelist.json',dict(terms=whitelist,policy='external morphology recorded at patient level; not per-image expert grounding',ontology_sha256=digest(OLD/'private/ontology.json')))
    tokenizer=BertTokenizer.from_pretrained(WORK/'assets/bert-base-uncased',local_files_only=True)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    with np.load(source['gmdb_path']) as d:arrays={k:d[k].copy() for k in d.files}
    train_set=set(arrays['train'].tolist());val_set=set(arrays['validation'].tolist())
    texts=[];members=[];observations=[];lost=0
    for i in sorted(train_set|val_set):
        codes=sorted({terms[t-1] for t,state in zip(arrays['ids'][i],arrays['states'][i]) if t>0 and state==0 and terms[t-1] in whitelist})
        if len(codes)<2:continue
        text='Recorded facial features: '+'; '.join(whitelist[c] for c in codes)+'.'
        if len(tokenizer.encode(text))>128:
            lost+=1;continue
        members.append(i);texts.append(text);observations.append(codes)
    vocab={t:i for i,t in enumerate(sorted(set(texts)))}
    groups=np.array([vocab[t] for t in texts],np.int64)
    members=np.array(members,np.int64)
    train=np.array([j for j,i in enumerate(members) if int(i) in train_set],np.int64)
    val=np.array([j for j,i in enumerate(members) if int(i) in val_set],np.int64)
    assert len(train)>=100 and len(val)>=30 and not set(train)&set(val)
    tokenized=tokenizer(texts,padding='max_length',truncation=False,max_length=128,return_tensors='np')
    codes_sorted=sorted(whitelist);code_lookup={c:i for i,c in enumerate(codes_sorted)}
    present=np.zeros((len(members),len(codes_sorted)),bool)
    for i,codes in enumerate(observations):
        for c in codes:present[i,code_lookup[c]]=True
    np.savez(WORK/'private/paired.npz',members=members,groups=groups,train=train,validation=val,
        input_ids=tokenized['input_ids'],attention_mask=tokenized['attention_mask'],present=present,
        labels=arrays['labels'][members],native_features=arrays['features'][members])
    save(WORK/'private/captions.json',dict(texts=texts))
    counts=dict(train_patients=len(train),validation_patients=len(val),train_caption_groups=len(set(groups[train])),
        validation_caption_groups=len(set(groups[val])),shared_caption_groups=len(set(groups[train])&set(groups[val])),
        excluded_long_captions=lost,whitelisted_public_terms=len(whitelist),training_diseases=int(len(set(arrays['labels'][members[train]]))),
        validation_diseases=int(len(set(arrays['labels'][members[val]]))))
    save(WORK/'data_receipt.json',dict(state='passed',counts=counts,paired_sha256=digest(WORK/'private/paired.npz'),
        captions_sha256=digest(WORK/'private/captions.json'),source_data_receipt_sha256=digest(SOURCE/'data_receipt.json'),
        source_gmdb_sha256=source['gmdb_sha256'],features=features,mask_path=str(SOURCE/'private/local_mask.npy'),
        visual=load(SOURCE/'protocol.json')['visual'],patient_overlap=0,exact_crop_overlap=0,
        supervised_diagnosis_loss=False,expert_BCH_or_GMDB_test_used=False,
        BCH_cross_source_identity_audit='manifest_unavailable',updated_at_utc=now()))
    save(WORK/'protocol.json',dict(state='locked_before_training',seeds=[11,29],conditions=['real','shuffled'],
        batch_size=32,max_epochs=20,min_epochs=5,patience=5,learning_rate=1e-5,weight_decay=.05,
        warmup_steps=50,min_lr=1e-6,gradient_clip=1.,query_tokens=32,hidden_size=768,layers=12,
        cross_attention_frequency=2,projection_dimension=256,max_text_length=128,
        train_precision='BF16 with FP32 similarity and loss reductions',
        initialization='official BERT-base initialization; new cross-attention and query parameters; no prealigned Q-Former claimed',
        source='pinned official LAVIS Qformer.py; isolated transformers4.26.1',
        objective='equal-weight ITC + ITM + image-grounded language modeling; no disease classification head/loss',
        visual='frozen pinned CHOP CPT -> GMDB ArcFace TransFace; cached 144x768 local tokens per image; up to 4 images per patient',
        attention='ITC visual queries never receive paired text; ITM bidirectional query/text; ITG causal text attends cached visual queries',
        clinical_adaptation='positive recorded external morphology only; patient-level weak captions, no per-image grounding claim; no HPO absence inference',
        negatives='identical captions are ITC multi-positives; ITM excludes identical and subset/superset term sets; remaining mismatches are contrastive training constructs, not clinical absence labels',
        shuffled_control='fixed train-only cyclic patient permutation with no self pairs and no identical captions; clean validation unchanged',
        selection='development mean of image-to-caption and caption-to-image group-balanced R@1; tie R@5; retain epoch0 if all updates regress',
        evaluation='cross-modal retrieval, image-only disease retrieval against train gallery, native visual retrieval on identical filtered cohort; no comparison to unfiltered 320-patient scores',
        data_policy='same approved patient split; no expert/BCH/test model selection; BCH manifest unavailable; repeated development evaluations exploratory',
        network_during_patient_processing=False,clinical_media_export=False,updated_at_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(WORK/'preparation_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

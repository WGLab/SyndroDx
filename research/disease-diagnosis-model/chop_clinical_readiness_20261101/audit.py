"""CHOP expert input/candidate audit; patient-level details remain on HPC."""
import collections
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parent
PROJECT=WORK.parent
sys.path.insert(0,str(PROJECT/'clinical_evidence_agent_20261101'))
from common import load,save,sha,now,write_jsonl


def parse_mondo(path):
    terms={};aliases={};exact=collections.defaultdict(set)
    for block in path.read_text().split('[Term]')[1:]:
        match=re.search(r'^id: (MONDO:\d{7})$',block,re.M)
        if not match:continue
        mid=match[1]
        name=re.search(r'^name: (.+)$',block,re.M)
        terms[mid]={'name':name[1] if name else mid,'obsolete':'is_obsolete: true' in block,
                    'replaced_by':re.findall(r'^replaced_by: (MONDO:\d{7})',block,re.M)}
        for alt in re.findall(r'^alt_id: (MONDO:\d{7})$',block,re.M):aliases[alt]=mid
        if terms[mid]['obsolete']:continue
        for line in block.splitlines():
            if line.startswith('xref: OMIM:') and 'MONDO:equivalentTo' in line:
                m=re.match(r'xref: (OMIM:\d+)',line)
                if m:exact[m[1]].add(mid)
            if line.startswith('property_value: skos:exactMatch '):
                m=re.search(r'(?:OMIM:|https?://(?:identifiers.org/omim/|omim.org/entry/))(\d+)',line)
                if m:exact['OMIM:'+m[1]].add(mid)
    return terms,aliases,exact


def canonical(code,terms,aliases):
    visited=set()
    while code not in visited:
        visited.add(code)
        if code in aliases:code=aliases[code];continue
        record=terms.get(code)
        if record is None:return None
        if not record['obsolete']:return code
        if len(record['replaced_by'])!=1:return None
        code=record['replaced_by'][0]
    return None


def exact_class_mapping(omim_field,exact):
    codes=sorted(set(re.findall(r'\d{6}',str(omim_field))))
    targets=[exact.get('OMIM:'+code,set()) for code in codes]
    # Multiple OMIM entries are accepted only if ALL independently resolve to the
    # same single exact MONDO concept. No fuzzy, ancestor or broad matching.
    if targets and all(len(x)==1 for x in targets) and len(set.union(*targets))==1:
        return next(iter(targets[0]))
    return None


def main():
    (WORK/'private').mkdir(parents=True,exist_ok=True,mode=0o700)
    old=PROJECT/'chop_hpo_increment_20261101/expert'
    protocol=load(old/'protocol.json');source=old/'private/arrays.npz';idxpath=old/'private/index.json'
    if sha(source)!=protocol['arrays_sha256'] or sha(idxpath)!=protocol['index_sha256']:
        raise ValueError('expert_snapshot_changed')
    idx=load(idxpath)
    with np.load(source) as f:data={k:f[k].copy() for k in f.files}
    if len(set(idx['groups']))!=len(idx['groups']):raise ValueError('duplicate_patient_groups')
    reg=load(PROJECT/'model_releases/CURRENT.json')
    if sha(reg['manifest'])!=reg['sha256']:raise ValueError('release_manifest_changed')
    manifest=load(reg['manifest']);release=Path(reg['manifest']).parent
    def asset(name):
        entry=manifest['artifacts'][name];p=release/entry['path']
        if sha(p)!=entry['sha256']:raise ValueError('asset_hash_mismatch')
        return load(p)
    classes=asset('disease_vocabulary')['labels'];vocab=asset('HPO_term_index')['term_ids']
    old_vocab=load(PROJECT/'chop_hpo_increment_20261101/private/term_index.json')['term_ids']
    if vocab!=old_vocab:raise ValueError('HPO_vocabulary_order_mismatch')
    inverse={i+1:t for i,t in enumerate(vocab)}
    mondo=PROJECT/'ontology/mondo.obo'
    if sha(mondo)!=protocol['mondo_sha256']:raise ValueError('ontology_changed')
    terms,aliases,exact=parse_mondo(mondo)
    metadata=Path('/controlled-data/GMDB_v1.1.0/gmdb_metadata_v1.1.0/gmdb_metadata/gmdb_syndromes_v1.1.0.tsv')
    if sha(metadata)!=protocol['GMDB_disease_metadata_sha256']:raise ValueError('disease_dictionary_changed')
    all_classes={};ambiguous_targets=set()
    with metadata.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            mapped=exact_class_mapping(row['OMIM'],exact)
            all_classes[str(row['syndrome_id'])]={'name':row['syndrome_name'],'mapped':mapped}
            if mapped is None:
                for code in re.findall(r'\d{6}',row['OMIM']):ambiguous_targets.update(exact.get('OMIM:'+code,set()))
    candidate_for_class=[all_classes.get(str(c),{}).get('mapped') or 'UNMAPPED_GMDB:'+str(c) for c in classes]
    candidate_codes=sorted(set(candidate_for_class));candidate_index={c:i for i,c in enumerate(candidate_codes)}
    map_array=np.array([candidate_index[c] for c in candidate_for_class],np.int64)
    canonical_labels=[canonical(label,terms,aliases) for label in idx['labels']]
    canonical_y=np.array([candidate_index.get(label,-1) for label in canonical_labels],np.int64)
    valid=data['ids']!=0
    known=valid&np.isin(data['states'],[0,1]);recorded=valid.any(1);eligible=known.any(1)
    covered=canonical_y>=0
    all_reference={v['mapped'] for v in all_classes.values() if v['mapped']}
    head_labels=set(load(PROJECT/'matched_craniofacial_benchmark_20261031/private/records.private.json')['labels'])
    full_head={all_classes.get(str(c),{}).get('mapped') for c in head_labels}-{None}
    reasons=collections.Counter();disease_reasons=collections.defaultdict(set);rows=[]
    for i,(group,label,canonical_label) in enumerate(zip(idx['groups'],idx['labels'],canonical_labels)):
        if covered[i]:reason='covered_by_exact_current_classifier_mapping'
        elif canonical_label is None:reason='invalid_obsolete_or_unresolved_MONDO_concept'
        elif canonical_label in full_head:reason='outside_paired_classifier_but_in_full_visual_training_vocabulary'
        elif canonical_label in all_reference:reason='outside_trained_classifier_but_exact_in_GMDB_reference'
        elif canonical_label in ambiguous_targets:reason='GMDB_reference_has_ambiguous_exact_crosswalk'
        else:reason='no_exact_equivalent_in_GMDB_reference'
        reasons[reason]+=1;disease_reasons[reason].add(label)
        rows.append({'group_id':group,'original_label':label,'canonical_label':canonical_label,
            'disease_name':terms.get(canonical_label,{}).get('name'),
            'coverage_reason':reason,'legacy_covered':bool(data['covered'][i]),'current_exact_covered':bool(covered[i]),
            'recorded_HPO_terms':int(valid[i].sum()),'known_polarity_HPO_terms':int(known[i].sum()),
            'target_population_adjudication':'pending','diagnosis_adjudication':'pending'})
    write_jsonl(WORK/'private/patient_coverage_review.jsonl',rows)
    # Link provenance from the existing pseudonymous index, without real-MRN mapping.
    cohort_path=PROJECT/'matched_craniofacial_benchmark_20261031/expert/private/cohort.json'
    cohort_receipt=load(PROJECT/'matched_craniofacial_benchmark_20261031/expert/audit.json')
    if sha(cohort_path)!=cohort_receipt['cohort_sha256']:raise ValueError('cohort_changed')
    key_to_group={}
    for row in load(cohort_path)['rows']:
        group=row['group_id'] or row['patient_key']
        if group in set(idx['groups']):
            if row['patient_key'] in key_to_group and key_to_group[row['patient_key']]!=group:raise ValueError('identity_conflict')
            key_to_group[row['patient_key']]=group
    dbpath=PROJECT/'chop_patient_linkage_20261101/private/linked_metadata.sqlite'
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    by_source=collections.Counter();contexts=collections.Counter();review=[]
    for key,code,assertion,source_name,source_rows,event,context in db.execute('SELECT patient_key,code,assertion,source,source_rows,event_date,context_present FROM hpo'):
        if key not in key_to_group:continue
        by_source[source_name]+=1
        if source_name!='Arcus_clinician_HPO':continue
        contexts['records_with_local_context' if context else 'records_without_local_context']+=1
        contexts['records_with_record_creation_date' if event else 'records_without_date']+=1
        review.append({'patient_key':key,'group_id':key_to_group[key],'HPO':code,'recorded_assertion':assertion,
            'source':source_name,'source_database':str(dbpath),'record_creation_date':event,
            'context_available_in_source':bool(context),'source_rows':int(source_rows),
            'reviewed_assertion':None,'reviewed_experiencer':None,'clinical_event_time':None,
            'evidence_available_at':None,'decision_time':None,'reviewer_ref':None,'source_document_reference':None,
            'training_admitted':False,'review_note':None})
    db.close()
    write_jsonl(WORK/'private/hpo_review_queue.jsonl',review)
    pending=collections.Counter((r['group_id'],r['HPO']) for r in review)
    missing=sum((g,inverse[int(t)]) not in pending for row,g in enumerate(idx['groups']) for t in data['ids'][row] if t)
    if missing:raise ValueError('cached_HPO_without_linked_source')
    # Store a separate strict evaluation view; historical arrays remain unchanged.
    strict_ids=data['ids'].copy();strict_states=data['states'].copy()
    strict_ids[~known]=0;strict_states[strict_ids==0]=0
    np.savez_compressed(WORK/'private/strict_arrays.npz',features=data['features'],ids=strict_ids,states=strict_states,
        labels=canonical_y,recorded_HPO=recorded,eligible_HPO=eligible,covered=covered,
        queries=data['queries'],class_to_candidate=map_array)
    save(WORK/'private/strict_index.json',{'groups':idx['groups'],
         'labels':[c if c is not None else 'UNRESOLVED_LABEL:'+old for c,old in zip(canonical_labels,idx['labels'])],
         'candidates':candidate_codes})
    audit={'state':'complete','patients':len(rows),'diseases':len(set(idx['labels'])),
        'recorded_HPO_patients':int(recorded.sum()),'recorded_HPO_tokens':int(valid.sum()),
        'tokens_by_state':{str(s):int((valid&(data['states']==s)).sum()) for s in [0,1,2]},
        'known_polarity_HPO_patients':int(eligible.sum()),'legacy_classifier_covered_patients':int(data['covered'].sum()),
        'strict_exact_classifier_covered_patients':int(covered.sum()),
        'canonicalization_recovered_patients':int((covered&~data['covered']).sum()),
        'canonicalization_lost_patients':int((~covered&data['covered']).sum()),
        'recorded_and_covered_patients':int((recorded&covered).sum()),
        'known_polarity_and_covered_patients':int((eligible&covered).sum()),
        'current_classifier_classes':len(classes),'exact_mapped_classifier_classes':sum(not x.startswith('UNMAPPED') for x in candidate_for_class),
        'coverage_failure_counts':dict(reasons),'diseases_by_coverage_reason':{k:len(v) for k,v in disease_reasons.items()},
        'full_visual_training_vocabulary_coverage_patients':sum(x in full_head for x in canonical_labels),
        'all_GMDB_reference_exact_coverage_patients':sum(x in all_reference for x in canonical_labels),
        'linked_HPO_rows_by_source':dict(by_source),'clinician_source_context_counts':dict(contexts),
        'HPO_review_records':len(review),'HPO_review_patient_groups':len({r['group_id'] for r in review}),
        'raw_notes_read':False,'images_read':False,'MRN_mapping_read':False,'new_clinical_assertions_created':0,
        'paired_multimodal_efficacy_estimable':bool((eligible&covered).any()),
        'historical_state2_results_status':'exploratory unresolved-annotation sensitivity, not validated clinical multimodal efficacy',
        'legacy_protocol_sha256':sha(old/'protocol.json'),'arrays_sha256':sha(WORK/'private/strict_arrays.npz'),
        'index_sha256':sha(WORK/'private/strict_index.json'),'model_manifest_sha256':reg['sha256'],
        'updated_at_utc':now()}
    save(WORK/'audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        save(WORK/'failure.json',{'state':'failed','error_type':type(e).__name__,
            'locations':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)],'updated_at_utc':now()})
        raise SystemExit(1) from None

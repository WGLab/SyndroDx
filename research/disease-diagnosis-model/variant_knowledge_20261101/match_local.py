"""Offline candidate evidence linkage; never turn a public assertion into a patient diagnosis."""
import collections
import csv
import datetime
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
os.umask(0o077)
PATTERN=re.compile(r'\b((?:N[MCGPR]|X[MRP])_\d+\.\d+)(?:\([A-Za-z0-9_.-]+\))?\s*:\s*([cgnmrp]\.[A-Za-z0-9_*+?=><()\[\]./\-]+)')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,obj):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()


def expressions(value):
 result=set()
 for ref,change in PATTERN.findall(value or ''):
  change=change.rstrip('.,;')
  while change.endswith(')') and change.count(')')>change.count('('):change=change[:-1]
  result.add(ref+':'+change)
 return result


def public_expression(value):
 if ':' not in value:return value
 reference,change=value.split(':',1)
 return reference.split('(',1)[0]+':'+change


def clingen_rows(path):
 with path.open() as f:
  for line in f:
   if 'GENE SYMBOL' in line:
    yield from csv.DictReader(itertools.chain([line],f));return
 raise RuntimeError('missing_ClinGen_header')


def main():
 (ROOT/'private').mkdir(exist_ok=True,mode=0o700)
 downloads=json.loads((ROOT/'download_receipt.json').read_text())
 assert downloads['state']=='complete'
 for name,receipt in downloads['sources'].items():assert sha(ROOT/'public'/name)==receipt['sha256']
 source=ROOT.parent/'chop_patient_linkage_20261101/private/variant_review/train_face_variant_candidates.tsv'
 receipt=json.loads((ROOT.parent/'chop_patient_linkage_20261101/variant_review_receipt.json').read_text())
 assert sha(source)==receipt['sha256']
 with source.open() as f:candidates=list(csv.DictReader(f,delimiter='\t'))
 targets=collections.defaultdict(set);symbols=set()
 for i,r in enumerate(candidates):
  for token in expressions(r['hgvs']):targets[token].add(i)
  if re.fullmatch(r'[A-Za-z][A-Za-z0-9-]*',r['gene_symbol']):symbols.add(r['gene_symbol'].upper())
 gene_validity=collections.defaultdict(list);dosage=collections.defaultdict(list);orpha=collections.defaultdict(list)
 for name,table in [('clingen_gene_validity.csv',gene_validity),('clingen_gene_dosage.csv',dosage)]:
  for r in clingen_rows(ROOT/'public'/name):
   gene=r['GENE SYMBOL'].strip().upper()
   if gene in symbols:table[gene].append(r)
 tree=ET.parse(ROOT/'public/orphadata_genes.xml')
 for disorder in tree.iter('Disorder'):
  for association in disorder.findall('./DisorderGeneAssociationList/DisorderGeneAssociation'):
   gene=(association.findtext('./Gene/Symbol') or '').upper()
   if gene in symbols:orpha[gene].append(dict(orpha_id=disorder.findtext('OrphaCode'),disease_name=disorder.findtext('Name'),
      association_type=association.findtext('./DisorderGeneAssociationType/Name'),association_status=association.findtext('./DisorderGeneAssociationStatus/Name')))
 del tree
 matches=collections.defaultdict(set);requested=set()
 save(ROOT/'matching_status.json',dict(state='matching_exact_reference_HGVS_locally',candidate_records=len(candidates),updated_at_utc=now()))
 with gzip.open(ROOT/'public/clinvar_hgvs4variation.txt.gz','rt') as f:
  for line in f:
   if line.startswith('#Symbol\t'):header=line.rstrip('\n').split('\t');break
  ix={v:i for i,v in enumerate(header)}
  for n,line in enumerate(f,1):
   cells=line.rstrip('\n').split('\t')
   if len(cells)!=len(header):raise RuntimeError('unexpected_HGVS_row_width')
   for field,kind in [('NucleotideExpression','exact_versioned_nucleotide_expression'),('ProteinExpression','protein_expression_only_candidate')]:
    token=public_expression(cells[ix[field]])
    if token in targets:
     vid=cells[ix['VariationID']];requested.add(vid)
     for i in targets[token]:matches[i].add((vid,kind,token))
   if n%1000000==0:save(ROOT/'matching_status.json',dict(state='scanning_public_HGVS',public_rows_scanned=n,candidate_records_with_hits=len(matches),updated_at_utc=now()))
 variants=collections.defaultdict(list)
 fields=['VariationID','#AlleleID','Type','Name','GeneSymbol','ClinicalSignificance','ReviewStatus','PhenotypeIDS','PhenotypeList','Assembly','Origin','LastEvaluated']
 with gzip.open(ROOT/'public/clinvar_variant_summary.txt.gz','rt') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   if r['VariationID'] in requested:variants[r['VariationID']].append({k:r.get(k,'') for k in fields})
 counter=collections.Counter();patients=collections.defaultdict(set)
 output=ROOT/'private/variant_evidence_candidates.jsonl';tmp=output.with_suffix('.tmp')
 with tmp.open('w') as f:
  for i,r in enumerate(candidates):
   gene=r['gene_symbol'].upper();m=sorted(matches.get(i,set()));ids=sorted({x[0] for x in m})
   nucleotide={x[0] for x in m if x[1]=='exact_versioned_nucleotide_expression'}
   categories=[]
   if m:categories.append('any_HGVS_candidate')
   if nucleotide:categories.append('nucleotide_expression_candidate')
   if len(nucleotide)==1:categories.append('single_nucleotide_VariationID_candidate')
   if len(nucleotide)>1:categories.append('ambiguous_nucleotide_VariationID_candidates')
   if m and not nucleotide:categories.append('protein_only_candidate')
   if gene_validity[gene]:categories.append('ClinGen_gene_disease_knowledge')
   if dosage[gene]:categories.append('ClinGen_dosage_knowledge')
   if orpha[gene]:categories.append('Orphadata_gene_disease_knowledge')
   for category in categories:counter[category]+=1;patients[category].add(r['patient_key'])
   row=dict(variant_key=r['variant_key'],patient_key=r['patient_key'],group_id=r['group_id'],
       source_gene=r['gene_symbol'],source_HGVS=r['hgvs'],source_assessment_code=r['assessment_code'],
       matches=[dict(VariationID=v,match_type=k,HGVS=t) for v,k,t in m],
       public_variant_assertions={v:variants[v] for v in ids},gene_disease_knowledge=gene_validity[gene],dosage_knowledge=dosage[gene],orphadata_knowledge=orpha[gene],
       status='candidate_evidence_only_local_assertion_genotype_and_condition_review_pending',diagnosis_training_admitted=False,
       patient_pathogenicity_inferred=False)
   f.write(json.dumps(row)+'\n')
 tmp.replace(output)
 summary=dict(state='candidate_public_evidence_linked_not_diagnostic_truth',patients=len({r['patient_key'] for r in candidates}),records=len(candidates),
       record_counts=dict(counter),patient_counts={k:len(v) for k,v in patients.items()},
       private_output_sha256=sha(output),source_candidate_sha256=receipt['sha256'],public_source_sha256={k:v['sha256'] for k,v in downloads['sources'].items()},
       clinical_code_dictionary_decoded=False,patient_variant_assertions_verified=False,genome_assembly_inferred=False,
       no_private_outbound_requests=True,diagnosis_training_admissions=0,locked_v1_modified=False,updated_at_utc=now())
 save(ROOT/'matching_receipt.json',summary);save(ROOT/'matching_status.json',dict(state='complete_candidate_matching',updated_at_utc=now()))
 print(json.dumps({k:summary[k] for k in ['state','patients','records','record_counts','patient_counts']}))


if __name__=='__main__':
 try:main()
 except Exception as e:
  save(ROOT/'matching_failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()));raise

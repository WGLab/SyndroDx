"""Download only fixed public bulk URLs; no patient-derived query or input."""
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
SOURCES={
 'clingen_gene_validity.csv':'https://search.clinicalgenome.org/kb/gene-validity/download',
 'clingen_gene_dosage.csv':'https://search.clinicalgenome.org/kb/gene-dosage/download',
 'orphadata_genes.xml':'https://www.orphadata.com/data/xml/en_product6.xml',
 'clinvar_variant_summary.txt.gz':'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz',
 'clinvar_hgvs4variation.txt.gz':'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/hgvs4variation.txt.gz'}
LOCK=threading.Lock();STATUS={}
os.umask(0o077)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,data):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def status(name,**values):
 with LOCK:
  STATUS[name]=dict(**values,updated_at_utc=now())
  save(ROOT/'download_status.json',dict(state='fetching_public_resources',resources=STATUS,outbound_patient_data=False,updated_at_utc=now()))


def fetch(name,url):
 target=ROOT/'public'/name;receipt=ROOT/'receipts'/(name+'.json')
 if receipt.exists() and target.exists():
  r=json.loads(receipt.read_text());h=hashlib.sha256()
  with target.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  if h.hexdigest()==r['sha256']:
   status(name,state='complete',bytes=r['bytes'],sha256=r['sha256']);return r
 for attempt in range(1,4):
  try:
   status(name,state='connecting',attempt=attempt,bytes=0)
   req=urllib.request.Request(url,headers={'User-Agent':'CraniofacialResearch-public-bulk-download/1.0'})
   with urllib.request.urlopen(req,timeout=45) as response:
    if response.headers.get('Content-Type','').lower().startswith('text/html'):raise ValueError('unexpected_HTML')
    tmp=target.with_name(target.name+'.part');n=0;h=hashlib.sha256();last=time.monotonic()
    with tmp.open('wb') as f:
     while chunk:=response.read(1024**2):
      n+=len(chunk)
      if n>3*1024**3:raise ValueError('public_file_exceeds_declared_size_budget')
      f.write(chunk);h.update(chunk)
      if time.monotonic()-last>5:
       status(name,state='downloading',attempt=attempt,bytes=n);last=time.monotonic()
    metadata=dict(url=url,resolved_url=response.geturl(),last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
   if n<100:raise ValueError('unexpected_short_download')
   opener=gzip.open(tmp,'rt',errors='replace') if name.endswith('.gz') else tmp.open(errors='replace')
   with opener as f:header=f.read(8192)
   if '<html' in header.lower() or 'captcha' in header.lower():raise ValueError('unexpected_HTML')
   if name.endswith('.xml') and '<' not in header:raise ValueError('expected_XML')
   if 'clinvar_variant_summary' in name and 'AlleleID' not in header:raise ValueError('unexpected_ClinVar_schema')
   if 'hgvs4variation' in name and 'VariationID' not in header:raise ValueError('unexpected_HGVS_schema')
   if name.startswith('clingen') and 'GENE' not in header.upper():raise ValueError('unexpected_ClinGen_schema')
   tmp.replace(target)
   r=dict(state='complete',**metadata,bytes=n,sha256=h.hexdigest(),retrieved_at_utc=now(),outbound_patient_data=False)
   save(receipt,r);status(name,state='complete',bytes=n,sha256=r['sha256']);return r
  except Exception as e:
   status(name,state='retrying' if attempt<3 else 'failed',attempt=attempt,error_type=type(e).__name__)
   if attempt<3:time.sleep(3)
 return dict(state='failed',url=url)


def main():
 for name in ['public','receipts']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={name:pool.submit(fetch,name,url) for name,url in SOURCES.items()}
  results={name:f.result() for name,f in jobs.items()}
 complete=sum(v['state']=='complete' for v in results.values())
 save(ROOT/'download_receipt.json',dict(state='complete' if complete==len(results) else 'partial',completed=complete,total=len(results),sources=results,outbound_patient_data=False,updated_at_utc=now()))
 with LOCK:save(ROOT/'download_status.json',dict(state='complete' if complete==len(results) else 'partial',resources=STATUS,completed=complete,total=len(results),outbound_patient_data=False,updated_at_utc=now()))
 print(json.dumps(dict(completed=complete,total=len(results),outbound_patient_data=False)))


if __name__=='__main__':main()

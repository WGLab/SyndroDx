"""Check every indexed full-genetics path under the known image mount."""
import csv,json,concurrent.futures,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'data_release_audit'
p=Path('/controlled-data/shared/wang_lab/quan/projects/MedicalDBs/CHOP_Images/final_results/arcus_genetics_photos_faith.csv')
with p.open() as f:
 h=f.readline();f.seek(0);rr=list(csv.DictReader(f,delimiter='\t' if h.count('\t')>h.count(',') else ','))
base=Path('/controlled-data/craniofacial_data');summary={};available=[]
for col in ['file_path','modified_file_path']:
 ps=sorted({r[col] for r in rr if r.get(col)})
 def check(s):
  path=Path(s) if Path(s).is_absolute() else base/s
  try:status='file' if path.is_file() else 'directory' if path.is_dir() else 'missing'
  except OSError:status='os_error'
  return s,str(path),status
 c=collections.Counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
  for original,resolved,status in pool.map(check,ps):
   c[status]+=1
   if status=='file':available.append({'source_column':col,'indexed_path':original,'resolved_path':resolved})
 summary[col]={'unique_indexed_paths':len(ps),'status_counts':dict(c)}
 print(col,summary[col],flush=True)
summary.update(rows=len(rr),source=str(p),resolution_root=str(base),scope='All unique paths checked under stated mount; no alternate-root recovery or filename guessing. Index and copied top10 images are not additive.')
(OUT/'full_genetics_path_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'full_genetics_available_paths.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source_column','indexed_path','resolved_path'],delimiter='\t');w.writeheader();w.writerows(available)

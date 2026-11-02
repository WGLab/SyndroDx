"""All expert-classification arms, preserving cohort and candidate scope."""
import sys,os,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent;Q=ROOT.parent/'cranioface_qwen_alignment_20261027';sys.path.insert(0,str(Q))
import numpy as np
from runtime import load,save,env,SLURM,PYTHON,notify,now,digest
def free_jobs():
 text=subprocess.run([str(SLURM/'squeue'),'-u','research_user','-h','-o','%i|%j|%T|%N'],env=env(),capture_output=True,text=True,check=True).stdout
 steps=subprocess.run([str(SLURM/'squeue'),'--steps','-u','research_user','-h','-o','%i'],env=env(),capture_output=True,text=True,check=True).stdout
 busy={s.split('.')[0] for s in steps.splitlines() if s.rsplit('.',1)[-1].isdigit()}
 return [dict(id=v[0],node=v[3]) for line in text.splitlines() if len(v:=line.split('|'))==4 and v[1].startswith('h100-hold20-') and v[2]=='RUNNING' and v[0] not in busy]
def launch(job,kind,arm,shard,world,attempt):
 name=f'expert-expanded-{kind}-{arm}-{shard}-{attempt}';e=env();e['PYTHONPATH']=str(Q)+':'+e.get('PYTHONPATH','')
 cmd=[str(SLURM/'srun'),'--jobid='+job['id'],'--job-name='+name,'--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=88G','--gpus-per-task=1','--overlap','--exact','--mpi=none',str(PYTHON),str(ROOT/f'worker_{kind}.py'),arm,str(shard),str(world)]
 with (ROOT/'logs'/(name+'.private.log')).open('ab') as f:return subprocess.Popen(cmd,env=e,stdout=f,stderr=f,start_new_session=True)
def score(world):
 data=load(ROOT/'private/patients.json');patients=data['patients'];mapping=data['class_to_mondo'];vocab=data['vocabulary'];results={}
 subsets=dict(all_expert=list(range(len(patients))),covered=[i for i,r in enumerate(patients) if r['covered']],paired_covered=[i for i,r in enumerate(patients) if r['covered'] and r['has_HPO']],missing_HPO=[i for i,r in enumerate(patients) if not r['has_HPO']])
 for kind in ['qwen','biomed']:
  for arm in ['fusion','hpo']:
   records=[r for i in range(world) for r in load(ROOT/'private'/f'{kind}_{arm}_{i}.json')['records']];by={r['index']:r for r in records}
   if set(by)!=set(range(len(patients))) or len(records)!=len(patients):raise RuntimeError('expert_inference_coverage_error')
   hits={}
   for i,r in by.items():
    scores=np.asarray(r['scores'])
    if len(scores)!=len(vocab):raise RuntimeError('candidate_class_dimension_mismatch')
    ranked=[]
    for j in np.argsort(-scores,kind='stable'):
     mid=mapping.get(vocab[int(j)]['label'])
     if mid and mid not in ranked:ranked.append(mid)
    hits[i]=[int(patients[i]['label'] in ranked[:k]) for k in [1,5,10]]
   results[kind+'_'+arm]={}
   for name,ids in subsets.items():
    diseases=sorted({patients[i]['label'] for i in ids})
    results[kind+'_'+arm][name]=dict(patients=len(ids),diseases=len(diseases),**{f'macro_top{k}':float(np.mean([np.mean([hits[i][j] for i in ids if patients[i]['label']==d]) for d in diseases])) if diseases else None for j,k in enumerate([1,5,10])})
 report=dict(state='complete',models=results,audit=load(ROOT/'audit.json'),
  inference='frozen development-selected checkpoints; Qwen pair-logit mean across every expert image; Biomed patient-mean image embedding',
  candidate_mapping='fixed training vocabulary, exact MONDO/OMIM equivalents only; deduplicate equivalent diagnoses',
  interpretation='all-expert denominator includes diseases absent from the trained classifier; paired_covered is the matched HPO benefit cohort',
  limitations=['Previously inspected expert cohort; exploratory transfer evaluation, not untouched external validation.','Clinician HPO timing/assertion and expert-source diagnoses await new clinical adjudication.','Prior near-duplicate quarantines remain excluded.'],updated_at_utc=now())
 save(ROOT/'results.json',report)
 body=['Expanded CHOP expert classification evaluation completed.','Aggregate cohort coverage and model results follow. No patient records, identifiers or images are included.',__import__('json').dumps(report,indent=2)]
 notify('expanded_expert_results_complete','\n'.join(body));save(ROOT/'notification.json',dict(state='notifier_invoked',updated_at_utc=now()))
def main():
 os.umask(0o077);lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not load(ROOT/'audit.json'):time.sleep(10)
 while not free_jobs():time.sleep(20)
 plan=load(ROOT/'protocol.json') or dict(world=len(free_jobs()),Qwen_checkpoints={a:load(Q/'classifier'/a/'best_checkpoint.json') for a in ['fusion','hpo']},
  Biomed_checkpoints={a:digest(Q.parent/'transface_biomedbert_alignment_20261027/fusion'/a/'best.pt') for a in ['fusion','hpo']},
  patient_manifest_sha256=digest(ROOT/'private/patients.json'),created_at_utc=now())
 save(ROOT/'protocol.json',plan);world=plan['world'];tasks=[(kind,arm,i) for kind in ['qwen','biomed'] for arm in ['fusion','hpo'] for i in range(world)];active={};tries={};finished={}
 save(ROOT.parent/'CURRENT_OPERATIONAL_HANDOFF.json',dict(active_experiment=str(ROOT),workflow_job=os.environ.get('SLURM_JOB_ID'),status_file=str(ROOT/'status.json'),results_file=str(ROOT/'results.json'),updated_at_utc=now()))
 while True:
  done={t for t in tasks if load(ROOT/'features'/('_'.join(map(str,t))+'.json')).get('state')=='complete'}
  if len(done)==len(tasks):break
  pool=free_jobs()
  for t in tasks:
   if t in done or (t in active and active[t].poll() is None):continue
   if t in active and active[t].returncode==0:
    finished.setdefault(t,time.monotonic())
    if time.monotonic()-finished[t]<180:continue
   if not pool:break
   tries[t]=tries.get(t,0)+1
   if tries[t]>2:raise RuntimeError('repeated_expert_worker_failure')
   active[t]=launch(pool.pop(0),*t,world,tries[t])
  save(ROOT/'status.json',dict(state='expert_inference_running',completed_shards=len(done),total_shards=len(tasks),updated_at_utc=now()));time.sleep(15)
 score(world);save(ROOT/'status.json',dict(state='complete',updated_at_utc=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(ROOT/'failure.json',dict(error_type=type(e).__name__,updated_at_utc=now()))
  try:notify('expert_inference_failure','Expanded expert inference requires inspection: '+type(e).__name__+'. Patient data remain on HPC; allocations preserved.')
  except Exception:pass
  raise

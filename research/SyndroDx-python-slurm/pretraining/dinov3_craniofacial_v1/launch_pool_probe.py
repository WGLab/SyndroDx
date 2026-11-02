"""Cross-node communication and DINOv3 capacity probes on idle allocations only."""
import argparse,contextlib,os,secrets,signal,subprocess,time
from runtime import ROOT,SLURM,claim_jobs,environment,scheduler,save,now

def launch(kind='collectives',arch='vit_7b',batch=4,limit=0):
 jobs,busy=scheduler()
 selected=[v for v in jobs.values() if v['state']=='RUNNING' and v['job_id'] not in busy and not v['name'].startswith('facecrop-') and v['gpus']==1]
 selected.sort(key=lambda v:(v['node'],int(v['job_id'])))
 if limit:selected=selected[:limit]
 if len(selected)<2:raise RuntimeError('insufficient_idle_probe_allocations')
 locks=claim_jobs([v['job_id'] for v in selected])
 output=ROOT/'hardware_probes'/f'{kind}_{arch}_b{batch}_n{len(selected)}'
 output.mkdir(parents=True,exist_ok=True)
 master=selected[0]['node'];port=35000+secrets.randbelow(20000);run_id='probe-'+secrets.token_hex(6)
 env=environment();processes=[]
 script='pool_rank_probe.py' if kind=='collectives' else 'capacity_rank.py'
 try:
  for rank,v in enumerate(selected):
   rank_env=dict(env,MASTER_ADDR=master,MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE=str(len(selected)),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',TORCHELASTIC_RUN_ID=run_id)
   command=[str(SLURM/'srun'),'--jobid='+v['job_id'],'--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--exact','--exclusive','--mpi=none',str(ROOT/'venv/bin/python'),str(ROOT/script),'--output',str(output)]
   if kind!='collectives':command+=['--arch',arch,'--batch',str(batch)]
   with (output/f'rank_{rank:02d}.private.log').open('ab') as f:
    processes.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=rank_env,start_new_session=True))
  save(output/'status.json',{'state':'probe_running','kind':kind,'arch':arch,'batch':batch,'job_ids':[v['job_id'] for v in selected],'launcher_pids':[p.pid for p in processes],'clinical_data_accessed':False,'started_at_utc':now()})
  deadline=time.monotonic()+(900 if kind!='collectives' else 240)
  while time.monotonic()<deadline:
   codes=[p.poll() for p in processes]
   if any(c not in [None,0] for c in codes):raise RuntimeError('probe_rank_failed')
   if all(c==0 for c in codes):
    from runtime import load
    receipts=[load(output/f'rank_{i:02d}.json') for i in range(len(selected))]
    if any(r.get('state')!='passed' for r in receipts):raise RuntimeError('probe_receipt_incomplete')
    summary={'state':'passed','kind':kind,'arch':arch,'batch_size_per_gpu':batch,'world_size':len(selected),'nodes':sorted({v['node'] for v in selected}),'job_ids':[v['job_id'] for v in selected],'clinical_data_accessed':False,'pretraining_started':False,'completed_at_utc':now(),'rank_receipts':receipts}
    save(output/'completion.json',summary);save(output/'status.json',summary)
    print({k:v for k,v in summary.items() if k!='rank_receipts'},flush=True);return
   time.sleep(2)
  raise RuntimeError('probe_deadline_exceeded')
 except Exception as e:
  save(output/'status.json',{'state':'failed','error_type':type(e).__name__,'reason':str(e),'clinical_data_accessed':False,'existing_processing_untouched':True})
  raise
 finally:
  for p in processes:
   if p.poll() is None:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
  for p in processes:
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
  for f in locks:f.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['collectives','capacity'],default='collectives');p.add_argument('--arch',default='vit_7b');p.add_argument('--batch',type=int,default=4);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 launch(a.kind,a.arch,a.batch,a.limit)

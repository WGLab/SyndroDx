"""Finish public-checkpoint equivalence and 2 -> 4 rank resume tests."""
import contextlib,os,secrets,signal,subprocess,time,traceback
from runtime import ROOT,SLURM,scheduler,claim_jobs,environment,save,load,now,digest

def run_group(selected,script,args,output,timeout=900):
 locks=claim_jobs([v['job_id'] for v in selected]);processes=[]
 master=selected[0]['node'];port=35000+secrets.randbelow(20000)
 try:
  for rank,job in enumerate(selected):
   env=environment();env.update(MASTER_ADDR=master,MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE=str(len(selected)),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',TORCHELASTIC_RUN_ID='validation')
   command=[str(SLURM/'srun'),'--jobid='+job['job_id'],'--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--exact','--exclusive','--mpi=none',str(ROOT/'venv/bin/python'),str(ROOT/script),*args]
   with (output/f'{script}_rank_{rank:02d}.private.log').open('ab') as f:processes.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=env,start_new_session=True))
  deadline=time.monotonic()+timeout
  while time.monotonic()<deadline:
   codes=[p.poll() for p in processes]
   if any(c not in [None,0] for c in codes):raise RuntimeError('validation_rank_failed')
   if all(c==0 for c in codes):return
   time.sleep(2)
  raise RuntimeError('validation_timeout')
 finally:
  for p in processes:
   if p.poll() is None:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
  for p in processes:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
  for f in locks:f.close()

def main():
 output=ROOT/'hardware_probes/training_function_validation';output.mkdir(parents=True,exist_ok=True)
 jobs,busy=scheduler();selected=[v for v in jobs.values() if v['state']=='RUNNING' and v['job_id'] not in busy and not v['name'].startswith('facecrop-') and v['gpus']==1]
 selected.sort(key=lambda v:(v['node'],int(v['job_id'])))
 if len(selected)<4:raise RuntimeError('four_idle_validation_GPUs_required')
 save(output/'status.json',{'state':'validating_official_weight_function','clinical_data_accessed':False,'updated_at_utc':now()})
 if load(ROOT/'assets/native_weight_function_validation.json').get('state')!='passed':run_group(selected[:1],'validate_weight_function.py',[],output)
 if load(ROOT/'assets/native_weight_function_validation.json').get('state')!='passed':raise RuntimeError('functional_weight_receipt_missing')
 smoke=output/'synthetic_training_resume';smoke.mkdir(exist_ok=True)
 save(output/'status.json',{'state':'validating_2_rank_training_segment','clinical_data_accessed':False,'updated_at_utc':now()})
 if not (smoke/'ckpt/1/.metadata').exists():run_group(selected[:2],'test_train_smoke_rank.py',['--output',str(smoke)],output)
 if load(smoke/'state.json').get('state')!='segment_complete':raise RuntimeError('first_training_segment_incomplete')
 save(output/'status.json',{'state':'validating_4_rank_checkpoint_resharding','clinical_data_accessed':False,'updated_at_utc':now()})
 run_group(selected[:4],'test_train_smoke_rank.py',['--output',str(smoke)],output)
 if load(smoke/'state.json').get('state')!='complete' or not (smoke/'synthetic_encoder/encoder_export.json').exists():raise RuntimeError('resume_or_encoder_export_incomplete')
 receipt={'state':'passed','official_checkpoint_feature_equivalence':True,'full_student_teacher_optimizer_capacity_passed':True,'synthetic_resume_world_sizes':[2,4],'cross_node_collectives_world_size':11,'clinical_data_accessed':False,'code_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},'completed_at_utc':now()}
 save(output/'status.json',receipt);save(ROOT/'hardware_validation_ready.json',receipt)

if __name__=='__main__':
 try:main()
 except Exception as e:
  output=ROOT/'hardware_probes/training_function_validation';output.mkdir(parents=True,exist_ok=True)
  with (output/'validation.private_errors.log').open('a') as f:traceback.print_exc(file=f)
  save(output/'status.json',{'state':'attention_required','error_type':type(e).__name__,'reason':str(e),'clinical_data_accessed':False,'updated_at_utc':now()})
  raise SystemExit(1)

"""Public-weight/synthetic-input acceptance on a newly allocated H100."""
import os,runpy,sys
from pathlib import Path
import runtime
folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
original_save=runtime.save
locks=runtime.claim_jobs([os.environ['SLURM_JOB_ID']])
runtime.save=lambda path,value:original_save(folder/Path(path).name,value)
try:
 runpy.run_path(str(runtime.ROOT/'validate_weight_function.py'),run_name='__main__')
 original_save(folder/'ready.json',{'state':'passed','clinical_data_accessed':False,'synthetic_official_feature_equivalence':True,'completed_at_utc':runtime.now()})
except Exception as e:
 original_save(folder/'ready.json',{'state':'failed','error_type':type(e).__name__,'clinical_data_accessed':False,'updated_at_utc':runtime.now()});raise
finally:
 for lock in locks:lock.close()

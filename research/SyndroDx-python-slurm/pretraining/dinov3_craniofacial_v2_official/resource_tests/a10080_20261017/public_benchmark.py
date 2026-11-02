"""Public-weight, synthetic-image inference benchmark. No clinical inputs."""
import gc,json,os,statistics,sys,time
from pathlib import Path
BASE=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v2_official')
sys.path.insert(0,str(BASE))
from common import save,now,V1
OUT=BASE/'resource_tests/a10080_20261017'
slot=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
receipt=OUT/f'slot_{slot:02d}.json'

def main():
 import torch
 from transformers import AutoModel
 os.umask(0o077);torch.set_num_threads(4)
 prop=torch.cuda.get_device_properties(0)
 info=dict(slot=slot,job_id=os.environ['SLURM_JOB_ID'],GPU_name=prop.name,GPU_memory_GiB=prop.total_memory/2**30,clinical_data_accessed=False,weights='cached original public DINOv3 ViT7B',purpose='choose efficient inference batches for subsequent local evaluation',started_at_utc=now())
 if prop.total_memory<78*2**30:
  save(receipt,dict(info,state='released_not_80GB'))
  return 3
 save(receipt,dict(info,state='loading_public_encoder'))
 model=AutoModel.from_pretrained(str(V1/'assets/dinov3-vit7b16'),local_files_only=True,dtype=torch.bfloat16).to('cuda').eval()
 results=[]
 with torch.inference_mode():
  for batch in [16,32,64,128,256,512]:
   image=None;result=None
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   try:
    image=torch.randn(batch,3,256,256,device='cuda',dtype=torch.bfloat16)
    for _ in range(2):result=model(pixel_values=image);del result;result=None
    torch.cuda.synchronize();times=[]
    for _ in range(5):
     first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
     first.record();result=model(pixel_values=image);last.record();last.synchronize()
     times.append(first.elapsed_time(last));del result;result=None
    row=dict(state='passed',batch_size=batch,mean_milliseconds=statistics.mean(times),median_milliseconds=statistics.median(times),images_per_second=batch*1000/statistics.mean(times),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
   except torch.cuda.OutOfMemoryError:
    row=dict(state='out_of_memory',batch_size=batch,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
   finally:
    del image,result;gc.collect();torch.cuda.empty_cache()
   results.append(row);save(receipt,dict(info,state='benchmarking',results=results,updated_at_utc=now()))
   if row['state']!='passed':break
 viable=[x for x in results if x['state']=='passed' and x['peak_reserved_GiB']<74]
 if not viable:raise RuntimeError('no_safe_inference_batch')
 best=max(viable,key=lambda x:x['images_per_second'])
 save(receipt,dict(info,state='benchmark_complete_reservation_retained',results=results,selected_inference_batch_size=best['batch_size'],selected_images_per_second=best['images_per_second'],inference_only_not_a_training_batch_recommendation=True,completed_at_utc=now()))
 return 0

if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  save(receipt,dict(state='failed',slot=slot,error_type=type(exc).__name__,clinical_data_accessed=False,updated_at_utc=now()))
  raise SystemExit(1)

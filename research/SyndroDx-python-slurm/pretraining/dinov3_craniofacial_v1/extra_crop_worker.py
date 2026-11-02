"""A released primary GPU assists the remaining private crop queue."""
import argparse,fcntl,time,traceback
from runtime import load,save,now
from crop_delegation import ACCEL,PIPELINE,CropWork,crop_module
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args()
lock=(ACCEL/f'worker_{a.slot:02d}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 if load(ACCEL/f'extra_{a.slot:02d}.transfer.json').get('state')!='sealed':raise RuntimeError('sealed_transfer_required')
 save(ACCEL/f'worker_{a.slot:02d}.json',{'state':'loading_extra_crop_model','updated_at_utc':now()})
 mod=crop_module();models=mod.LocalModels(PIPELINE)
 from PIL import Image
 blank=Image.new('RGB',(128,128),'white');check=models.ask([blank,blank],mod.PROMPT);blank.close()
 if not mod.negative_control_passed(check):raise RuntimeError('extra_crop_negative_control_failed')
 work=CropWork(a.slot,mod)
 while work.pending():work.step(models,limit=8)
 save(ACCEL/f'worker_{a.slot:02d}.json',{'state':'primary_crop_partition_complete','completed':len(work.done),'total':len(work.tasks),'updated_at_utc':now()})
except Exception as e:
 with (ACCEL/f'worker_{a.slot:02d}.private_errors.log').open('a') as f:traceback.print_exc(file=f)
 save(ACCEL/f'worker_{a.slot:02d}.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()});raise SystemExit(1)

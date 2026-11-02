"""Reuse one local Qwen model for delegated primary cropping and existing QC."""
import argparse,traceback
from runtime import ROOT,RUN,save,now
from crop_delegation import CropWork,crop_module,PIPELINE,ACCEL
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args()
try:
 save(ROOT/'independent_qc'/f'slot_{a.slot:02d}.json',{'state':'loading_shared_crop_QC_models','slot':a.slot,'updated_at':__import__('time').time()})
 mod=crop_module();models=mod.LocalModels(PIPELINE)
 from PIL import Image
 blank=Image.new('RGB',(128,128),'white');check=models.ask([blank,blank],mod.PROMPT);blank.close()
 if not mod.negative_control_passed(check):raise RuntimeError('assisted_crop_negative_control_failed')
 work=CropWork(a.slot,mod)
 import independent_qc
 independent_qc.main(a.slot,shared_models=models,crop_work=work)
except Exception as e:
 import time
 with (ACCEL/f'worker_{a.slot:02d}.private_errors.log').open('a') as f:traceback.print_exc(file=f)
 save(ROOT/'independent_qc'/f'slot_{a.slot:02d}.json',{'state':'attention_required','slot':a.slot,'error_type':type(e).__name__,'updated_at':time.time()});raise SystemExit(1)

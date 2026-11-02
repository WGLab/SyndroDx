"""A released primary GPU reviews assigned tail work entirely inside HPC."""
import argparse,traceback,time
from runtime import ROOT,save
import independent_qc
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args()
independent_qc.STAGE=ROOT/'qc_tail'
try:independent_qc.main(a.slot)
except Exception as e:
 with (ROOT/'qc_tail'/f'slot_{a.slot:02d}.private_errors.log').open('a') as f:traceback.print_exc(file=f)
 save(ROOT/'qc_tail'/f'slot_{a.slot:02d}.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at':time.time()});raise SystemExit(1)

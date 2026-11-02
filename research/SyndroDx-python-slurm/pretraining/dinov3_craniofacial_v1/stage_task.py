"""Finite CPU stage with private diagnostics and a content-free completion receipt."""
import argparse,traceback
from runtime import ROOT,save,now
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['identity','release'],required=True);a=p.parse_args()
folder=ROOT/'stage_tasks';folder.mkdir(exist_ok=True)
try:
 from release_data import identity_plan,final_release
 save(folder/(a.kind+'.json'),{'state':'running','updated_at_utc':now()})
 if a.kind=='identity':identity_plan();state='complete'
 else:state=final_release()
 save(folder/(a.kind+'.json'),{'state':'complete' if state=='ssl_data_ready' else state,'updated_at_utc':now()})
except Exception as e:
 with (folder/(a.kind+'.private_errors.log')).open('a') as f:traceback.print_exc(file=f)
 save(folder/(a.kind+'.json'),{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()})
 raise SystemExit(1)

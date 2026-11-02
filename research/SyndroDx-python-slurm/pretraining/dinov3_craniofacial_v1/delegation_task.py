import argparse,traceback
from crop_delegation import ACCEL,claim_owner,merge_owner
from runtime import save,now
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['claim','merge'],required=True);p.add_argument('--owner',required=True,type=int);p.add_argument('--expected-host',required=True);a=p.parse_args()
try:
 if a.kind=='claim':claim_owner(a.owner,a.expected_host)
 else:merge_owner(a.owner,a.expected_host)
except Exception as e:
 ACCEL.mkdir(exist_ok=True)
 with (ACCEL/f'owner_{a.owner:02d}.{a.kind}.private_errors.log').open('a') as f:traceback.print_exc(file=f)
 save(ACCEL/f'owner_{a.owner:02d}.{a.kind}_failure.json',{'state':'attention_required','error_type':type(e).__name__,'updated_at_utc':now()});raise SystemExit(1)

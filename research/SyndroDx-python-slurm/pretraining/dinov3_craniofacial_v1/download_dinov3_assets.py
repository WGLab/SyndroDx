"""Download approved public model assets; never open clinical data."""
import datetime,json,os,traceback
from pathlib import Path
os.umask(0o077)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
ROOT=Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
OUT=ROOT/'assets/dinov3-vit7b16'
OUT.mkdir(parents=True,exist_ok=True)
def status(value):
 value['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=ROOT/'assets/model_download_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
try:
 from huggingface_hub import snapshot_download,HfApi
 repo='facebook/dinov3-vit7b16-pretrain-lvd1689m'
 revision=HfApi().model_info(repo).sha
 status({'state':'downloading_official_weights','repository':repo,'revision':revision})
 snapshot_download(repo_id=repo,revision=revision,local_dir=OUT,allow_patterns=['*.safetensors','*.json','LICENSE.md','README.md'],max_workers=4)
 index=json.loads((OUT/'model.safetensors.index.json').read_text())
 from safetensors import safe_open
 keys=set()
 for shard in sorted(set(index['weight_map'].values())):
  with safe_open(OUT/shard,framework='pt',device='cpu') as f:keys.update(f.keys())
 if keys!=set(index['weight_map']):raise RuntimeError('model_shard_key_coverage_mismatch')
 status({'state':'official_weights_downloaded','repository':repo,'revision':revision,'path':str(OUT),'sharded_keys_complete':True,'clinical_data_accessed':False})
except Exception as e:
 with (ROOT/'assets/model_download.private.log').open('a') as f:traceback.print_exc(file=f)
 status({'state':'attention_required','error_type':type(e).__name__,'clinical_data_accessed':False})
 raise SystemExit(1)

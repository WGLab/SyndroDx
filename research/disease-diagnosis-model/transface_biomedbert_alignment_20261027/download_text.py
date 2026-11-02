"""Download public weights only. No clinical inputs or authentication tokens sent."""
import os
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
from huggingface_hub import snapshot_download
from common import WORK,MODEL_ID,save,digest,now

root=WORK/'assets/biomedbert'
snapshot_download(MODEL_ID,local_dir=root,token=False,
    allow_patterns=['config.json','pytorch_model.bin','model.safetensors','tokenizer_config.json',
        'tokenizer.json','vocab.txt','special_tokens_map.json'],max_workers=3)
files={p.name:digest(p) for p in root.iterdir() if p.is_file()}
if 'config.json' not in files or not {'model.safetensors','pytorch_model.bin'}&files.keys():
    raise RuntimeError('incomplete_public_model_download')
save(WORK/'assets/biomedbert_receipt.json',dict(state='complete',model=MODEL_ID,files=files,updated_at_utc=now()))
print('Public biomedical text weights downloaded and checksummed.')

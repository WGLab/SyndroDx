"""Warm one offline GPU replica, then run the locally installed handler."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

os.umask(0o077)
for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                  'DO_NOT_TRACK':'1','WANDB_DISABLED':'true','TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ[key]=value
ROOT=Path(__file__).resolve().parent


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def serve(slot):
    folder=ROOT/'replicas';folder.mkdir(exist_ok=True)
    lock=(folder/f'slot_{slot:02d}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=folder/f'slot_{slot:02d}.json'
    save(status,{'state':'loading_model','slot':slot,'updated_at':time.time()})
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Assigned GPU unavailable')
    if torch.cuda.get_device_properties(0).total_memory<70*1024**3:
        raise RuntimeError('GPU is not an 80GB-class device')
    settings=json.loads((ROOT/'config.json').read_text())
    model=Qwen3VLForConditionalGeneration.from_pretrained(settings['cached_model'],dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(settings['cached_model'],local_files_only=True)
    processor.tokenizer.padding_side='left'
    image=Image.new('RGB',(96,96),'white')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Reply with OK only.'}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],return_tensors='pt').to(model.device)
    with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=16)
    if output.shape[-1]<=inputs.input_ids.shape[-1]:raise RuntimeError('GPU generation returned no tokens')
    del inputs,output;image.close()
    save(status,{'state':'ready','slot':slot,'synthetic_gpu_generation_passed':True,'updated_at':time.time()})
    print('OFFLINE_QWEN_GPU_REPLICA_READY',flush=True)
    while not (ROOT/'annotation_start.json').is_file():time.sleep(5)
    spec=importlib.util.spec_from_file_location('local_qwen_annotation_handler',ROOT/'qwen_annotation_handler.py')
    handler=importlib.util.module_from_spec(spec);spec.loader.exec_module(handler)
    save(status,{'state':'annotating','slot':slot,'updated_at':time.time()})
    handler.run(model,processor,slot)
    save(status,{'state':'worker_finished','slot':slot,'updated_at':time.time()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--slot',type=int,required=True);args=parser.parse_args()
    try:serve(args.slot)
    except Exception:
        folder=ROOT/'replicas';folder.mkdir(exist_ok=True)
        with (folder/f'slot_{args.slot:02d}.error.log').open('a') as handle:traceback.print_exc(file=handle)
        save(folder/f'slot_{args.slot:02d}.json',{'state':'failed','slot':args.slot,'updated_at':time.time()})
        print('QWEN_REPLICA_FAILED_DETAILS_ON_HPC',flush=True)
        raise SystemExit(1)

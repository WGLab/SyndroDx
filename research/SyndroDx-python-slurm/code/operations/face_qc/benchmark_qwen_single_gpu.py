"""Measure single-H100 BF16 throughput without changing active annotations."""
import csv
import json
import time
from pathlib import Path
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import vlm_screen as base

out = base.ROOT / 'face_qc/qwen_single_gpu_benchmark'
out.mkdir(exist_ok=True)
assert torch.cuda.device_count() == 1
model = Qwen3VLForConditionalGeneration.from_pretrained(base.MODEL, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa').eval()
processor = AutoProcessor.from_pretrained(base.MODEL, local_files_only=True, min_pixels=4096, max_pixels=1048576)
processor.tokenizer.padding_side = 'left'
with base.INPUT.open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))[:32]
images = []
for row in rows:
    with Image.open(row['original_image_path']) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    images.append(im)
messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': base.PROMPT.read_text()}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
with (out / 'measurements.jsonl').open('w', buffering=1) as f:
    for size in [4, 8, 16, 32]:
        for repeat in range(2):
            inputs = generated = None
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            record = {'batch_size': size, 'repeat': repeat, 'model': str(base.MODEL), 'dtype': 'bfloat16'}
            try:
                inputs = processor(text=[text] * size, images=images[:size], padding=True, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=240, do_sample=False)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                valid = 0
                for r in raw:
                    try:
                        base.parse_response(r)
                        valid += 1
                    except (TypeError, ValueError):
                        pass
                record.update(status='ok', seconds=elapsed, images_per_second=size / elapsed,
                              valid_json=valid, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3)
                (out / f'predictions_b{size}_r{repeat}.json').write_text(json.dumps([
                    {'review_id': row['review_id'], 'raw_response': response} for row, response in zip(rows[:size], raw)], indent=2) + '\n')
            except torch.cuda.OutOfMemoryError:
                record.update(status='oom')
            finally:
                del inputs, generated
                torch.cuda.empty_cache()
            f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            if record['status'] == 'oom':
                break
        if record['status'] == 'oom':
            break

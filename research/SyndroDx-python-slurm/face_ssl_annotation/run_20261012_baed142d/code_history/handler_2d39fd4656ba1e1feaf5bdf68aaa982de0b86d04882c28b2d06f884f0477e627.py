"""Direct offline Qwen screening and within-pseudonym visual duplicate review."""
import argparse
from collections import defaultdict, deque
import concurrent.futures
import csv
import fcntl
import hashlib
import hmac
import itertools
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

os.umask(0o077)
for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                  'DO_NOT_TRACK':'1','WANDB_DISABLED':'true','TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ[key]=value
RUN=Path(__file__).resolve().parent
from annotation_pipeline import references, canonical_path, parse_label, eligible


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def run(model,processor,slot):
    settings=json.loads((RUN/'config.json').read_text())
    workers=RUN/'direct_workers';workers.mkdir(exist_ok=True)
    claims=workers/'claims';claims.mkdir(exist_ok=True)
    import torch
    from PIL import Image,ImageOps
    def state(name,**details):
        save(workers/f'slot_{slot:02d}.status.json',{'state':name,'slot':slot,'updated_at':time.time(),**details})
    def generate(batches,token_limit=384):
        texts=[];images=[]
        for photos,prompt in batches:
            content=[]
            for index,photo in enumerate(photos):
                content.extend([{'type':'text','text':f'Image {index+1}:'},{'type':'image','image':photo}])
            content.append({'type':'text','text':prompt})
            texts.append(processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
            images.extend(photos)
        inputs=processor(text=texts,images=images,padding=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():outputs=model.generate(**inputs,do_sample=False,max_new_tokens=token_limit)
        text=processor.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        del inputs,outputs
        return text
    def process_shard(shard):
        state('processing',shard=shard)
        status_path=workers/f'shard_{shard:02d}.status.json'
        def shard_state(name,**details):
            save(status_path,{'state':name,'shard':shard,'slot':slot,'updated_at':time.time(),**details})
        groups={}
        with (RUN/'direct_inputs'/f'shard_{shard:02d}.jsonl').open() as handle:
            for line in handle:
                value=json.loads(line)
                groups[value['patient_key']]={photo['photo_id']:(photo['path'],value['patient_known']) for photo in value['photos']}
        database=sqlite3.connect(workers/f'shard_{shard:02d}.sqlite')
        database.executescript('''
           CREATE TABLE IF NOT EXISTS labels(photo_id TEXT PRIMARY KEY,patient_key TEXT,path TEXT,
             patient_known INTEGER,status TEXT,label TEXT,width INTEGER,height INTEGER,duplicate_of TEXT,raw TEXT,updated_at REAL);
           CREATE TABLE IF NOT EXISTS comparisons(window_id TEXT PRIMARY KEY,patient_key TEXT,result TEXT,state TEXT,updated_at REAL);
        ''')
        done={item[0] for item in database.execute('SELECT photo_id FROM labels')}
        tasks=[(photo,patient,path,known) for patient,photos in groups.items() for photo,(path,known) in photos.items() if photo not in done]
        def decode(task):
            try:
                before=Path(task[2]).stat()
                with Image.open(task[2]) as original:photo=ImageOps.exif_transpose(original).convert('RGB')
                dimensions=photo.size
                photo.thumbnail((settings['max_image_edge'],settings['max_image_edge']),Image.Resampling.LANCZOS)
                after=Path(task[2]).stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    photo.close();return task,None,None,'source_changed'
                return task,photo,dimensions,'ok'
            except Exception as error:return task,None,None,'image_error:'+type(error).__name__
        def store(item,status,label=None,raw=None):
            task,photo,dimensions,_=item
            database.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (task[0],task[1],task[2],int(task[3]),status,json.dumps(label) if label else None,
                 dimensions[0] if dimensions else None,dimensions[1] if dimensions else None,None,raw,time.time()))
            database.commit()
        batch_size=settings['batch_size'];prompt=(RUN/'prompt.txt').read_text()
        def infer(items):
            nonlocal batch_size
            try:outputs=generate([([item[1]],prompt) for item in items])
            except torch.OutOfMemoryError:
                outputs=None
            if outputs is None:
                torch.cuda.empty_cache()
                if len(items)==1:raise RuntimeError('Single-image GPU memory failure')
                batch_size=max(1,batch_size//2);middle=len(items)//2
                infer(items[:middle]);infer(items[middle:]);return
            if len(outputs)!=len(items):raise RuntimeError('Response alignment mismatch')
            for item,raw in zip(items,outputs):
                try:store(item,'ok',parse_label(raw),raw)
                except (ValueError,TypeError,json.JSONDecodeError):store(item,'invalid_response',raw=raw)
                item[1].close()
            shard_state('annotating',processed=database.execute('SELECT count(*) FROM labels').fetchone()[0])
        shard_state('annotating')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            iterator=iter(tasks);futures=deque();batch=[]
            for _ in range(max(4,batch_size*2)):
                try:futures.append(pool.submit(decode,next(iterator)))
                except StopIteration:break
            while futures:
                item=futures.popleft().result()
                try:futures.append(pool.submit(decode,next(iterator)))
                except StopIteration:pass
                if item[3]!='ok':store(item,item[3]);continue
                batch.append(item)
                if len(batch)>=batch_size:infer(batch);batch=[]
            if batch:infer(batch)
        expected=sum(len(group) for group in groups.values())
        if database.execute('SELECT count(*) FROM labels').fetchone()[0]!=expected:raise RuntimeError('Incomplete image coverage')
        shard_state('qwen_patient_duplicate_review')
        compare_prompt='''These images were grouped locally for the same participant and the same broad view. Do not identify the person or infer a diagnosis. Compare the photographs for near duplication only. Different ages/time points, left versus right profiles, different head angles, different expressions, clinical changes, or useful additional visible structure must be retained separately. Mark redundant only when photographs show effectively the same capture/composition and preserve the same facial information, including resized/re-encoded copies or virtually identical bursts. Choose the clearer complete photograph as keeper. When uncertain, return no redundant group. Never group images merely because the person looks the same.
    Return only JSON: {"groups":[{"members":[1,2],"keep":1,"certainty":"high"}]}. Image numbers are one-based positions in this request. Every group has at least two distinct valid members, keep is one member, groups cannot overlap, and certainty must be high. Return {"groups":[]} when no confidently redundant group is present.'''
        accepted=defaultdict(list)
        for photo,patient,path,known,label in database.execute("SELECT photo_id,patient_key,path,patient_known,label FROM labels WHERE status='ok'"):
            value=json.loads(label)
            if known and eligible(value):accepted[(patient,value['view'])].append((photo,path))
        completed_windows={item[0] for item in database.execute('SELECT window_id FROM comparisons')}
        for (patient,view),items in accepted.items():
            chunks=[items[offset:offset+8] for offset in range(0,len(items),8)]
            # Within-block and all pairs of blocks cover every pair for this view.
            windows=itertools.chain(chunks,(first+second for first,second in itertools.combinations(chunks,2)))
            for window in windows:
                if len(window)<2:continue
                identifier=token('compare:'+','.join(item[0] for item in window))
                if identifier in completed_windows:continue
                photos=[];loaded=[]
                for item in window:
                    decoded=decode((item[0],patient,item[1],True))
                    if decoded[3]=='ok':photos.append(decoded[1]);loaded.append(item)
                raw=None;review_state='review_failed_keep_all'
                try:
                    if len(photos)>=2:
                        raw=generate([(photos,compare_prompt)],512)[0]
                        value=json.loads(raw.strip())
                        if not isinstance(value,dict) or set(value)!={'groups'} or not isinstance(value['groups'],list):raise ValueError('Invalid duplicate review')
                        used=set();decisions=[]
                        for group in value['groups']:
                            if not isinstance(group,dict) or set(group)!={'members','keep','certainty'}:raise ValueError('Invalid duplicate group')
                            members=group['members'];keeper=group['keep']
                            if not isinstance(members,list) or len(members)<2 or len(set(members))!=len(members):raise ValueError('Invalid group members')
                            if any(type(index) is not int or not 1<=index<=len(loaded) for index in members):raise ValueError('Invalid image index')
                            if keeper not in members or group['certainty']!='high' or used.intersection(members):raise ValueError('Invalid keeper or uncertainty')
                            used.update(members);decisions.append((members,keeper))
                        for members,keeper in decisions:
                            keep_id=loaded[keeper-1][0]
                            if database.execute('SELECT duplicate_of FROM labels WHERE photo_id=?',(keep_id,)).fetchone()[0] is not None:continue
                            for position in members:
                                if position!=keeper:
                                    candidate=loaded[position-1][0]
                                    # Avoid chains that drop a photograph serving as
                                    # another photograph's directly verified keeper.
                                    if database.execute('SELECT 1 FROM labels WHERE duplicate_of=? LIMIT 1',(candidate,)).fetchone():continue
                                    database.execute('UPDATE labels SET duplicate_of=? WHERE photo_id=? AND duplicate_of IS NULL',(keep_id,candidate))
                        review_state='qwen_reviewed'
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    review_state='memory_limit_keep_all'
                except (ValueError,TypeError,KeyError,json.JSONDecodeError):review_state='invalid_review_keep_all'
                finally:
                    for photo in photos:photo.close()
                database.execute('INSERT INTO comparisons VALUES(?,?,?,?,?)',(identifier,patient,raw,review_state,time.time()));database.commit()
                shard_state('qwen_patient_duplicate_review',completed_windows=database.execute('SELECT count(*) FROM comparisons').fetchone()[0])
        assert database.execute('PRAGMA quick_check').fetchall()==[('ok',)]
        database.close();shard_state('completed',expected_images=expected)
        print('DIRECT_QWEN_SHARD_COMPLETED',flush=True)
    count=settings['logical_shards']
    order=[(slot+offset)%count for offset in range(count)]
    for shard in order:
        status_path=workers/f'shard_{shard:02d}.status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('state')=='completed':continue
        claim=claims/f'shard_{shard:02d}'
        try:claim.mkdir()
        except FileExistsError:continue
        save(claim/'owner.json',{'slot':slot,'claimed_at':time.time()})
        try:process_shard(shard)
        except Exception:
            with (workers/f'shard_{shard:02d}.error.log').open('a') as handle:traceback.print_exc(file=handle)
            save(status_path,{'state':'failed','shard':shard,'slot':slot,'updated_at':time.time()})
            state('failed',shard=shard)
            raise
    state('finished_available_shards')

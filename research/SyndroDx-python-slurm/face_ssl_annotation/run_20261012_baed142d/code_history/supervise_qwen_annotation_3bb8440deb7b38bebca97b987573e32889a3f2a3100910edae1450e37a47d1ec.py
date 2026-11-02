"""Start granted replicas, share a local queue, and notify without data exports."""
from collections import defaultdict
import csv
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import time
import traceback

os.umask(0o077)
RUN=Path(__file__).resolve().parent
from annotation_pipeline import references,canonical_path,eligible


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def notify(event,message):
    settings=json.loads((RUN/'config.json').read_text());recipient=settings.get('notification_email')
    if not recipient:return
    folder=RUN/'notifications';folder.mkdir(exist_ok=True)
    lock=(folder/(event+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    receipt=folder/(event+'.json')
    if receipt.exists():return
    save(receipt,{'state':'sending','updated_at':time.time()})
    text='To: '+recipient+'\nSubject: HPC Qwen '+event+'\nContent-Type: text/plain; charset=utf-8\n\n'+message+'\nLocal status directory: '+str(RUN)+'\n'
    try:
        result=subprocess.run(['/usr/sbin/sendmail','-t','-oi'],input=text,text=True,capture_output=True,timeout=30)
        save(receipt,{'state':'accepted_by_local_mail_system' if result.returncode==0 else 'failed_review_required',
                      'returncode':result.returncode,'updated_at':time.time()})
    except Exception as error:save(receipt,{'state':'uncertain_no_automatic_resend','error_type':type(error).__name__,'updated_at':time.time()})


def input_references(settings):
    marker=RUN/'direct_input_manifest.json'
    if marker.exists():return
    if not settings.get('patient_column'):raise RuntimeError('Pseudo-patient grouping field unresolved')
    salt_file=RUN/'direct_identity_salt.bin'
    if not salt_file.exists():salt_file.write_bytes(secrets.token_bytes(32))
    salt=salt_file.read_bytes()
    def token(value):return hmac.new(salt,value.encode(),hashlib.sha256).hexdigest()
    # This is metadata parsing only: no image hash, image decoding, CPU face
    # detection, pose estimator, or image quality classifier is used.
    inputs=RUN/'direct_inputs';inputs.mkdir(exist_ok=True)
    groups=defaultdict(dict);known={};rows=0;aliases=0;missing=[]
    metadata=Path(settings['metadata']);before=metadata.stat()
    csv.field_size_limit(16*1024*1024)
    with metadata.open(newline='') as handle,(RUN/'input_aliases.tsv').open('w',newline='') as output:
        reader=csv.DictReader(handle,delimiter='\t')
        writer=csv.DictWriter(output,fieldnames=['alias_id','row_key','patient_key','patient_known','photo_id','original_path','source_subset','status'],delimiter='\t');writer.writeheader()
        for ordinal,row in enumerate(reader):
            rows+=1
            pseudo=(row.get(settings['patient_column']) or '').strip()
            valid=bool(pseudo) and pseudo.lower() not in {'none','null','nan','<na>'}
            patient=token('patient:'+pseudo) if valid else token('unknown-row:'+str(ordinal))
            known[patient]=valid
            paths=references(row.get(settings['image_column']))
            if not paths:paths=[None]
            for index,raw in enumerate(paths):
                aliases+=1
                path=str(canonical_path(raw,settings)) if raw else ''
                photo=token(patient+':path:'+path) if path else ''
                writer.writerow({'alias_id':token(f'alias:{ordinal}:{index}'),'row_key':token('row:'+str(ordinal)),
                    'patient_key':patient,'patient_known':int(valid),'photo_id':photo,'original_path':path,
                    'source_subset':row.get('subset',''),'status':'referenced' if path else 'missing_reference'})
                if path:groups[patient][photo]=path
                else:missing.append({'row_key':token('row:'+str(ordinal)),'reason':'missing_reference'})
    after=metadata.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Metadata changed while reading')
    handles=[(inputs/f'shard_{number:02d}.jsonl').open('w') for number in range(settings['logical_shards'])]
    try:
        for patient,photos in groups.items():
            shard=int(patient[:16],16)%settings['logical_shards']
            handles[shard].write(json.dumps({'patient_key':patient,'patient_known':known[patient],
                'photos':[{'photo_id':photo,'path':path} for photo,path in photos.items()]})+'\n')
    finally:
        for handle in handles:handle.close()
    save(RUN/'missing_references.json',missing)
    save(marker,{'state':'ready','input_rows':rows,'input_aliases':aliases,'patient_groups':len(groups),
        'unique_reference_tasks':sum(len(photos) for photos in groups.values()),'logical_shards':settings['logical_shards'],
        'image_file_hash_stage':False,'cpu_face_detection':False,'cpu_quality_filter':False,
        'pseudo_identifiers_exported':False,'handler_sha256':sha(RUN/'qwen_annotation_handler.py'),
        'input_aliases_sha256':sha(RUN/'input_aliases.tsv'),
        'shard_checksums':{path.name:sha(path) for path in inputs.glob('*.jsonl')}})


def requests():
    result=[]
    for name in ['single_gpu_allocations.json','a10080_allocations.json']:
        path=RUN/name
        if path.exists():result.extend(item for item in json.loads(path.read_text()).get('allocations',[]) if item.get('allocation_job_id'))
    return result


def launch_granted(settings):
    items=requests()
    if not items:return
    query=subprocess.run(['squeue','-j',','.join(item['allocation_job_id'] for item in items),'-h','-o','%i|%T|%R'],capture_output=True,text=True,timeout=30)
    (RUN/'replica_queue_private.log').write_text(query.stdout+'\n'+query.stderr)
    if query.returncode:return
    states={line.split('|')[0]:line.split('|')[1] for line in query.stdout.splitlines() if '|' in line}
    path=RUN/'replica_launches.json';launched=json.loads(path.read_text()) if path.exists() else {}
    for item in items:
        slot=item['worker'];job=item['allocation_job_id']
        if states.get(job)!='RUNNING' or str(slot) in launched:continue
        command=['srun','--jobid='+job,'--nodes=1','--ntasks=1','--cpus-per-task=4','--gpus-per-task=1',
            '--gpu-bind=single:1','--exclusive',settings['python'],str(RUN/'qwen_replica_service.py'),'--slot',str(slot)]
        with (RUN/'logs'/f'replica_{slot:02d}.log').open('a') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        launched[str(slot)]={'slot':slot,'allocation_job_id':job,'launcher_pid':process.pid,'state':'launched'}
        save(path,launched)
    save(RUN/'deployment_status.json',{'allocated_cards':sum(value=='RUNNING' for value in states.values()),
        'requested_cards':len(items),'updated_at':time.time()})


def finalize(settings):
    records={};window_counts=defaultdict(int)
    for shard in range(settings['logical_shards']):
        path=RUN/'direct_workers'/f'shard_{shard:02d}.sqlite'
        db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('Result database validation failed')
        for photo,patient,path,known,status,label,width,height,duplicate in db.execute('SELECT photo_id,patient_key,path,patient_known,status,label,width,height,duplicate_of FROM labels'):
            if photo in records:raise RuntimeError('Duplicate image task across shards')
            records[photo]=(patient,path,known,status,json.loads(label) if label else None,width,height,duplicate)
        for state,count in db.execute('SELECT state,count(*) FROM comparisons GROUP BY state'):window_counts[state]+=count
        db.close()
    manifest=json.loads((RUN/'direct_input_manifest.json').read_text())
    expected=set();paths=defaultdict(set)
    with (RUN/'input_aliases.tsv').open() as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            if row['photo_id']:
                expected.add(row['photo_id']);paths[row['original_path']].add(row['patient_key'])
    if set(records)!=expected:raise RuntimeError('Final reference coverage mismatch')
    conflicts={path for path,patients in paths.items() if len(patients)>1}
    columns=['photo_id','patient_key','original_path','decision','reason','view','width','height','face_bbox','qwen_duplicate_of']
    counts=defaultdict(int)
    with (RUN/'accepted_faces.tsv').open('w',newline='') as good,(RUN/'excluded_or_hold.tsv').open('w',newline='') as held:
        writers={True:csv.DictWriter(good,fieldnames=columns,delimiter='\t'),False:csv.DictWriter(held,fieldnames=columns,delimiter='\t')}
        for writer in writers.values():writer.writeheader()
        for photo,(patient,path,known,status,label,width,height,duplicate) in records.items():
            accepted=status=='ok' and bool(known) and label is not None and eligible(label) and not duplicate and path not in conflicts
            reason=('screening_passed' if accepted else 'conflicting_patient_reference' if path in conflicts else 'unresolved_patient_group' if not known else 'qwen_near_duplicate' if duplicate else status if status!='ok' else 'visual_criteria_not_met')
            counts[reason]+=1
            writers[accepted].writerow({'photo_id':photo,'patient_key':patient,'original_path':path,'decision':'face_candidate' if accepted else 'exclude_or_hold',
                'reason':reason,'view':label['view'] if label else '', 'width':width or '', 'height':height or '',
                'face_bbox':json.dumps(label['face_bbox']) if label else '', 'qwen_duplicate_of':duplicate or ''})
    summary={'state':'annotation_complete','completed_at':time.time(),'unique_reference_tasks':len(records),'complete_reference_coverage':True,
        'decisions':dict(counts),'qwen_duplicate_review_windows':dict(window_counts),'identity_conflict_paths':len(conflicts),
        'cpu_face_filter_used':False,'cpu_image_hash_stage_used':False,'original_files_deleted':False,
        'accepted_faces_sha256':sha(RUN/'accepted_faces.tsv'),'excluded_or_hold_sha256':sha(RUN/'excluded_or_hold.tsv'),
        'not_training_ready':True,'next_stage':'Background-preserving crops, crop validation, source/reservation checks and patient-isolated split.',
        'limits':['Automated visual labels have no independent expert accuracy estimate.',
                  'Failed/uncertain visual duplicate comparisons retain images; no complete pixel-level duplicate guarantee.',
                  'Identity groups rely on supplied pseudo MRN; known family/protected evaluation groups require later validation.']}
    save(RUN/'summary.json',summary);save(RUN/'completion.json',summary)
    notify('annotation completed','Local Qwen annotation and within-patient visual duplicate review have completed. Candidate files and detailed validation remain on HPC. Cropping and final training-data acceptance are next.')


def main():
    lock=(RUN/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    settings=json.loads((RUN/'config.json').read_text())
    input_references(settings)
    save(RUN/'annotation_start.json',{'state':'ready','handler_sha256':sha(RUN/'qwen_annotation_handler.py'),'created_at':time.time()})
    while True:
        if (RUN/'completion.json').exists():return
        launch_granted(settings)
        replicas=[json.loads(path.read_text()) for path in (RUN/'replicas').glob('slot_*.json')]
        if any(item.get('state') in {'ready','annotating','worker_finished'} for item in replicas):
            notify('deployment ready','An offline Qwen3-VL-32B GPU replica has passed synthetic GPU inference. Local image annotation will run without sending dataset content to OpenAI.')
        statuses=[]
        for shard in range(settings['logical_shards']):
            path=RUN/'direct_workers'/f'shard_{shard:02d}.status.json'
            statuses.append(json.loads(path.read_text()).get('state') if path.exists() else 'unclaimed')
        if all(state=='completed' for state in statuses):finalize(settings);return
        failed=any(state=='failed' for state in statuses) or any(item.get('state')=='failed' for item in replicas)
        if failed:notify('attention required','A local Qwen worker reported a failure. Other workers continue. Private diagnostics remain in the local run directory; no training release has been declared complete.')
        save(RUN/'supervisor_status.json',{'state':'running_with_failure' if failed else 'running',
            'completed_shards':statuses.count('completed'),'logical_shards':settings['logical_shards'],'updated_at':time.time()})
        time.sleep(20)


if __name__=='__main__':
    try:main()
    except Exception:
        with (RUN/'supervisor_error.log').open('a') as handle:traceback.print_exc(file=handle)
        save(RUN/'supervisor_status.json',{'state':'failed','updated_at':time.time()})
        try:notify('attention required','The local annotation supervisor needs attention. Details remain on HPC.')
        except Exception:pass
        print('SUPERVISOR_FAILED_DETAILS_RETAINED_ON_HPC',flush=True)
        raise SystemExit(1)

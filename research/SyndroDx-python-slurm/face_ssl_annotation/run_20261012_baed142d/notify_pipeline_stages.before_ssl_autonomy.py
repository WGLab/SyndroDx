"""Local stage-only mail; never attach or include dataset content."""
from pathlib import Path
import fcntl
import json
import os
import sqlite3
import subprocess
import time

os.umask(0o077)
RUN=Path(__file__).resolve().parent


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def state(path):
    try:return json.loads(path.read_text()).get('state')
    except (OSError,ValueError):return None


def notify(event,title,message):
    settings=json.loads((RUN/'config.json').read_text())
    recipient=settings.get('notification_email','')
    if not recipient or '\n' in recipient or '\r' in recipient:return
    folder=RUN/'notifications';folder.mkdir(exist_ok=True)
    with (folder/(event+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=folder/(event+'.json')
        if receipt.exists():return
        save(receipt,{'state':'sending','updated_at':time.time()})
        content=('To: '+recipient+'\nSubject: HPC stage completed: '+title+
                 '\nContent-Type: text/plain; charset=utf-8\n\n'+message+
                 '\nImages, MRI, patient information and individual annotations remain on HPC.\n')
        try:
            result=subprocess.run(['/usr/sbin/sendmail','-t','-oi'],input=content,text=True,
                                  capture_output=True,timeout=30)
            save(receipt,{'state':'accepted_by_local_mail_system' if result.returncode==0 else 'failed_review_required',
                          'returncode':result.returncode,'updated_at':time.time()})
        except Exception as error:
            save(receipt,{'state':'uncertain_no_automatic_resend','error_type':type(error).__name__,
                          'updated_at':time.time()})


def screening_finished():
    marker=RUN/'direct_input_manifest.json'
    if not marker.exists():return False
    expected=int(json.loads(marker.read_text())['unique_reference_tasks'])
    processed=0
    for path in (RUN/'direct_workers').glob('shard_*.sqlite'):
        connection=None
        try:
            connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)
            processed+=connection.execute('SELECT count(*) FROM labels').fetchone()[0]
        except sqlite3.Error:return False
        finally:
            if connection is not None:connection.close()
    return processed==expected


def main():
    with (RUN/'stage_notifications.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        folder=RUN/'notifications'
        if not (folder/'image screening completed.json').exists() and screening_finished():
            notify('image screening completed','Image screening',
                   'The first-pass image screening is complete. Exceptions remain held locally. Within-patient duplicate review and downstream processing may still be running.')
        stages=[
            ('completion.json','annotation_complete','annotation completed','Annotation and duplicate review',
             'Offline image annotation and within-patient automatic duplicate review are complete. Uncertain cases remain held locally. Cropping and training-data validation are next.'),
            ('crop_completion.json','crop_complete','cropping completed','Cropping',
             'Background-preserving cropping is complete. Crop validation is required before training release.'),
            ('crop_validation_completion.json','crop_validation_complete','crop validation completed','Crop validation',
             'Crop validation is complete. Patient/source checks and dataset splitting are required before training release.'),
            ('dataset_split_completion.json','patient_isolated_split_complete','dataset split completed','Patient-isolated dataset split',
             'Patient-isolated train, validation and test manifests have been produced. Final readiness checks are next.'),
            ('ssl_data_ready.json','ssl_data_ready','ssl data ready','SSL data readiness',
             'The local training-data release has passed readiness checks and is ready for SSL pretraining. This notification does not mean model training has started.'),
            ('ssl_training_completion.json','ssl_training_complete','ssl training completed','SSL training',
             'The configured SSL training run is complete. Model evaluation is the next stage.'),
        ]
        for marker,required,event,title,message in stages:
            if state(RUN/marker)==required:notify(event,title,message)
        save(RUN/'stage_notifications_status.json',{'state':'checked','updated_at':time.time()})


if __name__=='__main__':
    main()

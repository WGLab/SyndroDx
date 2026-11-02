"""Send one aggregate-only email after an explicit final cleanup acceptance record."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'face_qc' / 'completion_notification'
GATES = ('automatic_review', 'quality_review', 'deduplication', 'crop_validation',
         'patient_split_and_reservation_checks')


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def accepted_completion(config):
    marker = Path(config['completion_record'])
    if not marker.exists():
        return None
    data = json.loads(marker.read_text())
    if data.get('state') != 'cleanup_complete':
        return None
    if not all(data.get('gates', {}).get(gate) == 'passed' for gate in GATES):
        return None
    if data.get('expected_original_paths') != config['expected_original_paths']:
        return None
    if data.get('accounted_original_paths') != config['expected_original_paths']:
        return None
    if data.get('pending_decisions') != 0:
        return None
    counts = data.get('counts', {})
    if not all(type(counts.get(k)) is int and counts[k] >= 0
               for k in ('retained', 'excluded', 'resolved_hold', 'unreadable_or_missing')):
        return None
    if sum(counts[k] for k in ('retained', 'excluded', 'resolved_hold',
                               'unreadable_or_missing')) != config['expected_original_paths']:
        return None
    for name in ('cleaned_inventory', 'validation_report'):
        artifact = data.get('artifacts', {}).get(name, {})
        path = Path(artifact.get('path', ''))
        if not path.is_absolute() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != artifact.get('sha256'):
            return None
    return data


def run(state_dir=STATE, dry_run=False):
    state_dir.mkdir(exist_ok=True)
    with (state_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((state_dir / 'config.json').read_text())
        receipt = state_dir / 'delivery.json'
        if receipt.exists():
            return 'already_attempted'
        completion = accepted_completion(config)
        if completion is None:
            write_json(state_dir / 'status.json', {
                'state': 'waiting_for_final_cleanup_acceptance', 'checked_at': time.time()})
            return 'waiting'
        if dry_run:
            return 'ready'
        message = EmailMessage()
        message['From'] = config['sender']
        message['To'] = config['recipient']
        message['Subject'] = 'Craniofacial data cleanup completed / 数据清理完成'
        message['Date'] = formatdate(localtime=True)
        message['Message-ID'] = make_msgid()
        counts = completion['counts']
        message.set_content(
            '数据清理已完成，并生成最终验收记录。\n\n'
            '质量复核、去重、裁剪验证、患者划分及保留集检查均已通过。\n'
            f"原始路径总数：{completion['expected_original_paths']:,}\n"
            f"保留：{counts['retained']:,}\n"
            f"排除：{counts['excluded']:,}\n"
            f"已裁定暂缓使用：{counts['resolved_hold']:,}\n"
            f"无法读取或缺失：{counts['unreadable_or_missing']:,}\n\n"
            '此通知表示清理验收完成，不表示已启动训练。\n'
            '详细验收记录保存在项目的 face_qc/final_cleanup/completion.json。\n'
            '本邮件不附患者身份信息或照片。\n')
        delivery = {'state': 'sending', 'attempted_at': time.time(),
                    'message_id': str(message['Message-ID']), 'recipient': config['recipient']}
        # Persist before handing off: an uncertain/crashed attempt must not send duplicates.
        write_json(receipt, delivery)
        try:
            result = subprocess.run(['/usr/sbin/sendmail', '-oi', '-t'],
                                    input=message.as_bytes(), capture_output=True, timeout=60)
            delivery['state'] = 'accepted_by_local_mail_system' if result.returncode == 0 else 'failed'
            delivery['returncode'] = result.returncode
            delivery['stderr'] = result.stderr.decode(errors='replace')[-2000:]
        except Exception as error:
            delivery['state'] = 'delivery_uncertain_requires_review'
            delivery['error'] = str(error)
        write_json(receipt, delivery)
        write_json(state_dir / 'status.json', delivery)
        return delivery['state']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        result = run(dry_run=args.dry_run)
        if result not in ('waiting', 'already_attempted'):
            print(result, flush=True)
    except BlockingIOError:
        pass
    except Exception as error:
        print(f'Notification check failed: {error}', flush=True)
        raise

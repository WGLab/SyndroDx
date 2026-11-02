"""Private, read-only live report for an offline Qwen annotation run."""
import argparse
import csv
import fcntl
import html
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

os.umask(0o077)
REFRESH_SECONDS = 15
RECENT_LIMIT = 100
TZ = ZoneInfo('America/New_York')


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def screening_candidate(label):
    return (label.get('content') == 'human_face_photo'
            and label.get('face_count') == 'one'
            and label.get('completeness') == 'complete'
            and label.get('view') in {'frontal', 'profile', 'oblique'}
            and label.get('quality') == 'usable'
            and label.get('orientation') == 'upright'
            and label.get('context') == 'ordinary'
            and label.get('prominence') == 'adequate'
            and label.get('certainty') == 'high')


def read_status(path):
    try:
        return json.loads(path.read_text()).get('state', '')
    except (OSError, ValueError):
        return ''


def collect(run):
    rows = []
    processed = ok = unavailable = 0
    for path in sorted((run / 'direct_workers').glob('shard_*.sqlite')):
        connection = None
        try:
            connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0.5)
            connection.execute('PRAGMA query_only=ON')
            processed += connection.execute('SELECT count(*) FROM labels').fetchone()[0]
            ok += connection.execute("SELECT count(*) FROM labels WHERE status='ok'").fetchone()[0]
            for item in connection.execute('''
                SELECT path,status,label,patient_known,duplicate_of,updated_at
                FROM labels ORDER BY updated_at DESC LIMIT ?
            ''', (RECENT_LIMIT,)):
                image_path, status, encoded, known, duplicate, updated = item
                try:
                    label = json.loads(encoded or '{}')
                    if not isinstance(label, dict):
                        label = {}
                except ValueError:
                    label = {}
                passed = status == 'ok' and screening_candidate(label)
                if status != 'ok':
                    decision = '处理失败 / 待复核'
                    style = 'hold'
                elif duplicate:
                    decision = 'Qwen 相似图候选'
                    style = 'hold'
                elif passed and known:
                    decision = '筛选通过 · 暂定候选'
                    style = 'pass'
                elif passed:
                    decision = '患者分组待确认'
                    style = 'hold'
                else:
                    decision = '未通过 / 待复核'
                    style = 'hold'
                rows.append({'path': image_path, 'label': label, 'decision': decision,
                             'style': style, 'updated': float(updated or 0)})
        except (sqlite3.Error, OSError):
            unavailable += 1
        finally:
            if connection is not None:
                connection.close()
    rows.sort(key=lambda row: row['updated'], reverse=True)
    shard_states = [read_status(p) for p in (run / 'direct_workers').glob('shard_*.status.json')]
    replica_states = [read_status(p) for p in (run / 'replicas').glob('slot_*.json')]
    return rows[:RECENT_LIMIT], {
        'processed': processed, 'ok': ok, 'unavailable': unavailable,
        'active_gpu': replica_states.count('annotating'),
        'completed_shards': shard_states.count('completed'),
        'failed_shards': shard_states.count('failed'),
        'duplicate_review': shard_states.count('qwen_patient_duplicate_review'),
        'complete': (run / 'completion.json').exists(),
    }


VIEW_NAMES = {'frontal': '正脸', 'profile': '侧脸', 'oblique': '斜侧脸',
              'other': '其他', 'uncertain': '不确定'}
NAMES = {'high': '高', 'medium': '中', 'low': '低', 'usable': '可用',
         'poor': '较差', 'uncertain': '不确定', 'complete': '完整',
         'partial': '不完整', 'one': '单人脸', 'multiple': '多人脸', 'none': '无人脸'}


def render(run):
    rows, state = collect(run)
    now = datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')
    stage = '全部标注已完成' if state['complete'] else '标注进行中'
    table = []
    for index, row in enumerate(rows, 1):
        label = row['label']
        stamp = datetime.fromtimestamp(row['updated'], TZ).strftime('%m-%d %H:%M:%S')
        try:
            photo_uri = Path(row['path']).resolve().as_uri()
        except (ValueError, OSError):
            photo_uri = ''
        image_link = ('<a href="' + html.escape(photo_uri, quote=True)
                      + '" target="_blank" rel="noreferrer">打开本地原图</a>') if photo_uri else '无路径'
        values = [str(index), stamp, image_link,
                  html.escape(VIEW_NAMES.get(label.get('view'), '—')),
                  html.escape(NAMES.get(label.get('face_count'), '—')),
                  html.escape(NAMES.get(label.get('completeness'), '—')),
                  html.escape(NAMES.get(label.get('quality'), '—')),
                  html.escape(NAMES.get(label.get('certainty'), '—')),
                  '<span class="' + row['style'] + '">' + row['decision'] + '</span>']
        table.append('<tr>' + ''.join('<td>' + value + '</td>' for value in values) + '</tr>')
    if not table:
        table.append('<tr><td colspan="9">等待结果写入；页面会自动刷新。</td></tr>')
    warnings = []
    if state['unavailable']:
        warnings.append('部分分片暂时不可读，已显示其余结果；下次刷新重试。')
    if state['failed_shards']:
        warnings.append('有分片报告处理失败，需要检查本地任务状态。')
    page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="refresh" content="15">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Qwen 实时标注</title>
<style>
body{font:15px system-ui,sans-serif;background:#f4f6f9;color:#17243b;margin:0;padding:30px}
main{max-width:1500px;margin:auto}h1{font-size:27px;margin:0 0 8px}.muted{color:#56647b}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}.card{background:white;padding:16px 24px;border-radius:12px;min-width:140px}
.number{display:block;font-size:30px;font-weight:650}.notice{background:#fff6d8;padding:13px 18px;border-radius:10px;margin:14px 0}
.table{overflow-x:auto;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;white-space:nowrap}
th,td{padding:13px 16px;text-align:left;border-bottom:1px solid #e9edf3}th{background:#eaf0f7;font-size:13px}
a{color:#2359a9}.pass{color:#17704b}.hold{color:#9b650f}p{line-height:1.6}
</style></head><body><main>
<h1>Qwen3-VL-32B 实时标注</h1>
<p class="muted">STAGE · 每 15 秒刷新 · 最近一次读取：STAMP</p>
<div class="cards">
<div class="card">已写入记录<span class="number">PROCESSED</span></div>
<div class="card">成功解析的标注<span class="number">OK</span></div>
<div class="card">正在标注的 GPU<span class="number">GPUS</span></div>
<div class="card">已完成分片<span class="number">SHARDS</span></div>
<div class="card">相似图审核中的分片<span class="number">REVIEW</span></div>
</div>
<div class="notice">显示最近 100 条记录。“筛选通过”是暂定的图像质量判断；患者内相似图审核、汇总检查及后续裁剪可能改变结果。此页面不代表数据已经可以直接用于 SSL 训练。</div>
WARNINGS
<p class="muted">原图链接仅在能够访问 HPC 文件系统的本地浏览器中打开。页面使用本地文件，不调用外部服务，也不向聊天上传图片或标注。</p>
<div class="table"><table><thead><tr><th>序号</th><th>标注时间（纽约）</th><th>图片</th><th>视角</th><th>人脸数</th><th>完整性</th><th>质量</th><th>Qwen 自报确定性</th><th>当前判断</th></tr></thead>
<tbody>ROWS</tbody></table></div>
</main></body></html>'''
    replacements = {'STAGE': stage, 'STAMP': now, 'PROCESSED': str(state['processed']),
                    'OK': str(state['ok']), 'GPUS': str(state['active_gpu']),
                    'SHARDS': str(state['completed_shards']), 'REVIEW': str(state['duplicate_review']),
                    'WARNINGS': ''.join('<div class="notice">' + html.escape(w) + '</div>' for w in warnings),
                    'ROWS': ''.join(table)}
    # Replace only template text, never interpret row text as a template.
    import re
    page = re.sub(r'\b(STAGE|STAMP|PROCESSED|OK|GPUS|SHARDS|REVIEW|WARNINGS|ROWS)\b',
                  lambda match: replacements[match.group()], page)
    atomic_text(run / 'LIVE_RESULTS.html', page)
    temporary = run / 'live_annotations_recent.tsv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['image_path','updated_at','view','face_count','completeness',
                         'quality','qwen_certainty','provisional_decision'])
        for row in rows:
            label = row['label']
            writer.writerow([row['path'], datetime.fromtimestamp(row['updated'], TZ).isoformat(),
                             label.get('view',''), label.get('face_count',''),
                             label.get('completeness',''), label.get('quality',''),
                             label.get('certainty',''), row['decision']])
    temporary.replace(run / 'live_annotations_recent.tsv')
    atomic_text(run / 'live_view_status.json', json.dumps({
        'state': 'completed' if state['complete'] else 'running',
        'updated_at': time.time(), 'pid': os.getpid(),
    }) + '\n')
    return state['complete']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with (args.run / 'live_view.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if not args.once and read_status(args.run / 'live_view_status.json') == 'completed':
            return
        while True:
            try:
                complete = render(args.run)
                if complete or args.once:
                    return
            except Exception:
                with (args.run / 'live_view_error.log').open('a') as handle:
                    traceback.print_exc(file=handle)
                atomic_text(args.run / 'live_view_status.json', json.dumps({
                    'state': 'retrying', 'updated_at': time.time(), 'pid': os.getpid(),
                }) + '\n')
                if args.once:
                    raise SystemExit(1)
            time.sleep(REFRESH_SECONDS)


if __name__ == '__main__':
    main()

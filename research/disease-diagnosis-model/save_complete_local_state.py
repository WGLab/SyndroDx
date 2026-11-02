"""Capture a complete local handoff; never print project/data contents."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
FOLDER = Path((ROOT / 'local_checkpoints/ACTIVE_SAVE.txt').read_text().strip())
ZONE = ZoneInfo('America/New_York')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text()) if p.is_file() else default


def snapshot_files():
    # Copy all relevant code, manifests, annotations, audits, logs and review artifacts.
    # Large legacy training inputs/checkpoints and external originals stay at their
    # existing locations; this is an experiment-state checkpoint, not a source-archive backup.
    excluded = {'.git', '.agents', '.aws', '.codex', '.venv-faceqc', '__pycache__',
                'faces_384', 'ssl_ckpt', 'ssl_ckpt_run1', 'ssl_ckpt_smoke', 'local_checkpoints'}
    selected, referenced, entries = [], [], []
    for directory, dirs, files in os.walk(ROOT):
        base = Path(directory)
        dirs[:] = [d for d in dirs if d not in excluded]
        if base.relative_to(ROOT).parts[:1] == ('data',):
            dirs[:] = [d for d in dirs if d not in {'images_native', 'images_768'}]
        for name in files:
            p = base / name
            if name.endswith(('.sqlite-wal', '.sqlite-shm', '.db-wal', '.db-shm')):
                referenced.append({'path': str(p.relative_to(ROOT)),
                                   'kind': 'incorporated_into_consistent_sqlite_backup'})
                continue
            if p.is_symlink():
                referenced.append({'path': str(p.relative_to(ROOT)), 'kind': 'symlink_not_followed'})
                continue
            if p.suffix.lower() in {'.npy', '.npz', '.pt', '.pth', '.safetensors', '.task'}:
                referenced.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                   'kind': 'retained_in_original_location'})
                continue
            selected.append(p)
    total_bytes, copied = 0, 0
    for p in sorted(selected):
        relative = p.relative_to(ROOT)
        target = FOLDER / 'workspace_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = p.stat()
        if p.suffix.lower() in {'.sqlite', '.db'}:
            # Source databases may be frozen read-only and use WAL. Copy the
            # database plus sidecars locally before opening, so SQLite never
            # tries to update a frozen source or its shared-memory file.
            with tempfile.TemporaryDirectory(prefix='private-state-sqlite-', dir='/tmp') as work:
                staged = Path(work) / p.name
                sidecars = [Path(str(p) + suffix) for suffix in ('-wal', '-shm')]
                sources = [p] + [x for x in sidecars if x.exists()]
                stats = {str(x): (x.stat().st_size, x.stat().st_mtime_ns) for x in sources}
                for source in sources:
                    destination = Path(work) / source.name
                    shutil.copyfile(source, destination)
                    os.chmod(destination, 0o600)
                assert all((x.stat().st_size, x.stat().st_mtime_ns) == stats[str(x)] for x in sources), 'SQLite source changed during staging; retry snapshot'
                if target.exists():
                    os.chmod(target, 0o600)
                    target.unlink()
                src = sqlite3.connect(staged)
                dst = sqlite3.connect(target)
                src.backup(dst)
                dst.close(); src.close()
            for suffix in ('-wal', '-shm'):
                snapshot_sidecar = Path(str(target) + suffix)
                if snapshot_sidecar.exists():
                    snapshot_sidecar.unlink()
            kind = 'sqlite_consistent_backup_from_unchanged_local_stage'
        else:
            shutil.copyfile(p, target)
            kind = 'file_copy'
        after = p.stat()
        os.chmod(target, 0o600)
        size = target.stat().st_size
        entries.append({'path': str(relative), 'snapshot': str(target.relative_to(FOLDER)),
                        'bytes': size, 'sha256': sha(target), 'kind': kind,
                        'source_mtime': before.st_mtime,
                        'source_changed_during_copy': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)})
        copied += 1; total_bytes += size
        if copied % 100 == 0:
            dump(FOLDER / 'SAVE_STATUS.json', {'state': 'copying', 'files_copied': copied,
                                             'bytes_copied': total_bytes, 'updated_at': time.time()})
    dump(FOLDER / 'SNAPSHOT_MANIFEST.json', {'files': entries, 'large_retained_assets': referenced,
                                           'excluded_directory_names': sorted(excluded),
                                           'source_originals': 'External original photographs are retained at indexed HPC paths, not copied into the checkpoint.',
                                           'files_copied': copied, 'bytes_copied': total_bytes})
    return entries, copied, total_bytes


def save_reports(entries, copied, total_bytes):
    now = datetime.now(ZONE).isoformat()
    q = load('face_qc/qwen_stream_v5/summary.json', {})
    coverage = load('face_qc/qwen_stream_v5/coverage_checks.json', {})
    cpu = load('face_qc/full_original_audit_v1/summary.json', {})
    review = load('face_qc/qwen_stream_v5/review_analysis/summary.json', {})
    admission = load('ssl_admission_inventory/summary.json', {})
    pilot = load('background_preserving_pilot_v2/review_summary.json', {})
    mail = load('face_qc/completion_notification/status.json', {})
    delivery = load('face_qc/completion_notification/delivery.json', None)
    release = ROOT / 'data/pretrain_faces_v1'
    required_scripts = ['face_qc/pretrain_release.py', 'face_qc/pretrain_crop.py',
                        'face_qc/pretrain_qc_prompt.txt', 'face_qc/pretrain_visual_qc.py',
                        'face_qc/finalize_pretrain_release.py']
    implementation = {name: {'exists': (ROOT / name).is_file(),
                             'sha256': sha(ROOT / name) if (ROOT / name).is_file() else None}
                      for name in required_scripts}
    completed_times = []
    for worker in q.get('workers', []):
        if worker.get('state') == 'completed':
            completed_times.append(datetime.fromtimestamp(worker['updated_at_unix'], ZONE).isoformat())
    release_files = {name: (release / name).is_file() for name in
                     ['manifest.json', 'launch.json', 'train.tsv', 'dev.tsv', 'validation_report.json', 'release.json']}
    counts = collections.Counter()
    if (ROOT / 'face_qc/qwen_stream_v5/results.tsv').is_file():
        with (ROOT / 'face_qc/qwen_stream_v5/results.tsv').open() as f:
            for row in csv.DictReader(f, delimiter='\t'):
                counts['indexed_paths'] += 1
                counts['status:' + row['status']] += 1
                counts['triage:' + row['triage']] += 1
                if row['triage'] == 'face_candidate_quality_pending':
                    counts['candidate_view:' + row['view']] += 1
    state = {'saved_at': now, 'workspace': str(ROOT), 'snapshot': str(FOLDER),
             'cpu_audit': cpu, 'qwen_summary': q, 'qwen_final_coverage': coverage,
             'last_worker_completed_at_new_york': max(completed_times) if completed_times else None,
             'merged_coverage_file_mtime_new_york': datetime.fromtimestamp((ROOT / 'face_qc/qwen_stream_v5/coverage_checks.json').stat().st_mtime, ZONE).isoformat() if coverage else None,
             'recounted_result_inventory': dict(counts), 'review_analysis': review,
             'legacy_ssl_admission_inventory': admission, 'background_preserving_pilot_review': pilot,
             'new_release_implementation': implementation, 'new_release_artifacts': release_files,
             'completion_notification': mail, 'email_delivery_receipt': delivery,
             'snapshot_files': copied, 'snapshot_bytes': total_bytes,
             'current_phase': 'Qwen original screening completed; new final cropping/QC release preparation incomplete and not submitted unless live scheduler snapshot shows otherwise.',
             'not_complete': ['No frozen usable pretraining release', 'No bulk crop/QC job submitted by this workflow',
                              'New release GPU QC worker and finalizer missing', 'Near-duplicate/patient split validation incomplete',
                              'No new pretraining run'],
             'privacy_boundary': 'User explicitly forbids any dataset content being transmitted to OpenAI; detailed data and logs remain local. Only content-free execution state may be returned.',
             'records_cleared': False, 'clearing_scope': 'User clarification pending: conversation context versus HPC logs.'}
    dump(FOLDER / 'CURRENT_STATE.json', state)
    report = f'''# 完整进度保存（仅限 HPC 本地查看）

保存时间：{now}，America/New_York。

## 当前做到什么程度

1. **全量原图 CPU 审计完成**：{cpu.get('images', 'unknown')} 条索引路径。已记录文件是否存在、原图哈希、近似哈希、尺寸、EXIF、初步检测与裁剪框。此审计未批量导出训练裁剪。
2. **全量 Qwen 原图内容／视角标注完成并合并**：{q.get('processed_paths', 'unknown')} / {q.get('expected_original_paths', 'unknown')} 条路径，成功标注 {q.get('qwen_labelled_paths', 'unknown')} 条。所有工作路完成：{q.get('all_workers_completed')}。最终唯一 ID 覆盖检查：{coverage.get('complete_coverage')}；重复 ID：{coverage.get('duplicate_ids')}。
3. 最后一条工作路完成时间：{state['last_worker_completed_at_new_york']}；覆盖检查文件生成时间：{state['merged_coverage_file_mtime_new_york']}。
4. **尚未得到最终可用训练集**。旧准入清单仍为 {admission.get('admitted_images', 'unknown')} 张；新流程尚在实现阶段。Qwen 标签为候选判断，不是逐图最终质量通过。
5. 已有背景保留试验：复核 {pilot.get('reviewed', 'unknown')} 张，明确后续项 {pilot.get('specific_followup', 'unknown')} 张，最终质量批准 {pilot.get('final_quality_approved', 'unknown')} 张。
6. 没有启动新的正式预训练。

## 全量 Qwen 的路径级分类

```json
{json.dumps(q.get('status_counts', {}), ensure_ascii=False, indent=2)}
```

```json
{json.dumps(q.get('triage_counts', {}), ensure_ascii=False, indent=2)}
```

这些数字是路径数，不是患者数。待复核内的文件缺失记录不能算成可看图病例。近似哈希、模型视角和 certainty 均不等于经独立标注校准的准确率。

## 待复核的实际类型

```json
{json.dumps(review, ensure_ascii=False, indent=2)}
```

主要原因分组互斥；底层 flags 可重叠。完全相同原图可共享视觉复核，但患者身份、保留集和来源标记不能仅因视觉重复而解除。

## 尚未提交的新清理流程

用户已授权：直接完成数据清理，交付可用于预训练的数据，而不只是建议。已写的新代码存在情况：

```json
{json.dumps(implementation, ensure_ascii=False, indent=2)}
```

新发布目录状态：

```json
{json.dumps(release_files, ensure_ascii=False, indent=2)}
```

**不要把“代码已开始写”描述成“任务已提交”。** 新 worker、finalizer、近重复审核、数据加载样例、检查与提交尚需完成。现有 prepare 脚本会引用未实现脚本，因此现在不能直接运行并视为完整流程。

## 初步实现的规则，仍需验证

- 复用 Qwen v5 原始标注，不重跑已成功的相同分类任务。
- 保留正脸、斜脸、侧脸。侧脸不要求双眼都可见；检测失败不是非人脸证据。
- 使用可追溯 G_drive 原图；不使用旧去背景裁剪，也不使用来源未核实的 genetics-only 筛选进入首版。
- 专家／Ian 保留、身份冲突、历史 test 患者保持隔离。已有 val/dev 不可悄悄进入 train。
- 同一原图哈希只生成一个代表裁剪，保留路径别名与患者关联。
- 普通矩形裁剪保留真实背景；不得抠图、背景替换、修复补画、正脸化或改变面部几何。
- 没有 CPU 框时暂拟完整原图保留方案，尤其避免丢侧脸；必须由本地实际图像 QC 检查主体大小、完整性与背景范围。这个方案尚未验证，不是自动通过。
- 局部照片、手术图、严重模糊、多人目标未确认及方向争议，首版暂缓并明确用途，不删除原文件。
- 当前采样／质量方案没有独立专家 ground truth；自动模型同意不能写成专家或临床验证。
- 家系独立、疾病 zero-shot 协议尚未确立。不能因完成 SSL 数据包就声称这些条件成立。

## 完成邮件

通知状态：{mail.get('state', 'unknown')}；已有发送回执：{delivery is not None}。
本地定时任务每五分钟检查最终验收记录。只有质量、去重、裁剪和患者隔离完成、最终文件与哈希验收通过后，才能生成通知所需 completion.json。原图 Qwen 完成不触发“最终清理完成”邮件。

## 保存内容与边界

复制并校验了 {copied} 个项目状态文件，共 {total_bytes} 字节；逐项路径、大小和 SHA-256 见 SNAPSHOT_MANIFEST.json。包含代码、标注、进度、输入版本、审计、日志和本地复核材料。SQLite 用一致性备份。

这是一份实验状态与接续备份；外部原始照片和大型历史训练输入／权重仍在原位置，不重复复制整个照片档案。源原图位置、哈希和处理关联已由现有输入／输出清单保存。没有删除原图、标注、日志或聊天记录。完整聊天文本没有可用导出接口，决策与任务状态已重建保存；本地文件删除不能擦除已进入当前对话的内容。

详细调度、邮件服务和软件状态如已捕获，保存在 runtime_snapshot/，不应返回模型上下文。
'''
    (FOLDER / 'CURRENT_STATE.md').write_text(report)
    decisions = '''# 决策与约束（本地接续记录）

## 目标与授权

用户希望直接完成清理，获得可以拿来做视觉自监督预训练的数据。已授权裁剪、质量检查、去重、患者划分和部署必要的本地处理任务。未授权自动启动正式预训练、释放专家/Ian 保留集、删除原始照片或把数据上传到外部模型。

## 数据与研究口径

- 原始 craniofacial 档案覆盖全部可用诊断，不限定 top10，不以 GMDB 疾病空间限制图片。
- MONDO 映射概念与临床可信度、图片准入分别记录。有效 MONDO 的结构性诊断计入用户定义的 diagnosis 口径；不能把照片推断疾病当真值。
- PubMind 及派生诊断、基因、HPO、筛选等全部排除。弱标签、遗传检测推荐和临床断言需要独立来源核验。
- 四个历史候选 subset 已有库存与审计；expert/Ian 全部保护继续生效。它们不是全量多视角档案，也不是已清理的训练集。
- 遗传照片的 metadata／来源路径存在未解决项；缺失源文件不能假定已恢复。
- 完全重复与患者不同图片是不同概念。同一患者的多个视角可保留，但患者必须跨视角统一划分。

## 图像处理

- 用户明确拒绝去背景。只用原图上的常规矩形裁剪并保留范围内真实背景。
- 多视角范围已覆盖旧“仅正脸”口径：正脸、斜侧脸、侧脸均保留审核。
- 整图旋转与头部俯仰分开；过去发现自动旋转搜索及 Qwen 方向标签会误判。不得未经核验自动把所有 rotated 图转 90 度。
- 多人照片需要可信 metadata 确定目标，不能按最大脸、儿童脸或外貌猜测患者。
- 局部耳／嘴／身体与手术图可有独立研究用途，但首版完整面部 SSL 不直接混入。
- 最终预训练数据不带未经核实的疾病监督标签。

## 模型／训练讨论，尚未执行

- 用户偏好能训练的最大模型。讨论主路线为公开 DINOv3 ViT-7B/16 权重上的领域内继续自监督；H+/16 可作为规模对照。可行性仍需显存、吞吐和训练稳定性实测。
- 预训练初期采用同一图片的多种增强；不强制跨年龄、术前术后或未经核验患者多图拉近。
- 调整多裁剪以保留重要面部结构，温和光照增强，谨慎模糊和翻转，不做面部几何扭曲或生成式修复。
- 按患者／拍摄组／视角采样，限制连拍和大量图片患者的过度权重。
- 先建立冻结公开骨干基线，再评价领域适配；以患者隔离的下游性能为依据，不能只看 SSL loss。
- SSL 本身不保证疾病 zero-shot。后续还需疾病／HPO 知识对齐、可信临床监督、排序与校准，再构建 agent 流程。没有锁定疾病 zero-shot 数据协议。
- 目标期刊影响力没有保证；数据珍贵与算力充足是条件，不是已经验证的研究贡献。

## 资源与效率

- 用户要求通过 salloc 保留多卡，H100 分配八天；后续加 A100 80GB。
- GPU v5 使用独立内容哈希分配，迁移保留已有结果，生产内容不重复推理。
- 旧步骤可停止但 allocation 按用户要求保留。新任务必须查当前调度状态，不根据旧对话假定卡仍可用。
- 批处理代码和 prompt 要冻结、版本化、可续跑；不可修改正在运行的固定版本。

## 隐私限制，覆盖此前图片展示授权

用户明确：绝不把任何项目数据泄漏给 OpenAI。后续照片、病历、患者身份、逐条标注、样例、统计和详细日志全部留在 HPC。只使用本地缓存模型，禁用 telemetry、外部追踪和外部推理。工具只能返回不含数据的执行状态。不得再使用 view_image、图片展示、临床表行读取或向聊天打印数据。

此前展示过的照片和工具内容已经进入当前模型上下文，不能撤回。不能宣称整个历史对话从未经过 OpenAI。新限制只能控制后续处理。

用户另要求保存完整状态后清空“记录”，聊天上下文还是 HPC 日志的范围未明确。尚未删除任何记录。归档不等于删除。当前没有可调用的聊天删除／清空上下文工具。
'''
    (FOLDER / 'DECISIONS_AND_CONSTRAINTS.md').write_text(decisions)
    next_steps = '''# 后续操作与验收

1. 先读取 DATA_PRIVACY.md 和安全接续说明；不得把私人状态文件打印到 OpenAI。只让本地脚本读取并返回通用执行状态。
2. 用 runtime_snapshot 中的现状或重新本地查询确认 allocation、剩余时间、CPU 队列与存储。没有提交新清理任务，不要把完成的 Qwen 作业当正在跑裁剪。
3. 完成缺失的本地视觉 QC worker 和 release finalizer，检查新 prepare/crop 脚本、资源配置、checkpoint 锁、哈希验证、失败记录、resume 和无重复处理。
4. 先在合成图片／虚构患者上测试原图改变、裁剪越界、EXIF、内容别名、跨患者/保留集传播、近重复隔离、损坏输出、续跑和失败通知等关键分支。不得用正式数据写伪专家真值。
5. 准备固定的本地试处理样本，生成原图矩形裁剪，使用离线本地 Qwen 检查实际输出。必要的人工复核只能在本地进行，不能让 OpenAI 看图。明确区分自动 QC、抽样检查和独立专家验证。
6. 验证试处理后提交 CPU 批量裁剪与必要的本地 GPU 疑难/QC 任务。原有成功分类直接复用，按原图哈希只处理一份视觉内容，别名与身份检查仍逐条保存。
7. 完成原图/裁剪重复检查、患者/已知关联组隔离、专家/Ian/historical test 防泄漏与来源校验。未知家系关系及疾病 zero-shot 协议属于明确限制，不能伪造通过。
8. 输出 native 裁剪、训练图像、train/dev 清单、原图哈希/坐标/输出哈希、别名与患者分组、独立用途池、排除/暂缓原因、统计和数据加载样例。原始数据不删。
9. 对最终图片实际解码／文件哈希、清单唯一性、训练/开发患者不相交、保留集/内容不交叉、训练准入条件、覆盖账目及局部/操作图隔离作验收。所有 unresolved 项从首版清单暂缓，明确这并非整个档案全部可用。
10. 只有真实验收完成，才写本地 completion.json 接通已授权邮件；只通过本地邮件系统发汇总给用户。不能调用 OpenAI 可见的邮件 connector。发送一次有 receipt，模糊失败不自动重复发送。

不要自动开始正式预训练。本次授权的交付物是可用数据版本及可追溯的暂缓/排除记录。
'''
    (FOLDER / 'NEXT_STEPS_AND_ACCEPTANCE.md').write_text(next_steps)
    dump(FOLDER / 'VALIDATION.json', {
        'all_copied_files_have_sha256': all(len(x['sha256']) == 64 for x in entries),
        'unique_snapshot_paths': len(entries) == len({x['snapshot'] for x in entries}),
        'cpu_and_qwen_summaries_saved': bool(cpu) and bool(q),
        'qwen_final_coverage_saved': bool(coverage),
        'new_pipeline_missing_files_documented': any(not v['exists'] for v in implementation.values()),
        'privacy_constraints_saved': (FOLDER / 'DECISIONS_AND_CONSTRAINTS.md').is_file(),
        'no_source_files_deleted': True, 'snapshot_not_chat_deletion': True,
        'snapshot_not_complete_source_archive_backup': True,
        'files_changed_during_copy': [x['path'] for x in entries if x['source_changed_during_copy']]})
    dump(FOLDER / 'SAVE_STATUS.json', {'state': 'saved', 'files_copied': copied,
                                     'bytes_copied': total_bytes, 'updated_at': time.time(),
                                     'runtime_snapshot': 'May be added after this filesystem snapshot.'})
    (ROOT / 'local_checkpoints/LATEST_COMPLETE.txt').write_text(str(FOLDER) + '\n')
    (ROOT / 'local_checkpoints/LATEST.txt').write_text(str(FOLDER) + '\n')
    (ROOT / 'SAVED_STATE.md').write_text(
        '# 本地完整状态记录\n\n'
        '以下文件包含项目详细进度，只在 HPC 本地查看，不能打印到 OpenAI 对话。\n\n'
        f'- [当前进度]({FOLDER}/CURRENT_STATE.md)\n'
        f'- [完整结构化状态]({FOLDER}/CURRENT_STATE.json)\n'
        f'- [决策与约束]({FOLDER}/DECISIONS_AND_CONSTRAINTS.md)\n'
        f'- [接续步骤与验收]({FOLDER}/NEXT_STEPS_AND_ACCEPTANCE.md)\n'
        f'- [备份文件与哈希]({FOLDER}/SNAPSHOT_MANIFEST.json)\n'
        f'- [保存验证]({FOLDER}/VALIDATION.json)\n\n'
        '这是实验状态备份，包含代码、标注、进度、日志、审计和版本。原始照片与大型历史训练资产保留在原位置。没有删除数据、日志或聊天。\n')
    (ROOT / 'LOCAL_RESUME.md').write_text(
        '# Safe resume instructions\n\n'
        'Resume the HPC-local image cleaning and pretraining-data preparation. The user prohibits transmitting any dataset content to OpenAI. Keep photos, identifiers, clinical records, per-image labels, sample previews, statistics and detailed logs entirely on HPC. Only content-free execution status may be returned. Use cached offline local Qwen; disable telemetry and remote tracking.\n\n'
        'Full state is preserved locally and indexed by SAVED_STATE.md and local_checkpoints/LATEST_COMPLETE.txt. Do not print those private files or snapshots into the model context. Inspect them through local programs with redirected output. CPU original audit and Qwen screening completed; final usable data release is unfinished. New crop/QC implementation is incomplete and has not been submitted by this workflow. Complete missing generic code, validate locally, submit the pilot and bulk processing, produce a background-preserving patient-isolated pretraining data version with traceable holds, and trigger the existing local mail notifier only after evidence-backed release acceptance. No new pretraining is authorized. No records have been deleted; clarification about clearing scope is still pending.\n')


if __name__ == '__main__':
    os.chmod(FOLDER, 0o700)
    try:
        files, n, b = snapshot_files()
        save_reports(files, n, b)
        print('COMPLETE_LOCAL_STATE_SAVED', flush=True)
    except Exception as error:
        dump(FOLDER / 'SAVE_STATUS.json', {'state': 'save_failed', 'error_type': type(error).__name__, 'updated_at': time.time()})
        with (FOLDER / 'LOCAL_ERROR.log').open('a') as log:
            import traceback
            traceback.print_exc(file=log)
        print('LOCAL_SAVE_FAILED_DETAILS_RETAINED_LOCALLY', flush=True)
        raise SystemExit(1)

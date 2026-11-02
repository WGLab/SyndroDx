"""Build a fail-closed SSL preparation ledger, never a training release.

Uses conservative candidates and existing reservation guards. Automated screening
of originals is tracked separately from approval of training crops. Outputs with
patient IDs/paths are local controlled manifests; REPORT.md is aggregate-only.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from expert_reservation import filter_reserved

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ssl_admission_inventory'
INPUTS = [
    'four_subset_inventory/image_subsets.tsv',
    'data_release_audit/chop_patient_consistent_candidates.tsv',
    'data_release_audit/image_manifest.tsv',
    'data_release_audit/patient_manifest.tsv',
    'face_qc/subset4_screen_v1/results.tsv',
    'expert_holdout_audit/protected_mrns.tsv',
    'expert_holdout_audit/blocked_paths.tsv',
    'expert_holdout_audit/blocked_crop_hashes.tsv',
    'split_plan_v0/ian_protected_mrns.tsv',
    'split_plan_v0/ian_blocked_paths.tsv',
    'split_plan_v0/ian_blocked_crop_hashes.tsv',
    'split_plan_v0/patient_split_proposal.tsv',
    'DATA_SCOPE.md', 'expert_reservation.py', 'audit_metadata_sources.py',
]


def read(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(name, rows, fields=None):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status(reserved, technical, screen):
    if reserved:
        return 'excluded_reserved'
    if not technical:
        return 'pending_technical_or_identity_review'
    if screen == 'processing_error':
        return 'pending_screen_retry'
    if screen == 'not_screened_in_subset4_run':
        return 'pending_view_and_crop_qc'
    if screen == 'frontal_candidate':
        return 'pending_screen_validation_and_crop_qc'
    if screen == 'nonfrontal_candidate':
        return 'pending_nonfrontal_review'
    return 'pending_uncertain_view_review'


def main():
    OUT.mkdir(exist_ok=True)
    images = read(INPUTS[0])
    candidates = read(INPUTS[1])
    full = read(INPUTS[2])
    patient_records = {r['mrn']: r for r in read(INPUTS[3])}
    qc = read(INPUTS[4])
    plans = {r['mrn']: r for r in read('split_plan_v0/patient_split_proposal.tsv')}
    byhash = {r['crop_sha256']: r for r in candidates}
    qchash = {r['upstream_crop_sha256']: r for r in qc}
    assert len(byhash) == len(candidates) == len(images)
    assert len(qchash) == len(qc)
    assert {r['crop_sha256'] for r in images} == set(byhash)
    assert set(qchash) == {r['crop_sha256'] for r in images if r['subset'] == '4'}
    assert all(r['training_admitted'] == 'False' for r in images)

    # Check every available original/copy alias, not just the retained crop row.
    kept_full, _ = filter_reserved(full)
    kept_object_ids = {id(r) for r in kept_full}
    reserved_hashes = {r['crop_sha256'] for r in full
                       if id(r) not in kept_object_ids and r['crop_sha256']}
    kept_candidates, _ = filter_reserved(candidates)
    guarded_hashes = {r['crop_sha256'] for r in candidates} - {r['crop_sha256'] for r in kept_candidates}
    all_reserved_hashes = reserved_hashes | guarded_hashes
    initially_reserved = {r['mrn'] for r in images if r['subset'] == '1'}
    propagated_reserved = initially_reserved | {r['mrn'] for r in images
                                                if r['crop_sha256'] in all_reserved_hashes}
    ledger = []
    for r in images:
        h, p = r['crop_sha256'], r['mrn']
        c = byhash[h]
        assert (p, r['source'], r['cropping_path']) == (c['mrn'], c['source'], c['cropping_path'])
        q = qchash.get(h, {})
        if q:
            assert (q['mrn'], q['source'], q['image_path']) == (p, c['source'], c['image_path'])
        blocked = p in propagated_reserved
        tech = (c['technical_candidate'] == 'True' and c['eligible_identity_candidate'] == 'True'
                and c['crop_file_status'] == 'decoded' and c['duplicate_identity_conflict'] == 'False'
                and c['has_other_identity_conflict'] == 'False')
        decision = q.get('decision', 'not_screened_in_subset4_run')
        error = 'processing_error' in q.get('reasons', '')
        screen = 'processing_error' if error else decision
        reasons = ['expert_or_Ian_or_linked_identity_reserved'] if blocked else [
            'patient_partition_not_locked', 'zero_shot_disease_policy_not_locked',
            'family_and_near_duplicate_review_pending', 'final_training_crop_qc_pending',
            'current_file_revalidation_pending']
        if not blocked:
            reasons.append(status(False, tech, screen))
        ledger.append(dict(
            mrn=p, subset=r['subset'], subset_name=r['subset_name'], source=r['source'],
            crop_sha256=h, original_image_path=c['image_path'], cropping_path=c['cropping_path'],
            reservation_blocked=blocked, reservation_scope='all_patient_images' if blocked else 'none_detected',
            historical_splits=patient_records[p]['historical_splits'],
            historical_proposed_role=plans.get(p, {}).get('proposed_role', ''),
            experiment_partition='protected' if blocked else 'unassigned', partition_locked=False,
            zero_shot_disease_policy='pending_definition',
            technical_audit_pass=tech, technical_audit_scope='historical_snapshot_not_current_redecode',
            original_auto_screen=decision, original_auto_screen_processing_error=error,
            original_auto_screen_reasons=q.get('reasons', ''),
            final_training_crop_qc='pending', family_review='pending', near_duplicate_review='pending',
            image_has_mondo=r['image_has_mondo'], image_mondo_ids=r['image_mondo_ids'],
            disease_label_required_for_ssl=False,
            admission_status=status(blocked, tech, screen), blocking_reasons='|'.join(reasons),
            training_admitted=False,
        ))
    grouped = defaultdict(list)
    for r in ledger:
        grouped[r['mrn']].append(r)
    patients = []
    for p, rr in sorted(grouped.items()):
        assert len({r['subset'] for r in rr}) == 1
        first = rr[0]
        patients.append(dict(mrn=p, subset=first['subset'], candidate_images=len(rr),
                             sources='|'.join(sorted({r['source'] for r in rr})),
                             reservation_blocked=first['reservation_blocked'],
                             experiment_partition=first['experiment_partition'], partition_locked=False,
                             historical_splits=first['historical_splits'],
                             mondo_ids='|'.join(sorted({v for r in rr for v in r['image_mondo_ids'].split('|') if v})),
                             zero_shot_disease_policy='pending_definition',
                             family_review='pending', near_duplicate_review='pending',
                             training_admitted=False))
    aggregates = []
    for subset in ['1', '2', '3', '4']:
        rr = [r for r in ledger if r['subset'] == subset]
        counts = Counter(r['original_auto_screen'] for r in rr)
        aggregates.append(dict(subset=subset, patients=len({r['mrn'] for r in rr}), images=len(rr),
                               reserved_patients=len({r['mrn'] for r in rr if r['reservation_blocked']}),
                               reserved_images=sum(r['reservation_blocked'] for r in rr),
                               ssl_candidate_images=sum(not r['reservation_blocked'] for r in rr),
                               automatic_frontal_candidates=counts['frontal_candidate'],
                               automatic_nonfrontal_candidates=counts['nonfrontal_candidate'],
                               automatic_review=counts['review'],
                               not_screened_in_subset4_run=counts['not_screened_in_subset4_run'],
                               processing_errors=sum(r['original_auto_screen_processing_error'] for r in rr),
                               pending_final_crop_qc=sum(r['final_training_crop_qc'] == 'pending' for r in rr),
                               admitted_images=0))
    # A bounded calibration/review sample, not a prevalence sample or acceptance list.
    strata = defaultdict(list)
    for r in ledger:
        if not r['reservation_blocked']:
            key = (r['subset'], r['source'], 'processing_error' if r['original_auto_screen_processing_error'] else r['original_auto_screen'])
            strata[key].append(r)
    sample, seen_patients = [], set()
    for key, rr in sorted(strata.items()):
        n = 0
        for r in sorted(rr, key=lambda x: hashlib.sha256(('ssl_qc_review_v1:' + x['crop_sha256']).encode()).hexdigest()):
            if r['mrn'] in seen_patients:
                continue
            seen_patients.add(r['mrn'])
            sample.append(dict(mrn=r['mrn'], subset=r['subset'], source=r['source'],
                               screen_stratum=key[2], crop_sha256=r['crop_sha256'],
                               original_image_path=r['original_image_path'], cropping_path=r['cropping_path'],
                               human_original_view='', human_crop_quality='', identity_concern='',
                               reviewer='', reviewed_at='', review_notes='', training_admitted=False))
            n += 1
            if n == 30:
                break
    fields = ['mrn', 'source', 'crop_sha256', 'cropping_path', 'training_admitted']
    checks = dict(
        exact_image_coverage=len(ledger) == len({r['crop_sha256'] for r in ledger}) == 123982,
        exact_patient_coverage=len(patients) == 19448,
        subset_partitions=sum(r['images'] for r in aggregates) == len(ledger),
        reserved_subset_fully_blocked=all(r['reservation_blocked'] for r in ledger if r['subset'] == '1'),
        reservation_propagated_patientwide=all(len({r['reservation_blocked'] for r in rr}) == 1 for rr in grouped.values()),
        guarded_hashes_blocked=all(r['reservation_blocked'] for r in ledger if r['crop_sha256'] in all_reserved_hashes),
        no_unreviewed_admissions=not any(r['training_admitted'] for r in ledger),
        no_implicit_train_assignments=all(r['experiment_partition'] in {'unassigned', 'protected'} for r in ledger),
        qc_exact_subset4_coverage=set(qchash) == {r['crop_sha256'] for r in ledger if r['subset'] == '4'},
        qc_decisions_known=all(r['original_auto_screen'] in {'frontal_candidate', 'nonfrontal_candidate', 'review', 'not_screened_in_subset4_run'} for r in ledger),
        review_sample_no_reservations=not any(r['mrn'] in propagated_reserved for r in sample),
        review_sample_patient_unique=len(sample) == len({r['mrn'] for r in sample}),
        admitted_file_empty_by_design=True,
    )
    assert all(checks.values()), checks
    write('image_admission_ledger.tsv', ledger)
    write('patient_partition_ledger.tsv', patients)
    write('subset_summary.tsv', aggregates)
    write('qc_review_sample.tsv', sample)
    retry = [r for r in ledger if not r['reservation_blocked'] and r['original_auto_screen_processing_error']]
    write('screen_retry_queue.tsv', retry, list(ledger[0]))
    write('ssl_admitted_images.tsv', [], fields)
    summary = dict(compiled_at_utc=datetime.now(timezone.utc).isoformat(),
                   mode='preparation_only_not_training_release', total_patients=len(patients),
                   total_images=len(ledger), reserved_patients=len(propagated_reserved),
                   additional_reserved_patients_from_alias_audit=len(propagated_reserved - initially_reserved),
                   nonreserved_candidate_patients=sum(not r['reservation_blocked'] for r in patients),
                   nonreserved_candidate_images=sum(not r['reservation_blocked'] for r in ledger),
                   admitted_images=0, final_usable_image_count=None, subsets=aggregates,
                   status_image_counts=dict(Counter(r['admission_status'] for r in ledger)),
                   qc_review_sample_images=len(sample), screen_retry_queue_images=len(retry),
                   sample_method='deterministic SHA256 order; up to 30 distinct patients per subset/source/screen stratum; globally unique patients; not prevalence-weighted',
                   notes=['No training run or loader change.', 'No new patient split or zero-shot disease assignment.',
                          'No diagnosis label needed for SSL; reservation, QC and split checks still required.',
                          'Automated original-photo view screening is not approval of the stored crop.'])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'validation_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    versions = {name: dict(sha256=digest(ROOT / name), bytes=(ROOT / name).stat().st_size)
                for name in INPUTS + ['build_ssl_admission_inventory.py']}
    (OUT / 'input_versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    report(summary, aggregates, checks)
    print(json.dumps(summary, indent=2))


def report(s, rows, checks):
    table = '\n'.join('| ' + ' | '.join(str(r[k]) for k in [
        'subset', 'patients', 'images', 'reserved_images', 'automatic_frontal_candidates',
        'automatic_nonfrontal_candidates', 'automatic_review', 'not_screened_in_subset4_run']) + ' |' for r in rows)
    body = f'''# SSL 数据准入清单（准备版）

本清单落实已确认的范围：subset 1 全部保留；subset 2、3、4 仅作为 SSL 候选来源。原始四个 subset 保持不变，本次没有启动训练、修改加载器或释放患者。

## 当前数量

| Subset | 患者 | 图像 | 因保留规则排除的图像 | 自动正脸候选 | 自动非正脸候选 | 自动待审 | 未参加 subset4 全量筛查 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

非保留候选共 **{s['nonreserved_candidate_patients']:,} 人 / {s['nonreserved_candidate_images']:,} 张**。从完整来源清单的其他身份、原图路径、裁剪路径与哈希重新应用 expert/Ian 保护规则后，额外保留 **{s['additional_reserved_patients_from_alias_audit']} 人**。任何命中会传播到该患者全部图像。

**本次准入文件为 0 行，不表示可用图像为零。** 最终可用数量仍未知：患者实验划分、正脸/裁剪审核、家系与近重复检查尚未完成。所有 `training_admitted=False`。

## 准入规则

1. subset 1 与任何关联保留身份、路径、哈希命中的患者，全部排除 SSL。
2. subset 2、3、4 不要求疾病标签即可成为 SSL 候选；无 MONDO 不是排除原因。
3. 旧 train/val/ssl/test 字段仅记录历史暴露，不能自动变成新实验划分；expert 保留池内旧拟训练角色也不释放。
4. 将来划入开发、测试或疾病 zero-shot 留出的患者，其全部照片一起排除训练。目前没有自动选定零样本疾病或随机切分患者。
5. 技术可读性继承既有审计，不代表今天重新解码成功。正式训练前仍需检查实际输入。
6. 自动正脸候选仅是原图筛查结果；最终训练裁剪另行审核，尤其检查脸部完整性、旋转、清晰度与错误裁剪。自动非正脸结果也保留为待复核，不删除源图。
7. 病历、HPO、MONDO 标签都不作为本阶段 SSL 输入。标签仅用于队列统计和将来定义评估范围。PubMind 及派生资料仍全部排除。

## 当前可直接使用的工作清单

- `image_admission_ledger.tsv`：每张图的 subset、保留规则、历史划分、原图自动筛查、裁剪审核状态与阻塞原因。
- `patient_partition_ledger.tsv`：患者级待分配清单；保留患者标为 protected，其他患者标为 unassigned。
- `qc_review_sample.tsv`：**{s['qc_review_sample_images']} 张**按 subset、来源与自动筛查状态抽取的审核工作样本，每位患者最多一张。保留原图和裁剪路径，人工判断字段为空；并未完成审核。此样本不用于估计总体正脸比例。
- `screen_retry_queue.tsv`：**{s['screen_retry_queue_images']} 张**自动处理报错图像的重试队列；本次尚未运行重试。
- `ssl_admitted_images.tsv`：只有表头，明确防止将候选清单误当作已准入训练数据；并未连接到现有训练加载器。
- `subset_summary.tsv`、`summary.json`：汇总统计；`input_versions.json`：输入版本；`validation_checks.json`：一致性检查。

含患者标识与路径的 TSV 仅用于受控本地操作；对外使用本报告的汇总数据。重跑会重建这些派生文件；人工审核应另存正式结果，不直接在派生样本文件中维护唯一审核记录。

## 现有加载器的接入条件

本次只审阅了 `ssl_dataset.py`，未修改其行为。它已有 expert/Ian 过滤，但仍从旧标签清单读取图像，且不读取本次准入状态。正式接入之前必须改为仅消费已锁定的准入文件，不能直接运行旧训练入口来绕过审核。

现有实现还按 metadata 的同患者、同日期生成跨照片正样本，日期真实性尚未验证；首轮应先关闭这种配对，仅使用同一合格照片的增强视图。读取失败目前会生成黑图，正式训练应改为明确报错或记录并排除失败样本，不能静默替代。增强参数也需要检查是否过度裁掉颅面结构。本次并未启动这些训练改动。

## 下一步的顺序

先利用审核样本校准原图视图判断和最终裁剪要求，同时拟定患者级开发/测试划分与疾病留出策略。随后扩展 subset 2、3 的 QC、解决 subset 4 报错与不确定项，完成图像准入，才导出非空训练集。图像训练清单锁定后，采用患者均衡采样，避免照片多的患者主导学习。

本次完成 **{sum(checks.values())}/{len(checks)} 项检查**，包括全集覆盖、subset 分区、患者级保留传播、原图 QC 精确关联、无人误准入与审核样本保留隔离。运行 `python build_ssl_admission_inventory.py` 可重建；该脚本没有释放数据的功能。
'''
    (OUT / 'REPORT.md').write_text(body)


if __name__ == '__main__':
    main()

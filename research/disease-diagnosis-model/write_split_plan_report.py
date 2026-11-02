import csv,json
from pathlib import Path
from expert_reservation import filter_reserved
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'split_plan_v0'
s=json.loads((OUT/'summary.json').read_text())
with (OUT/'patient_split_proposal.tsv').open() as f:p=list(csv.DictReader(f,delimiter='\t'))
with (ROOT/'data_release_audit/chop_patient_consistent_candidates.tsv').open() as f:images=list(csv.DictReader(f,delimiter='\t'))
remaining,removed=filter_reserved(images)
roles=['train_candidate','dev_candidate','test_candidate','expert_test_reserve','label_identity_review']
rows=[]
for role in roles:
 rr=[r for r in p if r['proposed_role']==role and int(r['candidate_crops'])]
 source=sum(r['source_label_member']=='True' for r in rr)
 rows.append(f"| {role} | {source} | {len(rr)-source} | {len(rr)} | {sum(int(r['candidate_crops']) for r in rr)} |")
text='''# 患者级划分草案 v0

状态：仅规划，所有 training_admitted=False，没有释放训练患者，没有启动训练。
本文阈值是可调整的实验设计，不是期刊规定。专家 TSV 和 Ian 来源保持分开；
没有用 Ian 原始诊断文本自动推断 MONDO，也没有准入 PubMind 派生证据。

## 数量

本轮覆盖原 expert 表有保守图像的 946 人，以及 Ian 名单额外的 90 人，
去重合计 1,036 人。Ian 有图者 117 人中，27 人与 expert 表重合。
这不是新增了 90 人的可用数据，而是将原图像池里已有的 90 人纳入明确保留。

| 拟用途 | 来自 expert TSV | 额外 Ian/关联患者 | 合计患者 | 候选图 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

原始 expert 表的其余患者仍保留，没有因为暂时缺乏合格图像而从保护列表删除。
当前 SSL 实际保护仍包含所有 expert 源患者和 Ian 全名单及关联身份，
而非仅保留草案中的开发/测试患者。当前排除这些保留者后的保守池为
{len(remaining):,} 张、{len({r['mrn'] for r in remaining}):,} 人，全部仍待正脸 QC。

## 可复现分配规则

1. 所有 Ian 原始名单及已知身份/完全复制关联组优先保留为 expert_test_reserve。
   这不是最终测试认证；审核标准、标签映射、QC、历史暴露仍须核实。
2. 没有当前保守候选图片者标记 no_eligible_image。
3. 已知多身份关联、缺失/畸形 MONDO、诊断文本缺失、多个 MONDO、与 manual-fix
   不同者进入 label_identity_review。多诊断不被强行压成一个标签。
4. 剩余单标签患者按疾病分配：至少 20 人时约 70/15/15；5–19 人时约 80/20
   训练/开发，不单列小型测试；1–4 人先用于训练候选，不能据此评价该病种泛化。
   人数按保留 Ian 和已知问题后的可分配患者计算。
5. 固定 seed 20260928-v0。来源标签内按确定性哈希排序，再按来源比例交错分配，
   尽量平衡来源。实际分布在 source_allocation.tsv；不是来源独立外部验证。
6. 已知 MRN 别名、共用原图/裁剪路径和裁剪字节哈希用连通分组处理，组不跨用途。
   未获得可靠家系表，因此家系隔离未完成。近重复重编码图也不在当前保证内。

所有照片继承患者/关联组用途，即使某张照片未通过 QC。逐图草案包含全部已盘点
关联图像；表格的图像数仅来自更保守的候选池。日期没有用于时间外推划分。

## 最重要的实验边界

拟训练 666 人覆盖 99 个候选 MONDO ID，但常规拟测试的 117 人仅覆盖
22q deletion/duplication 两类。非 22q 只有 2 个拟开发患者。
所以该草案能支持首轮训练和 22q 初步诊断实验，不能证明 99 类疾病的泛化能力。

Ian 的 117 人全部先保留；90 人未被此 expert TSV 覆盖，因此需要利用独立原始
专家证据核实 MONDO。patient_split_proposal.tsv 保留 ian_diagnosis_raw 供审核。
这 90 人没有自动计入既有 MONDO 疾病覆盖。

每位拟评估患者都有 target_coverage_in_proposed_training 标记。只有训练中有
相应疾病，才能进入固定分类头的已见疾病评价。训练未见的疾病另用病例检索/
疾病知识匹配评估，不能把无法输出的类别硬塞进分类测试或事后删除来抬高分数。

建议下一轮优先审核非 22q 的训练候选和 Ian 保留者，建立真实的类别覆盖矩阵。
样本极少者可以在训练/开发池内探索交叉验证，但每折的全部患者图像都须重做
监督及自监督隔离；最终保留测试者不能参加这些轮次。

## 准入与历史暴露

- 所有候选诊断仍需独立临床来源/断言核实；genetest 推荐记录不是阳性结果。
- 图像视角/质量仍待审核，候选图数不是最终合格数量。
- 必须补核已知家系、近重复以及候选疾病定义后再锁定用途。
- 模型选择用开发集；最终测试不用于挑骨干、早停、阈值或背景处理方案。
- 旧全库 SSL 模型不能用于宣称本轮隔离评估独立；从公开基础权重开始。
- 草案没有自动宣称任何患者未经历史训练；historical_splits 原样保留供检查。

## 已实施的保护

expert_reservation.py 在原 expert 保留名单之外，额外读取本目录
ian_protected_mrns.tsv、ian_blocked_paths.tsv、ian_blocked_crop_hashes.tsv。
完整 Ian 名单有 {s['ian_source_patients']} 个 MRN，闭包包含 {s['ian_protected_ids']} 个 ID；
未发现照片的成员也保留。默认保护文件缺失时停止，原审计重跑不会覆盖 Ian 保护。
未来释放训练者要根据锁定版本生成新的排除规则，不能手工删除单个 MRN 绕过关联组。

## 文件

- patient_split_proposal.tsv：逐患者用途、来源、MONDO、原始 Ian 诊断、历史划分及待审项。
- image_split_proposal.tsv：所有已盘点关联图像继承用途，未准入训练。
- disease_allocation.tsv：逐疾病人数和用途；多标签待审患者会在多病种行出现，不能相加。
- stratum_allocation.tsv：22q、其他 MONDO、尚未解析三层数量。
- source_allocation.tsv：来源分布，供检查来源与用途是否混杂。
- review_priorities.tsv：优先非 22q 和映射问题；不是临床审核完成名单。
- input_versions.json / summary.json：输入版本与规则。

复现：python plan_expert_splits.py；python -m unittest test_split_plan.py test_expert_reservation.py；
python write_split_plan_report.py。保护规则生效，不需要也没有下载模型或提交训练作业。
'''
(OUT/'REPORT.md').write_text(text)
(OUT/'current_reservation_counts.json').write_text(json.dumps({'remaining_pending_qc_images':len(remaining),'remaining_patients':len({r['mrn'] for r in remaining}),'excluded_images':removed},indent=2)+'\n')
print('Report written;',len(remaining),'remaining pending-QC images')

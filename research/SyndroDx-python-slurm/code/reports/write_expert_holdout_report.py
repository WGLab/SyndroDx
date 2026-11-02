import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expert_holdout_audit'
s=json.loads((OUT/'summary.json').read_text())
with (OUT/'label_audit.tsv').open() as f:labels=list(csv.DictReader(f,delimiter='\t'))
with (OUT/'reserved_conservative_candidates.tsv').open() as f:ids={r['mrn'] for r in csv.DictReader(f,delimiter='\t')}
q22={r['mrn'] for r in labels if r['mrn'] in ids and set(r['mondo_ids'].split('|')) & {'MONDO:0018923','MONDO:0012020'}}
text=f'''# Expert CHOP 标签核查与患者保留

源文件：`/controlled-data/craniofacial_data/expert_chop_disease_mondo_labels.tsv`。
核查结论：值得优先保留的临床来源标签候选，但未完成逐例临床确认，不能整表直接认定为金标准或独立测试集。原始表及历史划分未修改。

## 核查结果

- {s['source_rows']:,} 行、{s['unique_source_mrns']:,} 个规范化 MRN，没有重复 MRN。
- 与 `CHOP_Diagnosis/expert_chop_disease_labels_combined_20260511.tsv` 字节完全一致，不是独立第二份确认。
- {s['rows_with_mondo']:,} 行可提取完整格式 MONDO ID，涉及 {s['unique_mondo_ids']:,} 个 ID；这些 ID 均存在于本地固定版本 MONDO，未发现已废弃 ID。词项有效不等于患者映射正确。
- {s['label_flags']['missing_mondo']} 行没有可提取的完整格式 ID，包含缺失或格式异常；33 行缺诊断文本；105 行有多个 ID。多 ID 可能为多诊断或待区分概念，未强制单标签。
- 4 行 MONDO 单元格格式异常，另有 1 行缺列导致 ID 落入日期列。没有静默修复源数据。
- 与后续 manual-fix 表比较，11 个患者的 MONDO 集合不同；没有自动选择后者。
- 813 行原始整段诊断不等于任一 MONDO 名称/EXACT 同义词，仅为人工检查线索，绝不能解释成 813 个错误。

## 来源证据

`data` 的来源标签可重叠：donna 1,954、imgc 1,578、genetest 513、manual 82。

- `CHOP_Diagnosis/Untitled.ipynb` 展示 Donna 22q 登记表与既有 expert 表的拼接、去重及映射逻辑。其 22q 映射分支将非 Duplication 默认归到 deletion；这里只是代码风险，未据此判定实际行错误。尚未逐例核对登记表及原始诊断证据。
- `Arcus_old/Processing.ipynb` 显示 imgc 来自 `imgc_diagnoses_20240716.xlsx`，manual 来自病历检查整理，genetest 读取遗传检测表并筛选 `Test Phase == Recommended`。methods / recommended_date 不可直接当成阳性检测结果；513 条带 genetest 来源的记录需核实最终结果，不能据此反过来断言全部诊断错误。
- 本次找到的是相关上游处理逻辑和当前表的等同副本，未找到并完整重放该文件所有行的生成链。文件没有统一的审核人、确认状态和审核日期字段。
- `Diagnosis_Prep.ipynb` 把 expert 与 PubMind 在下游合并；该下游混合表没有用来补标签。当前证据未证明本表混入 PubMind，但也不把未完成的来源追溯写成完全认证。
- 当前较宽图像候选池里，本表患者与 Ian 专家记录只重合 {s['overlap_ian_patients']} 人；这份表不等于 Ian 验证名单。

## 已实施隔离

- 全部 {s['unique_source_mrns']:,} 个源 MRN 都保留，即便当前没有可用照片或标签格式有问题。
- 沿已知 MRN 字段冲突、原图/裁剪路径和裁剪 SHA256 完全重复关联做保守闭包，额外隔离 {s['extra_identity_linked_mrns']} 个关联 ID，共 {s['protected_mrns']:,} 个保护 ID。关联仅用于隔离，不声称不同 ID 已确认为同一人。
- 图像盘点中保留 {s['reserved_image_rows']:,} 条记录：CF {s['reserved_by_source'].get('CHOP_craniofacial',0):,}、genetics {s['reserved_by_source'].get('CHOP_genetics',0):,}。
- 当前保守候选池中保留 **{s['reserved_conservative_patients']:,} 人、{s['reserved_conservative_crops']:,} 个不同裁剪**，仍待全量正脸 QC。
- 剩余 **{s['remaining_conservative_patients']:,} 人、{s['remaining_conservative_crops']:,} 个不同裁剪**，仅为待 QC 的 SSL 候选，不是已批准训练数据。
- 已知近重复、未知身份别名、家系关系、未进入盘点的其他副本尚不在闭包保证范围。

`ssl_dataset.py` 已在构造训练样本及同患者配对前，默认读取保留规则。MRN 规范化、已知别名、已知路径及清单自带哈希命中均排除。规则文件缺失时停止。现有 `ssl_train.py` 使用该加载器，因此未来重新启动会生效；已运行的作业不会追溯修改，其他自定义加载器必须显式使用这些规则。

该保护不等于全量正脸准入：旧 labels.tsv 的其余行仍可能存在 QC 问题，不应立即据此启动正式训练。

## 疾病分布与旧训练暴露

保留的保守候选涉及 {s['reserved_source_mondo_ids']} 个现有 MONDO ID，但只有 {s['reserved_source_diseases_at_least_5_patients']} 个 ID 至少 5 人。
22q deletion/duplication 合计 {len(q22)} 个不同患者，占 {len(q22)/len(ids):.1%}。
整体准确率容易被 22q 主导，必须同时报告疾病宏平均、非 22q 子集和逐病种样本数；不能依此宣称 122 种疾病都获得充分验证。

旧 labels.tsv 与保护集合交叠 {s['historical_label_rows_reserved']:,} 行、{s['historical_label_patients_reserved']} 个 MRN。
历史 split 行数：{json.dumps(s['historical_split_rows'])}。
这些是历史输入/划分重合，不是逐图训练日志证明。过去全库 SSL 的权重不能被这次隔离变成独立权重；新的评估需从未接触这批 CHOP 患者的基础权重重新开始。

## 待审核，不阻塞隔离

1. 先核对 4 个单元格格式异常、1 个缺列行、11 个后续版本差异。
2. 对实际有图像的 946 人核实诊断来源、断言和 MONDO 粒度，优先非 22q 病例。
3. 完成正脸 QC 后，再确定开发/校准/最终测试用途；目前整个保留池都不参与新训练。

## 文件和复现

- `protected_mrns.tsv`、`blocked_paths.tsv`、`blocked_crop_hashes.tsv`：训练排除规则。
- `reserved_conservative_candidates.tsv`：保留候选图像。
- `remaining_ssl_candidates_pending_qc.tsv`：排除后待 QC 的候选。
- `label_audit.tsv`、`label_review_queue.tsv`：逐行问题与证据状态。
- `input_versions.json`：源索引、本体及相关代码的大小、时间和 SHA256。
- `summary.json`、`guard_validation.json`：数量及排除检查。

运行 `python audit_expert_holdout.py`，随后 `python -m unittest test_expert_reservation.py` 和 `python write_expert_holdout_report.py`。
所有逐行清单包含内部患者标识，只供项目内部使用。本次未启动训练，未将任何标签自动提升为确诊真值。
'''
(OUT/'REPORT.md').write_text(text)
print('Report written; 22q unique patients:',len(q22))

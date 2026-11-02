import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'candidate_concept_audit'
s=json.loads((OUT/'summary.json').read_text())
with (OUT/'raw_label_concepts.tsv').open() as f:rr=list(csv.DictReader(f,delimiter='\t'))
translate={'disease_or_disorder':'疾病/临床障碍','disease_phenotype_overlap':'疾病与表型概念重叠','structural_diagnosis':'结构性诊断','phenotype':'表型/体征','procedure_or_workflow':'操作或流程（可能含混合描述）','injury_description':'损伤描述','nonspecific_finding':'非特异性发现','ambiguous_abbreviation':'歧义缩写','mixed_or_alternative':'混合/鉴别标签','missing':'空标签','unresolved':'未解析','gene_only':'仅基因名','specificity_requires_review':'概念过具体，待审','assertion_requires_review':'断言状态待审','workflow':'流程'}
rows=[f"| {translate[r['category']]} | {r['raw_labels']} | {r['remaining_patients']:,} | {r['remaining_unique_crops']:,} |" for r in s['categories']]
changed=[r for r in rr if r['legacy_mapping_changed']=='True']
text=f'''# 候选图像原始标签的 MONDO/HPO 分类与映射

状态：全标签覆盖的第一轮术语审计，不是已确认疾病真值。
范围：已盘点 CHOP craniofacial/genetics 正脸标签候选，共 {s['image_rows']:,} 条源记录，
{s['distinct_raw_strings']:,} 种原始字符串（非空 {s['nonempty_raw_strings']:,} 种）。
未声称覆盖所有 MedicalDBs 病历或所有非正脸目录；没有使用 PubMind、其派生标签或
旧 patient_groups 推断疾病。

MONDO 与 HPO 均使用本地 2026-09-01 版本，源文件、规则代码与保护清单 SHA256
记录于 input_versions.json。本轮不修改原始标签、旧映射、划分或保留患者名单。

## 结果

{s['lexically_mapped_raw_labels']} 种原始写法完成词汇层 MONDO 或 HPO 映射，包含
本体名称/EXACT 同义词和显式别名候选。别名仍需领域审核；该状态不是临床确认。
其余按操作、缺失、混合、缩写和未解析等原因保留。不能为了提高映射率强行选 ID。

排除 expert/Ian 后的候选池有 {s['remaining_images']:,} 个不同裁剪、
{s['remaining_patients']:,} 个 MRN，分类覆盖如下：

| 原始标签类别 | 写法数（全部源记录） | 剩余池关联患者 | 剩余池关联不同裁剪 |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

患者/图片可关联多个原始标签，各类别人数和图数不可相加。比如“空标签”患者也可能
在另一路径拥有诊断标签，不等于这些患者完全没有诊断。
同一个 SHA256 的不同路径可能带不同标签，本轮保留全部通过身份筛查的原始标签
证据，不因图片去重选了一个代表路径而丢失其他标签。

剩余池关联 {s['remaining_mondo_ids']} 个 MONDO 概念，其中 disease_or_disorder
类别涉及 {s['remaining_disease_or_disorder_ids']} 个 ID。其余包含结构性异常、疾病/HPO
重叠及损伤；**这些都不是已临床确认的疾病类别数**。普通 disease_or_disorder 也不
等于病因性综合征，可能包含皮肤病、良性病变或广义疾病组。
所有 weak_supervision_admitted=False，确诊疾病数保持 null。

## 本轮重点纠正

- UCLP/BCLP：可映射较宽的裂唇/腭裂概念，保留原文侧别；单独归为结构性诊断，
  不自动视为遗传综合征。MONDO 目标不保留原始侧别，所以标记 broader_concept。
- CP、CL、HFM、NF：先保留候选，需上下文确认；22q 11.2 未说明 deletion 或
  duplication 时也不自动选其中一个。
- hemihypertrophy：保留 HPO 表型，不推断 Beckwith–Wiedemann 或 isolated 状态。
- scaphocephaly / trigonocephaly / metopic ridge：头形描述不证明颅缝融合；明确
  sagittal syn 等才映射广义 craniosynostosis，并保留缝型 HPO。
- scar：防止小写瘢痕被本体的 SCAR 缩写匹配成严重皮肤不良反应。
- gynecomastia / macromastia：保留临床表型/障碍，不能沿用旧脚本的操作分类。
- FGFR3：基因名不能直接变为 Muenke 综合征。
- apert/pfeiffer 等混合/鉴别目录：拆分候选概念，暂不确认同时患病。
- CP-PR 等复合缩写：分别保留解释，不将整串作为一个疾病。
- 仅有 BROAD/NARROW 本体同义词不做 exact 自动匹配；遇到过具体的既有候选，
  例如 inherited、familial、isolated 等限定，保持待审。

相对于旧 raw_label_mondo_map.tsv，{len(changed)} 个原本已映射标签被撤回单一 MONDO
选择；旧文件未覆盖。更早 taxonomy 中的问题与旧 raw crosswalk 差异不是同一统计。

## 下一步如何使用

1. 按 terminology_review_queue.tsv 的剩余患者覆盖顺序审核高频歧义和别名。
2. 决定两个监督目标层：病因性疾病/综合征与结构性异常；疾病/HPO 重叠者单审。
   不因本体含 MONDO ID 就自动纳入主疾病分类头。
3. 再基于这些概念重新核查 cross-folder 一致性，区分相同、父子概念和可能并存。
   当前历史 A/B 等级仅保留为 provenance，没有升级成临床真值或重新计算等级。
4. 最后核对患者自身的诊断断言、时间与来源，并完成正脸 QC，才准入弱监督。

genetics 原始字段上游仍待追溯，映射只是在解释给定文字，绝不认证其来源或准确性。
没有外推诊断；没有用这些标签进行训练、自动伪标注或解除保留患者隔离。

## 产物

- raw_label_concepts.tsv：全标签分类、MONDO/HPO、关系、依据和覆盖数。
- concept_components.tsv：混合标签逐成分解释、候选与断言边界。
- patient_label_assertions.tsv：每位患者关联哪些原始标签，保留来源和旧 A/B 等级。
- remaining_mondo_concept_counts.tsv：扣除保留患者后的词汇层概念人数。
- terminology_review_queue.tsv：优先按剩余池患者覆盖排序的术语审核清单。
- category_counts.tsv、summary.json、input_versions.json：汇总与复现信息。

复现：python normalize_candidate_concepts.py；python -m unittest test_candidate_concepts.py；
python write_candidate_concept_report.py。
逐患者文件含内部 MRN，仅供本地研究审核；概念汇总没有患者姓名。
'''
(OUT/'REPORT.md').write_text(text)
print('Wrote concept report')

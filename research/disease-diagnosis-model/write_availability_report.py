import json
from pathlib import Path
OUT=Path(__file__).resolve().parent/'data_release_audit'
s=json.loads((OUT/'summary.json').read_text());e=s['evidence_in_identity_screened_CHOP'];rows=[]
for name,v in s['by_source'].items():
 rows.append(f"| {name} | {v['candidate_rows']:,} | {v['technical_rows']:,} | {v['technical_exact_unique_crops']:,} | {v['identity_screened_patients']:,}"+(' |' if name.startswith('CHOP') else '（不适用 CHOP 筛查） |'))
text='''# 数据可用性核对结果

本轮完成源索引整理、逐路径文件检查、裁剪解码及 SHA256 去重、患者身份
冲突筛查、临床/表型/人口学证据关联。没有修改原始数据或旧训练划分。
PubMind 和已知派生结果排除，全疾病范围保留。

**“文件可用、身份无已发现冲突的候选”不等于“已核实正脸和确诊的病例”。**
特别是本轮视觉抽样发现明确的误分类，最终严格正脸数量仍待全量 QC。

| 来源 | 正脸标签候选行 | 原图存在且裁剪可解码 | 可读裁剪的不同 SHA256 | 筛查后 CHOP 患者 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

CHOP 在记录级身份筛查、完全重复处理和已知抽样问题隔离后，保留
**{e['unique_crops']:,} 个不同裁剪文件、{e['patients']:,} 个 MRN** 作为待进一步
视图与诊断复核的候选。这个数是当前已索引候选池的结果，不是最终训练规模。
患者跨来源有重合，不能将两类 CHOP 患者数相加。

## 候选队列的证据覆盖

| 已有关联 | 患者数 | 当前边界 |
|---|---:|---|
| 结构化患者表 | {e['patient_bridge']:,} | MRN 关联，不证明每张照片身份正确 |
| 就诊诊断记录 | {e['clinical_dx_records']:,} | 未裁决本人/确诊状态及时间 |
| cTAKES HPO 候选 | {e['ctakes_hpo_candidate_patients']:,} | 否定、家族史、重复及时间待复核 |
| PheNominal HPO 候选 | {e['phenominal_hpo_candidate_patients']:,} | 来源及断言待复核 |
| 原文正文 | {e['raw_note_text_patients']:,} | 查询/选择来源仍待追溯，未作为预测输入准入 |
| HPO 候选＋出生日期字段＋性别编码 | {e['image_hpo_birth_date_sex_field_patients']:,} | 只是字段可关联；拍照年龄、编码有效性和表型真值未核实 |
| 既有 MONDO ID | {e['existing_mondo_patients']:,} | 涉及 {e['existing_mondo_ids']:,} 个候选 ID，未等同确诊疾病数 |
| Ian 专家记录 | {e['expert_review_record_patients']:,} | {e['expert_record_unique_crops']:,} 个不同候选裁剪；复核标准、视图和旧训练暴露待核 |
| 变异记录 | {e['variant_record_patients']:,} | HGVS 非空 {e['hgvs_record_patients']:,} 人；临床致病性字段非空 {e['variant_significance_record_patients']:,} 人 |
| crossfolder A/B | {e['crossfolder_A_or_B_patients']:,} | 目录标签一致性，不是独立临床确认 |

以上项目相交，不能相加。既有 MONDO 来源之间不一致的患者有
{e['existing_mondo_source_disagreement_patients']:,} 人；差异可能涉及错误、
多诊断或概念粒度，未自动选取“正确版本”。候选患者中有
{e['other_image_identity_conflict_patients']:,} 人在其他图像记录上仍有身份冲突，
因此记录级筛查不替代患者级身份裁决。

## 已隔离的问题

`image_issue_queue.tsv` 共 {s['review_queue_rows']:,} 条 CHOP 记录，包含文件问题、
MRN 冲突/缺失、无法对应患者表的非数字 ID、跨 ID 完全重复和视觉抽样问题。
这些原因可能重叠；源记录没有删除，仍可修复后重新审核。

详细的文件状态、身份状态及重复数量见 `summary.json` 的 by_source。
完全重复判断仅针对文件字节，不涵盖重新压缩、不同裁剪、近重复及家系。

## 正脸标签不能直接当作通过 QC

随机抽取 32 张 CF、16 张 genetics 候选图，助手缩略图初筛结果为：

- CF：14 张明显不符，3 张需进一步检查，15 张近似正脸候选。
- genetics：3 张明显不符，2 张需进一步检查，11 张近似正脸候选。

样本 1、8、31 的原图也分别显示侧向、仰拍和耳朵局部，问题并非裁剪显示
造成。抽样状态已写回候选清单，并传播到字节相同的文件；明显不符和存疑
记录暂不进入去重候选清单。其余未查看图像仍标记 pending。

这是小样本分层抽查，不能据此估算全队列合格比例，也不是专家临床审核。
近似正脸候选中仍有闭眼、光照等质量问题。非正脸 metadata 类别也可能有
假阴性，尚未全量重新筛查。

## 下一步先解决什么

1. **视图 QC**：制定严格正脸和可修复旋转的判定规则，先校准筛选器，再对
   全候选池批量筛查并复核边界样本；同步评估非正脸标签中的漏检。
2. **身份裁决**：优先修复有临床或专家证据的冲突记录，再做跨路径近重复
   与跨数据集患者重合检查。
3. **诊断/表型证据**：核对专家来源、原文断言和时间，重建患者级 MONDO
   真值；不能直接把目录术语映射或病历中的代码当作确诊。
4. **锁定验证队列**：保留原训练暴露记录，决定专家集用于开发还是最终测试，
   对新的独立评估排除患者及其所有图像的训练/SSL暴露。

最终合格正脸数、临床裁决后的疾病数、独立测试患者数在 summary.json 中
保持 null（待确认），没有编造成已完成的计数。

## 可直接审核的产物

- `chop_unique_candidates.tsv`：筛查、完全去重后的逐图候选。
- `patient_manifest.tsv`：候选患者、图像数、临床/表型/人口学及专家证据。
- `image_issue_queue.tsv`：可修复问题清单。
- `existing_mondo_candidate_counts.tsv`：已有 MONDO 的候选人数。
- `folder_mondo_candidate_counts.tsv`：目录术语映射人数，单独存放。
- `view_qc_sample.tsv`：逐图抽样判定与原因。
- `README.md`：检查定义、限制及复现步骤。
'''
if (OUT/'full_genetics_path_audit.json').exists():
 g=json.loads((OUT/'full_genetics_path_audit.json').read_text());text+='\n## 完整 genetics 索引\n\n'
 for c in ['file_path','modified_file_path']:
  z=g[c];text+=f"- {c}：{z['unique_indexed_paths']:,} 个路径，检查结果 {z['status_counts']}。\n"
 text+='\n仅在已知图像挂载根目录解析，未以相同文件名猜测其他位置，也未将复制的\ntop10 图像加算为新患者。完整索引的患者数仍不等于新增可用正脸患者数。\n'
q=s['patient_consistent_subset']
section=f'''## 更保守的患者级候选池

进一步暂时排除在任一已知图像记录中出现身份冲突的患者，得到
**{q['unique_crops']:,} 个不同裁剪文件、{q['patients']:,} 个 MRN**。
共 {q['patients_flagged_in_other_images']:,} 个候选患者等待身份问题解决，未永久删除。

- 就诊诊断记录：{q['clinical_dx_patients']:,} 人。
- HPO 候选：{q['hpo_candidate_patients']:,} 人；同时有出生日期字段及性别编码：{q['hpo_birth_date_sex_fields_patients']:,} 人。
- 原文正文：{q['raw_note_text_patients']:,} 人。
- 既有 MONDO：{q['existing_mondo_patients']:,} 人、{q['existing_mondo_ids']:,} 个候选 ID。
- Ian 专家记录：{q['expert_record_patients']:,} 人、{q['expert_record_unique_crops']:,} 个候选裁剪。
- 变异记录：{q['variant_patients']:,} 人。

建议从此池开始后续 QC。以上仍非已核实正脸、确诊病例或独立测试集。
逐图清单为 `chop_patient_consistent_candidates.tsv`。
下表使用较宽的记录级候选池作为分母。

'''
text=text.replace('## 候选队列的证据覆盖',section+'## 候选队列的证据覆盖')
(OUT/'AVAILABILITY_REPORT.md').write_text(text)
print('Wrote AVAILABILITY_REPORT.md')

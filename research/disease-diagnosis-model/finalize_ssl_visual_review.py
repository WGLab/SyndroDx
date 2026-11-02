"""Publish assistant first-pass visual observations and original-only work queues."""
import csv,json,hashlib
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'ssl_visual_review'
def read(p):
 with (ROOT/p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows,fields=None):
 with (OUT/p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=read('ssl_visual_review/review_manifest.tsv')
 observations={}
 for line in (OUT/'assistant_observations.txt').read_text().splitlines():
  number,code,note=line.split('|',2);rid='R'+number
  assert rid not in observations
  observations[rid]=(code,note)
 assert set(observations)=={r['review_id'] for r in manifest}
 meanings={'P':'source_candidate_rebuild_crop','A':'source_orientation_or_crop_adjustment','E':'exclude_image_from_frontal_task','R':'second_review_required'}
 rows=[]
 for r in manifest:
  code,note=observations[r['review_id']]
  # Earlier pair judgments concerned legacy crop issues. Under original-only
  # policy, these two originals can enter crop reconstruction review.
  if r['review_id'] in {'R028','R039'}:
   code='P';note+=' 原图可进入重建候选；此前问题针对旧去背景裁剪，不批准旧版本。'
  rows.append(dict(review_id=r['review_id'],mrn=r['mrn'],subset=r['subset'],source=r['source'],
                   crop_sha256=r['crop_sha256'],original_image_path=r['original_image_path'],
                   legacy_crop_path=r['cropping_path'],screen_stratum=r['screen_stratum'],
                   reviewer='assistant_visual_first_pass',review_method='original_and_legacy_crop_contact_sheet',
                   source_triage=meanings[code],observation=note,
                   lineage_identity_pending=r['lineage_identity_pending'],
                   identity_verified=False,legacy_input_allowed=False,
                   required_input='rectangular_crop_from_verified_original_with_real_background',
                   new_crop_created=False,new_crop_qc='pending',clinical_adjudication=False,training_admitted=False))
 write('assistant_visual_review.tsv',rows)
 index={r['crop_sha256']:r for r in rows}
 clean=read('ssl_image_cleaning/deduplicated_ssl_candidates.tsv')
 queue=[]
 for r in clean:
  review=index.get(r['crop_sha256'])
  triage=review['source_triage'] if review else 'not_visually_reviewed'
  action={'source_candidate_rebuild_crop':'priority_rebuild_from_original',
          'source_orientation_or_crop_adjustment':'orientation_review_then_rebuild',
          'exclude_image_from_frontal_task':'hold_out_nonfrontal_or_incomplete_image',
          'second_review_required':'review_original_before_rebuild',
          'not_visually_reviewed':'source_view_qc_pending'}[triage]
  queue.append(dict(mrn=r['mrn'],subset=r['subset'],source=r['source'],photo_group_id=r['photo_group_id'],
                    lineage_crop_sha256=r['crop_sha256'],original_image_path=r['original_image_path'],
                    legacy_crop_path=r['cropping_path'],legacy_input_allowed=False,
                    source_triage=triage,next_action=action,required_processing='background_preserving_rectangular_crop',
                    verified_original_provenance=False,new_crop_path='',new_crop_qc='pending',training_admitted=False))
 write('original_only_rebuild_ledger.tsv',queue)
 priority=[r for r in queue if r['next_action']=='priority_rebuild_from_original']
 write('priority_original_rebuild_queue.tsv',priority,list(queue[0]))
 counts=Counter(r['source_triage'] for r in rows)
 strata=defaultdict(Counter)
 for r in rows:strata[(r['subset'],r['screen_stratum'])][r['source_triage']]+=1
 write('review_by_subset_and_screen.tsv',[dict(subset=k[0],screen_stratum=k[1],**{name:c[name] for name in meanings.values()}) for k,c in sorted(strata.items())])
 checks=dict(all_290_reviewed=len(rows)==290 and len({r['review_id'] for r in rows})==290,
             all_legacy_inputs_blocked=all(not r['legacy_input_allowed'] for r in rows+queue),
             no_identity_inference=all(not r['identity_verified'] for r in rows),
             no_training_release=all(not r['training_admitted'] for r in rows+queue),
             complete_cleaned_cohort_coverage=len(queue)==len(clean)==118456,
             no_new_crop_claim=all(not r['new_crop_path'] for r in queue),
             excluded_images_not_prioritized=all(r['source_triage']=='source_candidate_rebuild_crop' for r in priority),
             protected_subset_absent=all(r['subset']!='1' for r in queue))
 assert all(checks.values()),checks
 summary=dict(review_date_local='2026-10-11',compiled_at_utc=datetime.now(timezone.utc).isoformat(),
              reviewed_pairs=len(rows),reviewer='assistant_visual_first_pass_not_clinician_consensus',
              source_triage_counts=dict(counts),original_only_ledger_images=len(queue),
              priority_original_rebuild_images=len(priority),lineage_pending_in_review=sum(r['lineage_identity_pending']=='True' for r in rows),
              new_crops_created=0,training_admissions=0,
              policy='No background-removed or background-replaced inputs; reconstruct rectangular crops from verified originals; no synthetic background restoration',
              limitations=['Contact-sheet first pass, not native-resolution assessment for every image.',
                           'No patient identity or diagnosis inferred from appearance.',
                           'Nonrandom stratified sample; do not extrapolate QC prevalence.',
                           'Borderline poses/expressions/occlusions require second review; no calibrated angle threshold.',
                           'All previously admitted counts were zero and remain zero. Existing loader has not been changed.'])
 for name,obj in [('summary.json',summary),('validation_checks.json',checks)]:
  (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
 inputs=['DATA_SCOPE.md','ssl_visual_review/assistant_observations.txt','ssl_visual_review/review_manifest.tsv',
         'ssl_image_cleaning/deduplicated_ssl_candidates.tsv','prepare_ssl_visual_review.py','finalize_ssl_visual_review.py']
 versions={p:digest(ROOT/p) for p in inputs}
 versions.update({str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.glob('contact_*.jpg'))})
 (OUT/'input_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
 (OUT/'REPORT.md').write_text(f'''# 290 张样本首轮目视审核与真实背景输入规则

已逐张查看 290 组原始照片/旧裁剪，共 15 页对照图。审核者为 AI assistant，属于首轮目视筛查，不是临床专家一致性裁决；原图身份和诊断不由外貌推断。每张都有编号、结论与原因。

## 新输入规则

按用户要求，后续训练、开发和测试只使用可追溯原始照片的普通矩形裁剪，保留裁剪内真实背景。去背景、背景替换及其缩放派生版本均不准入；不会生成或补画背景。自然拍摄的白色背景并不等于去背景，判断依据是来源和处理历史。

现有旧裁剪仅用于审计对照，**全部暂停作为新实验输入**。旧裁剪上的“通过候选”不构成使用授权；可用原图要重新生成新裁剪并审核。缺少或来源不明的原图进入恢复/待审。本轮新生成训练裁剪 0 张，训练准入 0 张。

## 原图首轮结果

| 分类 | 数量 | 后续动作 |
| --- | ---: | --- |
| 原图可进入裁剪重建候选 | {counts['source_candidate_rebuild_crop']} | 从原图定位人脸并做保留背景矩形裁剪，然后审核 |
| 需方向或裁剪调整 | {counts['source_orientation_or_crop_adjustment']} | 校正方向后复核，绝不通过变形把侧脸变正脸 |
| 不符合本次完整正脸范围 | {counts['exclude_image_from_frontal_task']} | 本任务隔离该图，源文件保留 |
| 需第二次审核 | {counts['second_review_required']} | 放大检查姿态、清晰度、遮挡和表情 |

常见问题包括侧面、明显仰头、手足或口腔局部、文档/手术图、横向旋转、闭眼/大幅表情、手或器具遮挡、原图缺失部分面部以及旧去背景处理的轮廓残留。面部异常、不对称或临床器具本身不自动等于排除；需区分视角与真实形态，以及器具是否遮挡关键区域。

对 R028、R039 的最初记录涉及旧裁剪的边界/分割问题；新政策下把其原图归为重建候选，旧裁剪仍不准入。其余边界判断保守进入复核。脸部表情和遮挡准入仍需固定标准，不能将这里的筛查当成最终 QC 标签。

原图可重建类别中，与清理后非保留、无已知身份待审候选清单交集为 **{len(priority)} 张**，已形成优先重建队列。完整原图处理台账覆盖 **{len(queue):,} 条候选**，其中未目视检查的行明确标为待审，没有把 290 张结果外推到全库。

## 文件

- [逐张审核结果](assistant_visual_review.tsv)：记录来源、原图/旧裁剪、目视结论与原因。包含受控患者标识，不用于对外发布。
- [原图重建优先队列](priority_original_rebuild_queue.tsv)：首轮候选与身份/保留筛查的交集，尚无新裁剪文件。
- [全候选真实背景处理台账](original_only_rebuild_ledger.tsv)：旧输入不允许、新输入路径空、原图溯源和新裁剪 QC 待审。
- [按 subset 与原自动筛查状态统计](review_by_subset_and_screen.tsv)。该样本分层抽取且每人一张，不能用于估计全库正脸比例或自动模型准确率。
- [汇总](summary.json)、[检查结果](validation_checks.json)、[输入及对照图版本](input_versions.json)。

## 下一步

先从优先队列重建保留背景的人脸矩形裁剪，保持比例、保留额头/下巴/侧面轮廓并限制多余背景；方向调整、边界夹断和目标不明确的单列复核。审核新裁剪后再扩大处理。不要恢复旧去背景文件，也不要直接启动仍读取旧裁剪的训练加载器。

当前目视依据是对照图缩略展示，并未逐张原尺寸确认清晰度；“候选”仍需新裁剪后的原尺寸审核。已有像素检查仅证明所抽样文件能解码和哈希一致。没有识别人脸身份、诊断疾病或确认最终训练资格。本轮 {sum(checks.values())}/{len(checks)} 项一致性检查通过，未修改旧训练加载器或删除任何源文件。
''')
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

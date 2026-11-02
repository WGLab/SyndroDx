"""Versioned view-scope reinterpretation of already inspected image pairs."""
import csv,json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'multiview_review_v1';OUT.mkdir(exist_ok=True)
# Manual view annotations based on the previously inspected original/pair sheets.
profile={33,35,43,47,76,78,84,125,127,129,130,134,140,169,245,247,248,249,251,254,256,257}
oblique={5,12,14,29,56,67,82,83,123,126,128,131,132,136,137,139,146,150,153,160,167,176,180,202,211,212,243,244,252,253,255,258,276,288}
nonface={48,59,79,80,142,161,189,196,204,209,250,277}
incomplete={179,187,210}
rows=list(csv.DictReader((ROOT/'ssl_visual_review/assistant_visual_review.tsv').open(),delimiter='\t'))
clean={r['crop_sha256'] for r in csv.DictReader((ROOT/'ssl_image_cleaning/deduplicated_ssl_candidates.tsv').open(),delimiter='\t')}
out=[]
for r in rows:
 i=int(r['review_id'][1:]);prior=r['source_triage']
 if i in profile:view='profile_candidate'
 elif i in oblique:view='oblique_candidate'
 elif i in nonface:view='non_face_or_local_body_document'
 elif i in incomplete:view='incomplete_face_original'
 elif prior=='exclude_image_from_frontal_task':
  assert '仰' in r['observation'],r['review_id']
  view='strong_pitch_candidate'
 elif prior=='source_orientation_or_crop_adjustment':view='orientation_or_crop_adjustment_pending'
 elif prior=='source_candidate_rebuild_crop':view='frontal_candidate'
 else:view='view_or_quality_borderline_review'
 action='keep_for_view_specific_quality_review' if view in {'profile_candidate','oblique_candidate'} else 'rebuild_then_review' if view=='frontal_candidate' else 'exclude_from_complete_face_task' if view in {'non_face_or_local_body_document','incomplete_face_original'} else 'special_view_or_quality_review'
 out.append(dict(review_id=r['review_id'],mrn=r['mrn'],subset=r['subset'],source=r['source'],
                 original_image_path=r['original_image_path'],lineage_crop_sha256=r['crop_sha256'],
                 prior_frontal_only_decision=prior,view_label=view,multiview_action=action,
                 view_label_basis='assistant_manual_reinterpretation_of_inspected_original_contact_sheets',
                 source_observation=r['observation'],patient_passes_current_identity_reservation_filter=r['crop_sha256'] in clean,
                 view_specific_quality='pending',legacy_background_removed_input_allowed=False,training_admitted=False))
def write(name,rr):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]),delimiter='\t');w.writeheader();w.writerows(rr)
write('view_review.tsv',out)
side=[r for r in out if r['view_label'] in {'profile_candidate','oblique_candidate'}]
write('side_oblique_reassessment_queue.tsv',side)
s=dict(reviewed_originals=len(out),view_counts=dict(Counter(r['view_label'] for r in out)),
       side_oblique_reconsidered=len(side),side_oblique_patient_filter_pass=sum(r['patient_passes_current_identity_reservation_filter'] for r in side),
       prior_exclusion_is_task_specific=True,training_admissions=0,
       source_review_sha256=hashlib.sha256((ROOT/'ssl_visual_review/assistant_visual_review.tsv').read_bytes()).hexdigest(),
       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
assert len(out)==290 and len({r['review_id'] for r in out})==290
assert not any(r['training_admitted'] for r in out)
(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
(OUT/'REPORT.md').write_text(f'''# 多视角审核口径修订

用户明确提出不能只保留正脸。本次保留正脸、斜侧脸、侧脸的独立视角标记，图像质量与视角分开判断，不再把非正脸整体视为无效数据。

根据已逐张查看的 290 组原图/旧裁剪，对旧的任务限定结论重新标注：

| 视角/状态 | 样本数 |
| --- | ---: |
'''+ '\n'.join(f'| {k} | {v} |' for k,v in s['view_counts'].items())+f'''

其中 **{len(side)} 张侧脸/斜侧脸**从旧“非正脸”队列转为多视角质量复核候选。该数字不是新的合格训练图数量；仍需核对完整轮廓、鼻尖/口唇/下巴是否被截断、遮挡、清晰度和原图来源。保留视角标签是基于已查看的原图对照，不是精确角度测量；边界视角待复核。

侧脸不能套用“两眼均可见”作为完整性条件。侧向方向如需记录，应写图像坐标方向；镜像或解剖左右未核实时不推断。明显仰头/俯视单列特殊视角，不自动并入侧脸。旋转只纠正照片方向，不做正脸化、对称化或几何变形。

## 训练使用建议

先比较单张图的自监督表征，并保留 view 标签做按视角的评估。采样以患者为单位，避免照片多或视角多的患者主导。后续可比较正脸训练、正脸+侧脸训练和视角分层采样的效果。日期/拍摄事件尚未核实时，不把同患者不同视角自动组成正样本对；同患者所有视角必须共享实验划分。

## 范围与输出

- [逐图多视角审核](view_review.tsv)
- [侧脸与斜侧脸复核队列](side_oblique_reassessment_queue.tsv)
- [汇总](summary.json)

历史正脸候选池包含上述视角混入，故这 290 张不是全库多视角分布样本。完整 craniofacial metadata 中旧标为侧脸的 106,029 行尚未在本轮全量审核或恢复；不能简单加到现有候选图像数量上。新多视角队列还需要从全库索引抽样校准，再进行身份/重复/保留隔离。

所有旧去背景输入继续不准入，expert/Ian 保留和身份待审不变。所有训练准入为 False。旧 `exclude_image_from_frontal_task` 保留为历史任务结论，本报告是新的视角范围解释，不覆写其原始审核记录。
''')
print(json.dumps(s,indent=2))

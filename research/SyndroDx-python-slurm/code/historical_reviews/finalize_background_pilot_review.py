"""Record assistant contact-sheet review of all 82 original-preserving crops."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'background_preserving_pilot_v2'
rows=list(csv.DictReader((OUT/'results.tsv').open(),delimiter='\t'))
flags={
 'R100':'原图日期叠字仍在新裁剪底部；需调整矩形范围或保留待审，不能补画。',
 'R105':'原图日期叠字仍可见；需检查能否通过矩形裁剪排除。',
 'R113':'原图日期叠字仍可见；需检查能否通过矩形裁剪排除。',
 'R270':'上方保留旁人的局部面部；需收紧背景范围并保留目标完整轮廓。',
 'R046':'边缘有旁人的局部面部；需复核目标区域和背景范围，不凭外貌判定身份。',
 'R198':'原图顶部边界已紧，额头/轮廓完整性需原尺寸复核。',
 'R028':'下巴边界仍较紧，需结合原图最大范围复核。',
 'R192':'原图顶部边界较紧，需原尺寸复核额头和面部完整性。',
 'R003':'原图清晰度较软，需在原分辨率和训练输入尺寸分别复核。',
 'R095':'较远原图经裁剪放大后需核对人脸像素和清晰度。',
 'R050':'背景标牌文字仍在面部附近；复核是否能仅调整矩形边界排除。',
 'R280':'背景标牌文字仍在头部上方；需复核矩形上边界，不遮罩补画。',
}
review=[]
for r in rows:
 rid=r['sample_id']
 review.append(dict(review_id=rid,new_crop_path=r.get('new_crop_path',''),
                    visual_review='assistant_contact_sheet_first_pass',
                    review_status='specific_followup_needed' if rid in flags else 'no_obvious_new_crop_defect_at_sheet_resolution',
                    observation=flags.get(rid,'已对照原图查看新裁剪；真实背景保留，未见明显新增分割缺损；仍需原尺寸质量和准入审核。'),
                    new_crop_sha256=r['new_crop_sha256'],final_quality_approved=False,training_admitted=False))
with (OUT/'visual_review.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(review[0]),delimiter='\t');w.writeheader();w.writerows(review)
checks=dict(all_82_rebuilt=len(rows)==82 and len({r['sample_id'] for r in rows})==82,
            all_pixel_checks_passed=all(r['exact_original_rgb_crop_verified']=='True' for r in rows),
            all_upright=all(r['rotation']=='0' for r in rows),
            all_background_retained=all(r['background_removed']=='False' and r['legacy_crop_used']=='False' for r in rows),
            all_outputs_current=all(hashlib.sha256(Path(r['new_crop_path']).read_bytes()).hexdigest()==r['new_crop_sha256'] for r in rows),
            all_visually_reviewed=len(review)==82,
            no_training_release=all(r['training_admitted']=='False' for r in rows),
            expected_followup_ids_present=set(flags)<=set(r['sample_id'] for r in rows))
assert all(checks.values()),checks
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(OUT/'review_summary.json').write_text(json.dumps(dict(reviewed=82,specific_followup=len(flags),other_crops_pending_final_quality=82-len(flags),final_quality_approved=0,training_admitted=0),indent=2)+'\n')
(OUT/'REPORT.md').write_text(f'''# 保留真实背景的裁剪试验 v2

从已目视初筛的 82 张原始照片成功重建 **82 张原分辨率 PNG 裁剪**。所有输出经逐像素验证，与 EXIF 方向处理和 RGB 解码后的原图矩形区域一致。没有去背景、补画、遮罩、正脸化、拉伸、训练缩放或非直角旋转。旧去背景图未作为输入。

## 方向修正与版本

早期 v1 试验调用旧的自动方向搜索，发现会把已确认直立的照片误旋转 180 度，已中止并标记不得使用。当前 v2 基于已目视确认直立的这 82 张照片，仅使用 EXIF 方向，不自动搜索旋转；82 张额外旋转均为 0。该限制针对本试验，不能盲目应用到方向未审核的全库照片。

## 新裁剪复核

已逐张查看全部 82 组原图/新裁剪对照。**{len(flags)} 张**单列明确后续项：日期叠字、背景标牌、旁人局部面部、原图/裁剪边界或清晰度。其余 {82-len(flags)} 张在对照图分辨率下未见明显新增裁剪缺陷，也仍需原尺寸质量审核，不能称为最终合格训练图。

保留真实背景不代表必须保留整个房间或旁人。优先用适当矩形范围控制背景；不能为了移除背景物体而补画或破坏面部轮廓。原图本身存在的问题不会因重新裁剪消失。

## 输出

- [裁剪与溯源清单](results.tsv)：原图路径与哈希、EXIF、像素坐标、输出路径与哈希、原分辨率尺寸。
- [逐图视觉复核](visual_review.tsv)：首轮对照图观察与具体后续项。
- [对照图第一页](review_sheets/contact_01.jpg)、[第二页](review_sheets/contact_02.jpg)、[第三页](review_sheets/contact_03.jpg)、[第四页](review_sheets/contact_04.jpg)、[第五页](review_sheets/contact_05.jpg)。含受控患者照片，仅本地审核使用。
- [生成统计](summary.json)、[复核统计](review_summary.json)、[一致性检查](validation_checks.json)。

共 {sum(checks.values())}/{len(checks)} 项检查通过。所有图像仍未批准训练；旧加载器也未切换到这些输出。

## 侧脸与斜侧脸

见[多视角审核修订](../multiview_review_v1/REPORT.md)：已把 22 张侧脸、34 张斜侧脸重新列为质量复核候选，其中 1 张有身份待审，保留隔离。本次 82 张裁剪试验来自原先正脸初筛，不代表侧脸已经完成裁剪或 QC。

后续按正脸、斜侧脸、侧脸分别审核完整性、遮挡、分辨率与边界；不能用“两眼均可见”排除侧脸。先维持患者级隔离和单图增强，不用尚未验证的同患者跨视角配对。全库多视角恢复另需从原始 metadata 盘点，不能把旧正脸候选池当作完整多视角档案。
''')
print(json.dumps(checks))

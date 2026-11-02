"""Non-destructive original/crop lineage cleaning; no training admission."""
import csv, hashlib, json
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timezone
from audit_metadata_sources import mrnkey
from expert_reservation import filter_reserved
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ssl_image_cleaning'

def read(p):
    with (ROOT/p).open() as f: return list(csv.DictReader(f,delimiter='\t'))
def write(name, rows, fields):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
def norm(p):return str(Path(p)) if p else ''
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    images=read('ssl_admission_inventory/image_admission_ledger.tsv')
    full=read('data_release_audit/image_manifest.tsv')
    candidates=read('data_release_audit/chop_patient_consistent_candidates.tsv')
    meta={r['crop_sha256']:r for r in candidates}
    candidate_hashes={r['crop_sha256'] for r in images}
    assert len(candidate_hashes)==len(images)==len(meta)
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    originals={}
    # Link paths only when exactly equal after lexical normalization. Filename,
    # date and patient ID alone never establish that two photos are identical.
    for r in full:
        h=r['crop_sha256'];p=norm(r['image_path'])
        if not h:continue
        find(h)
        if p:
            if p in originals:union(h,originals[p])
            else:originals[p]=h
    roots={find(h) for h in candidate_hashes}
    related=defaultdict(list)
    for r in full:
        h=r['crop_sha256'];p=norm(r['image_path'])
        root=find(h) if h else find(originals[p]) if p in originals else None
        if root in roots:related[root].append(r)
    kept,_=filter_reserved(full)
    kept_objects={id(r) for r in kept}
    reserved_roots={find(r['crop_sha256']) if r['crop_sha256'] else find(originals[norm(r['image_path'])])
                    for r in full if id(r) not in kept_objects and
                    (r['crop_sha256'] or norm(r['image_path']) in originals)}
    groups=defaultdict(list)
    for r in images:groups[find(r['crop_sha256'])].append(r)
    conflicts=set();reserved=set();group_info={}
    for root,rr in groups.items():
        ids={mrnkey(v) for row in related[root] for k in ['mrn','original_mrn','latest_norm_mrn','identity_candidates']
             for v in row.get(k,'').split('|') if mrnkey(v)}
        ids.update(r['mrn'] for r in rr)
        conflict=len(ids)>1
        protected=root in reserved_roots or any(r['reservation_blocked']=='True' for r in rr)
        if conflict:conflicts.update(r['mrn'] for r in rr)
        if protected:reserved.update(r['mrn'] for r in rr)
        group_info[root]=(ids,conflict,protected)
    aliases=[];ledger=[];representatives=[];group_rows=[]
    for root,rr in sorted(groups.items()):
        # A stable bookkeeping representative, never an assertion of best QC.
        chosen=min(rr,key=lambda r:(-int(meta[r['crop_sha256']].get('crop_width') or 0)*int(meta[r['crop_sha256']].get('crop_height') or 0),r['crop_sha256']))
        gid='photo_'+hashlib.sha256('|'.join(sorted({r['crop_sha256'] for r in related[root] if r['crop_sha256']})).encode()).hexdigest()[:24]
        ids,conflict,protected=group_info[root]
        paths={norm(r['image_path']) for r in related[root] if r['image_path']}
        for r in rr:
            patient_block=r['mrn'] in conflicts or r['mrn'] in reserved
            selected=r['crop_sha256']==chosen['crop_sha256']
            state='reserved_patient' if r['mrn'] in reserved else 'identity_review_patient' if r['mrn'] in conflicts else 'representative_pending_qc_and_split' if selected else 'alternate_crop_same_source_group'
            row=dict(mrn=r['mrn'],subset=r['subset'],source=r['source'],photo_group_id=gid,
                     image_id='crop_'+r['crop_sha256'],crop_sha256=r['crop_sha256'],
                     original_image_path=r['original_image_path'],cropping_path=r['cropping_path'],
                     related_original_paths=len(paths),candidate_crops_in_group=len(rr),
                     group_identity_conflict=conflict,patient_identity_review=r['mrn'] in conflicts,
                     patient_reserved=r['mrn'] in reserved,representative=selected,
                     cleaning_status=state,session_id='',session_status='unverified',
                     pixel_near_duplicate_status='not_screened',final_crop_qc='pending',
                     experiment_partition='protected' if r['mrn'] in reserved else 'unassigned',training_admitted=False)
            ledger.append(row)
            if selected and not patient_block:representatives.append(row)
        group_rows.append(dict(photo_group_id=gid,candidate_crops=len(rr),source_rows=len(related[root]),
                               original_paths=len(paths),all_related_identity_values=len(ids),
                               candidate_patients=len({r['mrn'] for r in rr}),
                               identity_conflict=conflict,reserved=protected,
                               provisional_representative_hash=chosen['crop_sha256']))
        for r in related[root]:
            aliases.append(dict(photo_group_id=gid,source=r['source'],mrn=r['mrn'],
                                original_mrn=r.get('original_mrn',''),latest_norm_mrn=r.get('latest_norm_mrn',''),
                                identity_candidates=r.get('identity_candidates',''),
                                original_image_path=r['image_path'],cropping_path=r['cropping_path'],
                                crop_sha256=r['crop_sha256']))
    review=[r for r in ledger if r['patient_identity_review']]
    fields=list(ledger[0])
    write('image_lineage_ledger.tsv',ledger,fields)
    write('deduplicated_ssl_candidates.tsv',representatives,fields)
    write('identity_review_queue.tsv',review,fields)
    write('source_aliases.tsv',aliases,list(aliases[0]))
    write('photo_groups.tsv',group_rows,list(group_rows[0]))
    per=[]
    for sub in ['1','2','3','4']:
        rr=[r for r in ledger if r['subset']==sub]
        kept=[r for r in representatives if r['subset']==sub]
        per.append(dict(subset=sub,input_images=len(rr),input_patients=len({r['mrn'] for r in rr}),
                        reserved_images=sum(r['patient_reserved'] for r in rr),
                        identity_review_images=sum(r['patient_identity_review'] for r in rr),
                        redundant_candidate_crops=sum(not r['representative'] for r in rr),
                        candidate_representatives=len(kept),candidate_patients=len({r['mrn'] for r in kept})))
    write('subset_summary.tsv',per,list(per[0]))
    checks=dict(input_coverage=len(ledger)==123982 and len({r['crop_sha256'] for r in ledger})==123982,
                one_representative_per_group=sum(r['representative'] for r in ledger)==len(groups),
                representatives_unique=len(representatives)==len({r['photo_group_id'] for r in representatives}),
                protected_absent=all(not r['patient_reserved'] for r in representatives),
                conflicts_absent=all(not r['patient_identity_review'] for r in representatives),
                subset1_excluded=all(r['subset']!='1' for r in representatives),
                no_admission=all(not r['training_admitted'] for r in ledger),
                no_unverified_session_assignment=all(not r['session_id'] for r in ledger),
                all_candidates_have_source_lineage=all(related[k] for k in groups),
                source_paths_linked=all(find(h)==find(originals[p]) for p,h in originals.items()))
    assert all(checks.values()),checks
    s=dict(compiled_at_utc=datetime.now(timezone.utc).isoformat(),input_images=len(ledger),
           photo_groups=len(groups),groups_with_multiple_candidate_crops=sum(len(rr)>1 for rr in groups.values()),
           alternate_candidate_crops=sum(not r['representative'] for r in ledger),
           reserved_patients=len(reserved),identity_review_patients=len(conflicts),identity_review_images=len(review),
           cleaned_candidate_images=len(representatives),cleaned_candidate_patients=len({r['mrn'] for r in representatives}),
           state_counts=dict(Counter(r['cleaning_status'] for r in ledger)),subsets=per,
           original_files_modified=0,training_admissions=0,
           scope='Exact byte-hash and exact original-path lineage; no pixel similarity or date-derived merge',
           representative_policy='Largest historical crop pixel area, then lexical hash; provisional until crop QC')
    for name,obj in [('summary.json',s),('validation_checks.json',checks)]:
        (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
    inputs=['ssl_admission_inventory/image_admission_ledger.tsv','data_release_audit/image_manifest.tsv',
            'data_release_audit/chop_patient_consistent_candidates.tsv','clean_ssl_image_lineage.py',
            'expert_reservation.py','audit_metadata_sources.py','DATA_SCOPE.md']
    inputs += [f'{d}/{f}' for d,files in [('expert_holdout_audit',['protected_mrns.tsv','blocked_paths.tsv','blocked_crop_hashes.tsv']),('split_plan_v0',['ian_protected_mrns.tsv','ian_blocked_paths.tsv','ian_blocked_crop_hashes.tsv'])] for f in files]
    (OUT/'input_versions.json').write_text(json.dumps({p:digest(ROOT/p) for p in inputs},indent=2)+'\n')
    tab='\n'.join('| '+' | '.join(str(r[k]) for k in ['subset','input_images','reserved_images','identity_review_images','redundant_candidate_crops','candidate_representatives'])+' |' for r in per)
    (OUT/'REPORT.md').write_text(f'''# SSL 图像清理：患者、原图与裁剪关联

本次完成元数据与精确重复层面的清理，不删除或覆盖任何原始照片。保留全部原始候选的去向记录，不修改此前四个 subset 或训练准入。

从 {len(ledger):,} 张字节唯一裁剪中，按完全相同原图路径和已知相同裁剪哈希的关联形成 **{len(groups):,} 个来源组**。其中 **{s['groups_with_multiple_candidate_crops']:,} 组**包含多个候选裁剪，共 **{s['alternate_candidate_crops']:,} 张**标为替代裁剪，不在去重候选表中重复采样。

来源组代表记录关联，而非经过像素确认的独立拍摄事件。相同文件名、相同患者或相同日期不会触发合并；不同路径的重新编码、缩放、近重复尚未完成像素筛查。

| Subset | 输入图像 | 保留患者图像 | 身份待审图像 | 替代裁剪 | 去重后候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
{tab}

保留规则和身份冲突作用于该患者全部图像。上述保留、待审和替代裁剪数量可能重叠，不能逐列相加。重新核查原图全部别名的身份字段后，**{len(conflicts):,} 名候选患者**进入身份待审。这些差异可能来自身份修正或旧记录，尚不能判定为真正错配；本轮保守隔离并保留来源证据。去重候选表剩 **{s['cleaned_candidate_patients']:,} 人 / {len(representatives):,} 张**；这是清理后的候选量，并非批准训练量。

每个来源组暂按旧技术审计的裁剪像素面积选代表，哈希用于打破并列；这个选择不代表图像质量最好。后续 QC 可在同组选择另一个合格裁剪，也可能整组排除。源文件保持不变。

## 输出

- `image_lineage_ledger.tsv`：全部候选的患者、来源组、裁剪 ID、处理状态及暂定代表标记。
- `source_aliases.tsv`：关联原始元数据行，保留原图与裁剪路径、身份和哈希。
- `photo_groups.tsv`：来源组统计。
- `deduplicated_ssl_candidates.tsv`：排除保留和身份待审患者、每组仅一个暂定代表的候选清单。
- `identity_review_queue.tsv`：身份待审患者的全部候选图像。
- `sample_pixel_audit.tsv`、`sample_pixel_summary.json`、`sample_similarity_review.tsv`：独立运行 `inspect_ssl_cleaning_sample.py` 产生的 290 张样本读图与像素相似度检查，不代表全库已完成近重复检查。
- `subset_summary.tsv`、`summary.json`：汇总；`input_versions.json`：输入版本。

拍摄事件 ID 留空，不能把路径中的日期当作已验证事件。所有记录 `training_admitted=False`，不会因这次去重自动进入训练。含患者标识和文件路径的表仅用于受控本地工作。

完成 {sum(checks.values())}/{len(checks)} 项检查。下一层清理是近重复像素筛查，以及逐图正脸/裁剪质量审核；实验分区、家系隔离仍需完成。运行 `python clean_ssl_image_lineage.py` 可重建派生输出；人工裁决应另存，以免重建覆盖。
''')
    print(json.dumps(s,indent=2))
if __name__=='__main__':main()

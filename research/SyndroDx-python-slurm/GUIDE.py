"""
SyndroDx 源码导航
=================

这是源码组织层，不是最终数据清洗/训练完成证明。所有 87 个原始脚本保持逐字节不变。
58 个完全相同的备份副本去掉；29 个仅存在于快照中的 face_qc 文件恢复后单独标注。

先运行：
  python3 syndrodx.py list
  python3 syndrodx.py flow
  python3 syndrodx.py show clean_ssl_image_lineage
  python3 syndrodx.py check --workspace /opt/craniofacial/disease-diagnosis-model

code/ 下按职责组织源码，不要直接运行其中的文件。原脚本大量依赖 __file__ 的位置和兄弟导入。
在 HPC 使用 install 先预览差异，再用 --apply 同步源码到原有工作目录；变更前自动备份。
同名不同内容的 HPC 文件默认拒绝覆盖；只有逐项核对后才使用 --replace-changed。
运行入口使用 run --workspace ... SCRIPT -- 原脚本参数；默认仅显示命令，--execute 才运行。
工具不创建虚构数据，不重写旧 manifest，不自动批准训练，不提交 GPU 作业，不发送邮件。

主线 A：来源审计 → 历史标签引导 → 候选图像与证据关联 → 词法概念/身份保留 → 划分提案。
主线 B：四子集 → 第四子集初筛 → SSL 准备台账 → 图像谱系与身份去重。
主线 C：完整多视角原图 CPU 审计 → 冻结 Qwen v5 审核 → original-only release → 矩形裁剪。
这些分支并非一条无条件可自动重跑的流水线。中间依赖人工记录、非 Python 配置和既有审计输出。
legacy_training 尚未接入主线 C；不能把 pretrain_release 的输出直接交给旧 ssl_train。

historical_reviews 包含固定样本编号、逐图观察和历史审批断言，只供追溯。
operations 包含作业启动、迁移、快照和邮件功能，不属于日常数据清理。
历史 Qwen v1-v4 保留独立版本，因为导入和冻结哈希依赖这些文件；不能直接合并或用 v5 替代。

上传规则：只允许 .py 也不等于可以公开。历史观察、内部路径、联系人和作业信息仍需逐项审核。
本次没有上传 GitHub。缺少 OBO、prompt、requirements.lock、人工审核、患者保留表、图片、数据。
请在 HPC 保留这些受控资源，不上传到公开仓库。
"""

# Slurm integration: see SLURM_GUIDE.md; jobs.py list/show/check are read-only.
# syndrodx.py install syncs Python only, not Slurm scripts.
# pretrain_visual_qc.py and finalize_pretrain_release.py were not supplied.

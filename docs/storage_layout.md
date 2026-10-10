# 共享数据集与项目文件存储

数据根目录为 `/data1/llx`。按用户要求，数据集独立存放，各项目可以复用；
BigSmallCollab 的预处理输入、教师缓存、模型和逐折训练产物统一放在
`/data1/llx/BigSmallcollab`。注意项目数据目录的大小写为 `BigSmallcollab`。
代码工作树 `/home/lixinli/BigSmallCollab` 保存代码、配置、文档，以及用户
查看的结果汇总报告；后者保存到真实的 `results/` 目录。

## 共享数据集

每个数据集有独立目录：`/data1/llx/<数据集名称>/`。现有
`BNCI2014001`、`BNCI2014004`、`BNCI2015001`、`AlexMI` 的根目录
NPY 保持原样。不同频段或生成流程使用明确的版本子目录，避免覆盖旧版本。

新 14001 全场次宽带数据位置：

```text
/data1/llx/BNCI2014001/broadband_0p1_75hz/
├── X.npy
├── labels.npy
├── y.npy
├── meta.csv
├── trials.csv
├── legacy_row_mapping.csv
├── manifest.json
├── verification.json
└── raw/
```

该版本包含 9 位被试、`0train` 与 `1test` 的全部 5184 trials。
`X.npy` 为 float64，形状 `(5184, 22, 1001)`，250 Hz，源频带 0.1–75 Hz。
`raw/` 保存生成该版本使用的原始 MAT。标签、通道顺序、场次/run、物理
trial UID、软件版本及文件哈希一同保存，其他项目可以直接读取。
共享源不按某个模型的要求执行 CAR、EA、重采样或通道插值。

官方预训练权重只保存一份，统一使用 `/data1/llx/pre_weight/`。
项目配置直接引用该路径，不为每个项目、模型或实验另存副本。

## 项目专用文件

```text
/data1/llx/BigSmallcollab/
├── cache/        # 模型输入、teacher_targets、历史缓存 artifacts
├── results/      # 逐折训练产物、配置、指标、预测和执行日志
├── weights/      # 确有独立用途的项目模型，不存重复官方预训练
├── git_lfs/      # 历史实验产物的本地 Git LFS 实体
└── migrations/   # 目录迁移与校验记录
```

模型输入存放在
`cache/eegfm_alignment_v2/model_inputs/`。最新 `wideband_npy_v3` 从共享
14001 全场次 NPY 中只选择 `0train`；二分类再选择左右手。MIRepNet 和
CBraMod 各有自己的预处理输入，不复制一份新的共享数据集。

当前 004/5001 的模型输入与目标缓存分别使用
`cache/reproductions/loso_source_refresh_004_5001_v1/model_inputs/` 和
`cache/reproductions/loso_source_refresh_004_5001_distillation_v1/teacher_targets/`。
这些路径已参与实验身份核对，本轮整理保持其路径和文件字节不变。

历史输入缓存归入外部 `cache/`。历史 `teacher_cache` 有代码直接读取原始
路径时，旧外部路径可使用指向新 `cache/` 的兼容软链接；数据实体只有一份。
工作树内不建立缓存或权重软链接。Git 历史和 `git_lfs/` 不在本轮删除范围内。

## 结果汇总报告

按用户最新要求，Excel、CSV 和 JSON 汇总报告保存到
`/home/lixinli/BigSmallCollab/results/`，使用真实文件和目录，不创建指向
`/data1` 的软链接。报告导出使用 `experiments.storage.REPORTS_ROOT` 与
`require_report_output()`，直接读取报告路径，不经过训练产物的历史路径映射。
生成的报告默认不提交 Git。

该目录只保存面向用户的报告和小型记录归档，不放 EEG 数组、教师目标、
模型参数或训练断点。若旧报告目录混入重复训练元数据或执行日志，将其
归入外部逐折产物目录；保留用户使用的 Excel、CSV 和 JSON 汇总。

## 保留范围

训练断点只用于未完成的训练。实验完成后，临时优化器、模型副本和训练
history 不再留在 `training_state.pt` 中；顺序处理被试折的入口使用轻量
`completed_state.pt` 保存终态 RNG、完成标记和原实验身份字段，不保留优化器。
历史大断点必须核对已完成状态和结果后才转换、删除；中断实验仍保留完整断点。

最终学生权重和 feature 投影层不默认永久累积。输入缓存、教师目标和微调
教师按后续复用需求保留，小型指标、预测、配置和来源记录用于报告与追溯。

用户计划继续开展蒸馏。当前新数据源 001/001-4/004/5001 的微调教师、
教师目标缓存、学生权重和投影层先保留，待明确后续复用范围再清理。
这一保留范围包括 585 个新源 LOSO 基线单元及 1512 个 004/5001 蒸馏单元。
新实验通常重新初始化投影层；继续既有训练或分析已对齐的特征时，
需要保留对应学生和投影层。推理不用投影层不等于它没有后续用途。

删除已退休实验的模型前，先让完成结果读取器接受保留的指标和来源记录，
避免因权重被清理而误判为未完成并重训。需要继续蒸馏的教师仍应验证实际
权重和目标缓存；删续训状态不等于删教师或学生的最终权重。

本轮授权清理及执行记录见 [整理计划](storage_cleanup_plan.md)。已删除
351 份大续训状态（13.58 GiB）、380 份 CodeBrain 最终权重（33.91 GiB）、
项目 `codebrain.pth` 及退休数值缓存，并归档文本成绩。总计实际释放约
48.06 GiB；操作和核对结果见 `results/storage_cleanup/cleanup_report.json`。
CodeBrain 的成绩 CSV 和文本 ZIP 位于 `results/archive/codebrain/`。

本轮 004/5001 三 seed 报告位于：

```text
results/loso_source_refresh_004_5001/
├── loso_distillation_accuracy_3seed.xlsx
├── all_results.csv
└── summary.json
```

## 路径迁移与数据身份

`experiments/storage.py` 集中定义共享数据集根目录、项目数据根目录和
新版 14001 路径。读取历史项目路径和此前的外置路径时，通过 resolver
映射到新目录；不创建工作树内的数据兼容软链接。

历史 manifest、trial 表及结果记录保留原字节内容和哈希，避免影响实验追溯。
迁移只调整存储位置，不改变 EEG 数值、session、类别、超参数或种子。
移动前核对依赖和正在运行的进程；未完成任务的续训状态不能列入清理范围。

数据写入和原子 checkpoint 写入经过外置路径检查。新增通用源数据放入
对应独立数据集目录；新增项目输入、教师目标、训练权重和逐折产物放入
`/data1/llx/BigSmallcollab` 的相应子目录。

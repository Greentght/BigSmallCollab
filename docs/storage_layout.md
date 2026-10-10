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

已有预训练权重继续使用 `/data1/llx/pre_weight/`。

## 项目专用文件

```text
/data1/llx/BigSmallcollab/
├── cache/        # 模型预处理输入、协议筛选输入等
├── results/      # 实验日志、指标、预测、教师目标、训练 checkpoint
├── weights/      # 项目维护的模型权重
├── git_lfs/      # 历史实验产物的本地 Git LFS 实体
└── migrations/   # 目录迁移与校验记录
```

模型输入存放在
`cache/eegfm_alignment_v2/model_inputs/`。最新 `wideband_npy_v3` 从共享
14001 全场次 NPY 中只选择 `0train`；二分类再选择左右手。MIRepNet 和
CBraMod 各有自己的预处理输入，不复制一份新的共享数据集。

历史 `cache/eegfm_alignment_v2/rebuilt/` 已经筛选实验所需的单个场次，
其中 001-4 还裁至 1000 点。它们属于此前配置对齐实验的协议输入，保留在
项目目录下。历史 `mne_data/` 和对应 LFS 实体也按旧实验来源保存。

## 结果汇总报告

按用户最新要求，Excel、CSV 和 JSON 汇总报告保存到
`/home/lixinli/BigSmallCollab/results/`，使用真实文件和目录，不创建指向
`/data1` 的软链接。报告导出使用 `experiments.storage.REPORTS_ROOT` 与
`require_report_output()`，直接读取报告路径，不经过训练产物的历史路径映射。
生成的报告默认不提交 Git。

## 保留范围

官方预训练权重使用 `/data1/llx/pre_weight/` 中的唯一副本。训练断点只用于
未完成的训练；最终学生权重和 feature 投影层不默认长期保留。微调后的
教师仅在后续蒸馏仍需导出目标时临时保留。输入和教师目标缓存按已计划的
复用需求保留，完成后不无限累积。小型指标、预测、配置和来源记录用于
结果报告和追溯。

当前旧 runner 将模型文件存在视作完成条件。清理现有模型前应修正该判断，
使已经保存的最终结果可独立读取，避免清理后误触发重训。

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

历史 manifest、trial 表及结果记录保留原字节内容和哈希，避免影响已完成
实验的追溯以及训练断点恢复。迁移只调整存储位置，不改变 EEG 数值、
session、类别、模型超参数或随机种子。实际目录移动前先保存训练断点，
完成后按原实验配置恢复后台队列。

数据写入和原子 checkpoint 写入经过外置路径检查。新增通用源数据放入
对应独立数据集目录；新增项目输入、教师目标、训练权重和逐折产物放入
`/data1/llx/BigSmallcollab` 的相应子目录。

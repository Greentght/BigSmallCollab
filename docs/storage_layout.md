# 数据与模型存储位置

用户要求：数据、模型权重及训练 checkpoint 保存到 `/data1/llx`，不再写入
BigSmallCollab 项目目录。运行入口使用 `experiments/storage.py` 的共享路径。

| 内容 | 保存位置 |
|---|---|
| 原有数据集 NPY | `/data1/llx/BNCI2014001`、`BNCI2014004`、`BNCI2015001`、`AlexMI` |
| 新 14001 全场次宽带 NPY | `/data1/llx/data_cache/loso_source_v3/BNCI2014001` |
| 模型输入缓存与参考源缓存 | `/data1/llx/data_cache/eegfm_alignment_v2` |
| 实验结果、权重、checkpoint、预测和教师缓存 | `/data1/llx/BigSmallCollab_results` |
| 预训练权重 | `/data1/llx/pre_weight` |
| 原项目 `weights/` 中的权重 | `/data1/llx/BigSmallCollab_weights` |
| 本地 Git LFS 实体文件 | `/data1/llx/BigSmallCollab_git_lfs` |

## 原 data_cache 内的内容

迁移前该目录实际占约 2.5 GB。其中新全场次源目录约 1.6 GB，包含约
871 MiB 的 float64 EEG 数组，以及约 744 MiB 的原始 MAT 文件；另外约
930 MiB 是二分类和四分类分别对应 MIRepNet、CBraMod 的四份模型输入。

- `loso_source_v3/BNCI2014001`：9 位被试、`0train` 与 `1test` 的全部
  5184 trials。250 Hz，22 通道，每 trial 1001 点，源频带 0.1–75 Hz。
  同时保存标签、场次/run 元信息、旧 trial 映射和来源哈希。
- `loso_source_v3/BNCI2014001/raw`：重新导出全 session 时使用的本地原始
  MAT 文件。它们是原始下载缓存，与已经滤波、分段的 NPY 用途不同。
- `eegfm_alignment_v2/rebuilt`：001-4、004、5001 的参考源 NPY 与 trial
  映射。它们用于此前数据来源及配置对齐实验。
- `eegfm_alignment_v2/model_inputs`：按实验 profile 和模型区分的预处理
  输入。最新 `wideband_npy_v3` 在全场次源上只选择 `0train`。
- `eegfm_alignment_v2/mne_data`：此前参考流程的原始下载缓存。

旧参考文件有一部分是 Git LFS 指针，其真实内容曾存于 `.git/lfs`。
共享读取函数会验证并解析外置 LFS 实体，避免把指针当作 NPY/MAT 读取。

## 路径切换与校验

迁移记录保存到 `/data1/llx/storage_migration_20261007.json`。复制采用
rsync，并逐目录执行 checksum 比较；清理旧副本必须以完整校验为前提。
历史 manifest 不改字节内容，以保留已记录的来源和哈希；其中旧绝对路径
通过共享函数映射到外置位置。迁移不修改数据值、session 或训练超参数。

数据写入、原子 checkpoint 写入及训练输出入口执行外置路径检查。
项目目录保留代码、配置、trial 清单和文档，不建立数据目录兼容软链接。

# 14001 全场次宽带 NPY 与两模型 LOSO

本轮按用户要求保存 14001 的全部 session，再对 MIRepNet 和 CBraMod 分别运行
001 二分类与 001-4 四分类 LOSO。实验仍使用原先的 `session_T`，对应
MOABB 1.2.0 的 `0train`；新缓存中同时保留 `1test`。

## 新源缓存

目录：`/data1/llx/data_cache/loso_source_v3/BNCI2014001/`。

- `X.npy`：真实 float64 NPY，形状 `(5184, 22, 1001)`，250 Hz，单位 µV。
- 源信号处理：连续 MOABB Raw 上 0.1–75 Hz IIR 滤波，再按事件提取 epoch。
- 9 位被试、两个完整 session；每 session 2592 trials，每被试每 session 288。
- 导出阶段保留 inclusive endpoint，不执行模型截窗、重采样、CAR、EA 或 notch。
- `labels.npy` 为原字符串标签，`y.npy` 为固定四分类编号；`meta.csv` 和
  `trials.csv` 保存 subject/session/run/event ordinal、原始文件哈希和物理 trial UID。
- `manifest.json` 保存单位、通道顺序、软件版本、处理参数及所有文件哈希。
- `legacy_row_mapping.csv` 将全部 5184 trials 与旧缓存一一对应。

缓存由 `experiments/finetune/export_bnci14001_all_sessions.py` 生成。原始 MAT
文件复用本地已有下载的完整 Git LFS 实体，在独立 raw 目录中保存。
旧 `/data1/llx/BNCI2014001` 缓存保留。

实际数值核验：新缓存 `0train` 的前 1000 点与此前宽带 MOABB v2 缓存
逐元素完全一致；UID、标签和顺序相同，最大绝对差与 RMSE 均为 0。

## 模型输入与划分

生成入口为 `experiments/finetune/prepare_bnci14001_wideband_inputs.py`。
输入路径为 `/data1/llx/data_cache/eegfm_alignment_v2/model_inputs/wideband_npy_v3/`，
与旧输入缓存隔离；每份输入从新的全场次源 NPY 读取并记录其哈希。

| 任务 | 所选 trial 数 | 每折训练 | 每折测试 |
|---|---:|---:|---:|
| 001，left/right | 1296 | 1152 | 144 |
| 001-4，left/right/feet/tongue | 2592 | 2304 | 288 |

两模型复用同一 trial UID 和类别映射。先选 `0train` 和任务类别，再取前
1000 个原生点；CBraMod 在重采样前去除 inclusive endpoint，与参考代码一致。

MIRepNet：宽带 epoch → 四阶双向 Butterworth 8–30 Hz → 选定任务中每位
被试单独 EA → 45 通道逆距离插值，输入 `(B, 45, 1000)`。测试被试 EA
使用其全部无标签 trials，记录为 transductive LOSO。二分类和四分类的
EA 协方差分别在各自 task 内计算。

CBraMod：250→200 Hz MNE 重采样 → 0.3–75 Hz → 60 Hz notch → CAR，输入
`(B, 22, 4, 200)`。四分类输入已核验与先前参考宽带模型输入逐元素一致。
这是一次从新 NPY 重新准备、重新训练的运行，不复用旧训练结果。

## 固定训练配置

配置规格：`configs/reproductions/bnci14001_wideband_loso_v3.yaml`。

| 模型 | Optimizer/LR | Batch/WD | Epochs |
|---|---|---|---|
| MIRepNet | Adam / 1e-3 | 8 / 1e-6 | 二分类 10，四分类 20 |
| CBraMod | AdamW / 1e-3 | 16 / 0.1 | 两任务均 20 |

MIRepNet 沿用当前原生模型及每 epoch cosine；CBraMod 沿用近期四分类
参考对齐配方：head dropout 0.5、warmup 5、min LR 1e-6、epoch 末更新
参考逐 step LR 表。将四分类配方迁移至二分类的来源在结果中明确记录。

seeds 为 0、1、2；四个模型/任务组合 × 9 折 × 3 seeds，共 108 个正式
训练单元。每折重新加载预训练权重，最终 epoch 后评估一次。
新 MIRepNet seeds 与旧 666/667/668 不作为严格配对种子比较。

## 后台调度和产物

调度器为 `experiments/finetune/dispatch_bnci14001_wideband_loso.py`，脱离
终端运行。只使用 GPU 1–9，最多六个 worker 同时运行；等待 GPU 利用率
不超过 40% 且空闲显存至少 10000 MiB。关闭聊天或 Cursor 后后台任务继续。

四个组合分别进行独立预检，再开始正式 seed worker。断点检查输入 manifest、
配置和预训练权重哈希。状态记录 PID、GPU、已完成折数和失败日志。

- 状态：`/data1/llx/BigSmallCollab_results/reproductions/loso_source_v3/execution_logs/wideband_14001_loso_status.json`
- 结果：`/data1/llx/BigSmallCollab_results/reproductions/loso_config_alignment_v2/wideband_npy_v3/`
- 每折保存固定轮数历史、最终权重、预测和最终指标。
- 最终汇总先对每 seed 的九被试等权平均，再报告三 seed 均值和样本标准差。
- CBraMod 四分类与近期窄带 NPY 和宽带 MOABB 对照比较实际 UID、标签、
  初始化及 batch 顺序；其他组合单独报告，避免混用不同种子的旧结果。

## 外置存储

按用户要求，数据缓存和模型产物统一保存到 `/data1/llx`。项目中只保留代码、
配置及说明。共享路径定义在 `experiments/storage.py`，同时映射历史 manifest
中的旧路径；历史 manifest 保持原字节内容，以保留实验输入身份和哈希。

- 数据缓存：`/data1/llx/data_cache/`。
- 结果、checkpoint 与预测：`/data1/llx/BigSmallCollab_results/`。
- 原 `weights/`：`/data1/llx/BigSmallCollab_weights/`。
- Git LFS 实体：`/data1/llx/BigSmallCollab_git_lfs/`。
- 迁移校验记录：`/data1/llx/storage_migration_20261007.json`。

存储路径切换不改变数据值、session、类别、模型超参数或实验随机种子。

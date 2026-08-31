# BigSmallCollab 代码阅读指南

这份文档按当前目录职责介绍怎么读代码。先记住一句话：

```text
配置和数据切分 -> adapter 统一模型 -> export 产出 artifact -> experiments 编排实验 -> collab 提供协同算法 -> eval 做统计
```

现在的阅读模式是：**不要从 `scripts/` 开始理解实验框架**。`scripts/` 只是工具箱；正式实验入口在 `experiments/`，可复用协同算法在 `collab/`。

## 1. 总体分层

```text
configs/       数据集、模型、实验 YAML 配置
config.py      读取 YAML 配置
paths.py       解析预训练权重路径

data/          数据加载、通道表、预处理、统一切分
models/        模型结构 + adapter 统一接入层
collab/        可复用协同算法和 artifact hub
experiments/   正式实验入口：config runner + active line-specific drivers
scripts/       工具入口：check/、export/、legacy/
eval/          acc/kappa、per-class 指标和 subject 级统计检验
results/       输出目录：artifacts、metrics、tuned summaries
```

核心设计是：模型代码已经 vendored 到 `models/`，但大模型依赖仍可能不兼容。因此训练和导出可以在各自 conda 环境里完成，协同阶段靠标准化 artifact 解耦。

目录边界：

- `collab/`：库层，只放可复用算法、artifact 读写和通用融合/蒸馏逻辑。
- `experiments/`：实验层，负责 protocol、condition、subject/seed 遍历、CSV 输出和专门实验 driver。
- `scripts/`：工具层，只放自检、导出、批处理和归档复现脚本。
- `scripts/legacy/`：历史/负结果复现，不作为新增实验入口。

## 2. 配置入口

先看这几个文件：

- `config.py`：提供 `load_dataset_config(name)` 和 `load_model_config(name)`。
- `configs/datasets/*.yaml`：定义数据集类别数、通道数、被试数、默认 seeds、`val_split`。
- `configs/models/*.yaml`：定义每个模型的 env、大小模型类型、epochs、lr、batch_size、weight_decay 等。
- `configs/exp/*.yaml`：正式实验配方。新增矩阵实验优先写这里。
- `paths.py`：解析预训练权重。权重默认走 `weights/*.pth`，也可以用环境变量覆盖，例如 `MIREPNET_WEIGHT`、`CBRAMOD_WEIGHT`、`LABRAM_WEIGHT`。

注意：`val_split` 在项目里表示测试集比例，例如 `0.3` 是 70% 校准/训练，30% 测试。

## 3. 数据流水线

数据相关文件：

- `data/eeg_dataset.py`：从 `/data1/llx/<DATASET>/` 读取原始 `.npy` 数据和标签，并按数据集规则选择 session、subject、类别。
- `data/split.py`：统一产生训练/测试切分，是跨模型样本顺序对齐的唯一来源。
- `data/preproc.py`：提供 EA、通道补齐、bandpass、notch 等基础预处理函数。
- `data/channels.py`：保存各数据集通道名和 scalp 位置，用于通道映射和补齐。

最重要的是 `data/split.py`：

```text
data.subject_split(dataset, subject, val_split, seed)
  -> X_tr, y_tr, X_te, y_te

data.loso_split(dataset, test_subject)
  -> X_tr, y_tr, subj_tr, X_te, y_te
```

所有模型必须使用这里生成的 split。后续 artifact 的第 i 行必须对应同一个样本，否则集成和蒸馏都会按行错位。

## 4. 模型接入层

统一接口在 `models/base.py`。

每个 adapter 都实现：

```text
preprocess(X_raw) -> 模型输入张量
build(num_classes) -> nn.Module
forward(model, x) -> feat, logits
```

`ModelAdapter` 基类再提供：

```text
finetune(...)
infer(...)
export(...)
mc_uncertainty(...)
```

模型注册在 `models/__init__.py` 的 `get_adapter(name, ...)`。

优先读 adapter，不要一开始钻进完整网络结构：

- `models/ifnet/adapter.py`：小模型，基本吃原始 `(B,C,T)`。
- `models/eegnet/adapter.py`：小模型，调用 `ResidualEEGNet`。
- `models/adfcnn/adapter.py`：小模型，调用 `ADFCNN_Net`。
- `models/mirepnet/adapter.py`：大模型，先做 EA + 45 通道补齐，再进入 `models/mirepnet/mlm.py`。
- `models/cbramod/adapter_native.py`：CBraMod settled native/tuned 版；`--model cbramod` 即走这里，`cbramod_native` 为兼容别名。
- `models/labram/adapter.py`：LaBraM，250Hz 转 200Hz patchify，并计算 `input_chans` 通道映射。

网络结构文件可以后读：

```text
models/ifnet/ifnet.py
models/eegnet/residual_eegnet.py
models/adfcnn/adfcnn.py
models/mirepnet/mlm.py
models/cbramod/cbramod.py
models/labram/modeling_finetune.py
```

## 5. Artifact Hub

artifact 不是 checkpoint。它是模型对固定样本导出的标准化中间产物，保存位置由 `collab/artifacts.py` 管：

```text
results/artifacts/<dataset>/<model>/<subject>_<seed>_<split>.npz
```

每个 `.npz` 包含：

```text
logits  模型 softmax 前分类分数，形状 (N,C)
feats   倒数第二层特征，形状 (N,D)
y       真实标签，形状 (N,)
```

用途：

- 测试时集成读取多个模型的 `test` artifact。
- 离线蒸馏读取 teacher 的 `train` artifact。
- 特征融合读取大/小模型的 frozen features。
- `y` 用来校验多个模型输出是否逐行对齐。

关键函数：

```text
collab.artifacts.save(...)
collab.artifacts.load(...)
collab.artifacts.load_aligned(...)
```

`load_aligned()` 会检查所有模型的 `y` 是否完全一致；不一致就报错。

## 6. scripts：工具入口

`scripts/` 现在只按工具理解。

### 6.1 自检

入口在 `scripts/check/`：

- `verify_foundation.py`：数据层 + 小模型 vendoring 逐位一致。
- `verify_backbones.py`：大模型 backbone build + forward。
- `smoke_test.py`：数据切分 + adapter forward 契约。

典型命令：

```bash
conda run -n mirepnet python scripts/check/verify_foundation.py
conda run -n mirepnet python scripts/check/verify_backbones.py --model mirepnet
conda run -n cbramod  python scripts/check/verify_backbones.py --model cbramod
conda run -n labram   python scripts/check/verify_backbones.py --model labram
conda run -n mirepnet python scripts/check/smoke_test.py --models ifnet eegnet adfcnn mirepnet
```

### 6.2 微调并导出 artifact

最常用入口是 `scripts/export/finetune_export.py`。

真实流程：

```text
scripts/export/finetune_export.py
  -> config.py 读取 dataset/model YAML
  -> data.subject_split(...) 得到 X_tr/y_tr/X_te/y_te
  -> models.get_adapter(model)
  -> adapter.build(num_classes)
  -> adapter.finetune(model, X_tr, y_tr, num_classes)
  -> adapter.export(train/test)
  -> collab.artifacts.save(...)
```

典型命令：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py --model ifnet --dataset BNCI2014004 --gpu 0
conda run -n mirepnet python scripts/export/finetune_export.py --model mirepnet --dataset BNCI2014004 --gpu 0
conda run -n cbramod  python scripts/export/finetune_export.py --model cbramod --dataset BNCI2014004 --gpu 1
conda run -n labram   python scripts/export/finetune_export.py --model labram --dataset BNCI2014004 --gpu 2
```

只跑部分 subject/seed：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py --model ifnet --dataset BNCI2014004 --subjects 0 1 --seeds 666 --gpu 0
```

LOSO 相关导出仍属于工具层：

- `scripts/export/export_preds.py`：统一导出 within 或 LOSO artifact。LOSO 时 artifact model 名通常变成 `<model>_loso`。
- `scripts/export/export_teacher_loso.py`：专门导出 LOSO teacher；MIRepNet 在 LOSO 中需要按 subject 分别 EA，再合并训练。
- `scripts/export/export_teacher_mc.py`：teacher + MC-dropout 不确定度。
- `scripts/export/export_teacher_loso_subjoof.py`：subject-OOF LOSO teacher 导出。

## 7. experiments：正式实验入口

`experiments/` 是现在读实验的主入口。读任何 active driver 时，按这个顺序看：

```text
parse_args / YAML config
  -> dataset/model config
  -> protocol 或 split
  -> artifact load / model adapter
  -> collab 算法调用
  -> eval.metrics
  -> results/metrics/*.csv
```

### 7.1 YAML 驱动 runner

优先入口：

```bash
conda run -n mirepnet python -m experiments.run configs/exp/distill_kd_within.yaml --report
```

相关文件：

- `experiments/run.py`：按 dataset/unit/seed/condition 循环运行，写 long-form metrics CSV。
- `experiments/protocols.py`：生成 within/LOSO cell。
- `experiments/methods.py`：把 YAML condition 映射成 `collab.distill.distill_student` 参数。

新增矩阵实验优先加 `configs/exp/*.yaml` 和 `experiments/methods.py` registry，而不是新增脚本。

### 7.2 蒸馏实验

入口在 `experiments/distill/`：

- `run_distill.py`：离线 KD + 特征对齐蒸馏的历史主入口；复杂 ablation 仍在这里。
- `run_loso_distill.py`：LOSO 逐 fold 学生蒸馏。
- `run_loso_subject_oof_kd.py`：LOSO subject-OOF KD。

离线蒸馏流程：

```text
teacher 先由 scripts/export 导出 train artifact
  -> experiments/distill 读取 teacher train artifact
  -> data.subject_split(...) 或 data.loso_split(...)
  -> assert teacher['y'] == y_tr
  -> get_adapter(student)
  -> collab.distill.distill_student(...)
  -> 写 results/metrics/*.csv
```

核心训练函数在 `collab/distill.py`：`distill_student(...)`。

基本损失：

```text
loss = CE(student, y)
     + lam_kd   * KL(student logits, teacher logits)
     + lam_feat * cosine_align(project(student feat), teacher feat)
```

典型命令：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py --model mirepnet --dataset BNCI2014004 --gpu 0
conda run -n mirepnet python experiments/distill/run_distill.py --dataset BNCI2014004 --teacher mirepnet --student ifnet --lam_kd 0.5 --lam_feat 0.5 --gpu 0
```

### 7.3 双向 / CR-AMD / BD-EEG / wrong-sample（负结果复现）

入口在 `experiments/bidir/` 和 `experiments/mask/`：

- `experiments/bidir/`：双向互蒸馏（`run_bidir_loso.py` 等）、CR-AMD（`run_cramd_loso.py`）、BD-EEG（`run_bdeeg_loso.py`）、feature-level mutual（`run_featbidir_*.py`）。结论整体 null/不稳定，保留复现。
- `experiments/mask/run_wrong_sample.py`：wrong-sample 利用 E0-E5，closed/null（仅 E2 correct-only KD 存活）。

典型命令（参数以 `scripts/legacy/run_bidir_full.sh` / `run_cramd_full.sh` / `run_bdeeg_parallel.sh` 为准）：

```bash
conda run -n mirepnet python experiments/bidir/run_cramd_loso.py --dataset BNCI2014001-4 --warmup 15 --total 50 --lam_bs 0.5 --lam_sb 0.1 --gpu 2 --tag cramd
```

> 归档说明：D0 及之后的集成/融合/路由/端到端微调基线/ target-support 线
> （原 `experiments/fusion/`、`experiments/adapt/`、`eval/d0.py`、`collab/{fusion,router,ensemble}.py`）
> 已于 2026-08-31 归档删除，可从 git tag `pre-consolidation` 恢复。

### 7.4 大模型原生适配和调参

入口在 `experiments/bigmodel/`：

- `cbramod_native_adapt.py`、`labram_native_adapt.py`：大模型 native pipeline 下游适配。
- `mirepnet_loso_adapt.py`：MIRepNet LOSO 端到端评估。
- `tune_mirepnet_loso.py`、`tune_cbramod_loso.py`：LOSO 场景 search + confirm。
- `tune_cbramod_native.py`、`tune_labram_native.py`：逐数据集 native tuning。
- `tune_cbramod_004_caronly.py`：BNCI2014004 CAR-only 精调。

阅读这些文件时重点看三件事：

```text
1. 它是否走 adapter，还是复刻 native pipeline。
2. 它如何处理 subject-wise EA、通道补齐、resample 和 montage。
3. 它的结果写到 results/metrics 还是 results/<model>_loso/tuned。
```

### 7.5 target-support / 少样本适配（已归档）

`experiments/adapt/`（target-support M0/M1a/M1b/M2）已于 2026-08-31 随
D0-onward 线一并归档删除，代码与结果可从 git tag `pre-consolidation` 恢复。

## 8. collab：协同算法库

`collab/` 是可复用算法层，不应该承担命令行编排。

主要文件：

- `collab/artifacts.py`：跨环境 artifact hub。
- `collab/distill.py`：离线 KD、feature align、DKD、prototype、relational、pearson 等核心训练函数。
- `collab/seed.py`：统一全栈播种（random/numpy/torch/cuda + cudnn）。
- `collab/bidirectional.py`、`collab/mutual.py`、`collab/bdeeg.py`：真正双向/互学习算法，由 `experiments/bidir/` 复现实验调用。

读 `collab/` 时不要从 argparse 或 CSV 输出角度读；它的核心问题是“给定数组、adapter 或 batch，算法怎么算”。

## 9. LOSO 和跨被试协同

LOSO 是 Leave-One-Subject-Out：留一个被试做测试，其余被试训练。

数据入口：

```text
data.loso_split(dataset, test_subject)
```

常见路径：

```text
scripts/export/export_teacher_loso.py
  -> 导出 teacher LOSO artifact

experiments/distill/run_loso_distill.py
  -> 读取 teacher LOSO artifact
  -> student 做跨被试蒸馏

experiments/bigmodel/*_loso*.py
  -> 大模型端到端 LOSO 评估或调参
```

典型命令：

```bash
conda run -n mirepnet python scripts/export/export_teacher_loso.py --dataset BNCI2014004 --gpu 0
conda run -n mirepnet python experiments/distill/run_loso_distill.py --dataset BNCI2014004 --student ifnet --gpu 0
```

## 10. legacy：规范启动器

`scripts/legacy/` 现在只保留 `run_*.sh` 启动器——它们是各实验线的规范命令记录
（mask/dkd/eakd/relational/proto/adaptive/bidir/cramd/bdeeg/loso/cbramod/labram），
路径已指向 `experiments/`，重跑时以它们为准（见 `REPRO.md`）。

- 双向/CR-AMD/BD-EEG 驱动在 `experiments/bidir/`；wrong-sample 在 `experiments/mask/`。
- 早期 `analyze_*`/`aggregate_*` 统计脚本已删，统一用 `python -m eval '<glob>'` 取代。
- D0-onward 线（fusion/adapt/d0）已归档，见 git tag `pre-consolidation`。

真正双向蒸馏和 artifact 离线蒸馏要分开理解：

```text
离线蒸馏：teacher 冻结 -> student 学 teacher artifact
双向蒸馏：两个模型同进程同时 forward、同时更新，并动态决定谁教谁
```

双向蒸馏要求两个模型能在同一个 conda 环境、同一个 Python 进程中同时加载。CBraMod/LaBraM 这类依赖冲突模型更适合走 artifact 解耦。

## 11. 评估与统计

基础指标在 `eval/metrics.py`：

```text
evaluate(y_true, y_pred) -> acc, kappa
preds_from_logits(logits)
per_class(y, pred, num_classes)
```

正式统计在 `eval/stats.py`。它会按 subject/fold 作为配对单位，先把 seeds 聚合成 per-unit mean，然后做：

```text
paired Wilcoxon
Holm correction
bootstrap CI
```

命令入口：

```bash
python -m eval 'results/metrics/*.csv' --baseline ifnet_base
```

理解实验结论时不要只看 raw mean；小提升要看 subject 级配对统计。

## 12. 推荐阅读顺序

### 12.1 第一次读项目

1. `README.md`
2. `CODE_READING_GUIDE.md`
3. `config.py`、`configs/datasets/*.yaml`、`configs/models/*.yaml`
4. `data/split.py`
5. `data/eeg_dataset.py`
6. `models/base.py`
7. `models/__init__.py`
8. 一个小模型 adapter，例如 `models/ifnet/adapter.py`
9. 一个大模型 adapter，例如 `models/mirepnet/adapter.py`
10. `scripts/export/finetune_export.py`
11. `collab/artifacts.py`
12. `experiments/run.py`、`experiments/protocols.py`、`experiments/methods.py`
13. `collab/distill.py` 或 `collab/ensemble.py`
14. `eval/metrics.py`、`eval/stats.py`

### 12.2 读离线蒸馏

1. `scripts/export/finetune_export.py`
2. `collab/artifacts.py`
3. `experiments/distill/run_distill.py`
4. `collab/distill.py`
5. `experiments/run.py`、`experiments/methods.py`
6. 对应 `configs/exp/*.yaml`
7. `eval/stats.py`

### 12.3 读融合和路由

1. `collab/artifacts.py`
2. `experiments/fusion/run_ensemble.py`
3. `collab/ensemble.py`
4. `experiments/fusion/run_balance_gate.py`
5. `collab/fusion.py`
6. `collab/router.py`
7. `docs/experiment_results_context.md`

### 12.4 读大模型 native/LOSO

1. 对应大模型 adapter：`models/mirepnet/adapter.py`、`models/cbramod/adapter_native.py`、`models/labram/adapter.py`
2. `experiments/bigmodel/*_adapt.py`
3. `experiments/bigmodel/tune_*_loso.py`
4. `data/preproc.py`、`data/channels.py`
5. `results/<model>_loso/tuned/*` 和 `eval/`

### 12.5 读历史负结果

1. `docs/experiment_results_context.md`
2. `PROGRESS.md` 中对应时间段
3. `scripts/legacy/README.md`
4. `scripts/legacy/bidir/` 或 `scripts/legacy/wrongsample/`
5. 对应 `collab/bidirectional.py`、`collab/mutual.py`、`collab/bdeeg.py`

网络结构文件可以最后读，因为它们解释的是单个模型内部如何产生特征，而不是项目如何组织实验。

## 13. 最容易混淆的点

```text
checkpoint != artifact
```

- checkpoint：模型权重，可以继续训练/推理。
- artifact：模型对固定样本导出的 logits/features/y，用来协同和分析。

```text
collab != experiments != scripts
```

- `collab/`：算法库，应该可 import、可复用、少副作用。
- `experiments/`：实验入口，负责 protocol、condition、遍历和落盘。
- `scripts/`：工具入口，负责自检、导出和归档复现。

```text
离线蒸馏 != 双向蒸馏
```

- 离线蒸馏：teacher 先导出 artifact，student 单向学习，teacher 不更新。
- 双向蒸馏：两个模型同时训练，同时更新，动态互教。

```text
模型代码在仓库内 != 所有模型能在同一环境同时跑
```

代码已经 vendored 到 `models/`，但依赖仍可能冲突。因此 artifact hub 仍然是项目的核心设计。

```text
新增实验优先 YAML，不优先新脚本
```

- 矩阵实验：新增 `configs/exp/*.yaml`，必要时扩展 `experiments/methods.py`。
- 临时专门 driver：放 `experiments/<line>/`。
- 导出、自检、批处理：才放 `scripts/`。

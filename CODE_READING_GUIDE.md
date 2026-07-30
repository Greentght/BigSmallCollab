# BigSmallCollab 代码阅读指南

这份文档按项目真实流水线介绍主要文件的作用。先记住一句话：

```text
统一切数据 -> 每个模型用 adapter 微调 -> 导出 artifact -> 协同模块读取 artifact 或在线共同训练 -> eval 做统计
```

## 1. 总体分层

```text
configs/      数据集、模型、实验 YAML 配置
config.py     读取 YAML 配置
paths.py      解析预训练权重路径

data/         数据加载、通道表、预处理、统一切分
models/       模型结构 + adapter 统一接入层
scripts/      常用实验入口脚本
collab/       协同算法：集成、蒸馏、双向蒸馏、特征融合
eval/         acc/kappa 和 subject 级统计检验
experiments/  更规范的 YAML 驱动实验 runner
results/      输出目录：artifacts 和 metrics
```

核心设计是：模型代码已经在本仓库 `models/` 里，但不同大模型依赖不兼容，所以训练仍可在各自 conda 环境里跑；协同阶段通过标准化 artifact 解耦。

## 2. 配置入口

先看这几个文件：

- `config.py`：提供 `load_dataset_config(name)` 和 `load_model_config(name)`。
- `configs/datasets/*.yaml`：定义数据集类别数、通道数、被试数、默认 seeds、`val_split`。
- `configs/models/*.yaml`：定义每个模型的 env、大小模型类型、epochs、lr、batch_size、weight_decay 等。
- `paths.py`：解析预训练权重。权重默认走 `weights/*.pth`，也可以用环境变量覆盖，例如 `MIREPNET_WEIGHT`、`CBRAMOD_WEIGHT`、`LABRAM_WEIGHT`。

这里要注意：`val_split` 在项目里表示测试集比例，例如 `0.3` 是 70% 校准/训练，30% 测试。

## 3. 数据流水线

数据相关文件：

- `data/eeg_dataset.py`：从 `/data1/llx/<DATASET>/` 读取原始 `.npy` 数据和标签，并按数据集规则选择 session、subject、类别。
- `data/split.py`：统一产生训练/测试切分，是跨模型样本顺序对齐的唯一来源。
- `data/preproc.py`：提供 EA、通道补齐、bandpass、notch 等基础预处理函数。
- `data/channels.py`：保存各数据集通道名和 scalp 位置，用于通道映射和补齐。

最重要的是 `data/split.py`：

```text
data.subject_split(dataset, subject, val_split, seed)
  -> 返回 X_tr, y_tr, X_te, y_te

data.loso_split(dataset, test_subject)
  -> 返回 X_tr, y_tr, subj_tr, X_te, y_te
```

所有模型必须使用这里生成的 split。否则后面的集成和蒸馏会按行错位，因为 artifact 里的第 i 行必须对应同一个样本。

## 4. 模型接入层：Adapter

模型统一接口在 `models/base.py`。

每个模型 adapter 都实现三件事：

```text
preprocess(X_raw) -> 模型输入张量
build(num_classes) -> nn.Module
forward(model, x) -> feat, logits
```

然后 `ModelAdapter` 基类提供：

```text
finetune(...)
infer(...)
export(...)
mc_uncertainty(...)
```

模型注册在 `models/__init__.py` 的 `get_adapter(name, ...)`。

各模型主要差异：

- `models/ifnet/adapter.py`：小模型，基本吃原始 `(B,C,T)`。
- `models/eegnet/adapter.py`：小模型，调用 `ResidualEEGNet`。
- `models/adfcnn/adapter.py`：小模型，调用 `ADFCNN_Net`。
- `models/mirepnet/adapter.py`：大模型，先做 EA + 45 通道补齐，再进入 `models/mirepnet/mlm.py`。
- `models/cbramod/adapter_native.py`：大模型 CBraMod（唯一实现，final settled native/tuned 版；`--model cbramod` 即走这里，`cbramod_native` 为兼容别名）。
- `models/labram/adapter.py`：LaBraM，250Hz 转 200Hz patchify，并计算 `input_chans` 通道映射。

网络结构文件本身可以后看：

```text
models/ifnet/ifnet.py
models/eegnet/residual_eegnet.py
models/adfcnn/adfcnn.py
models/mirepnet/mlm.py
models/cbramod/cbramod.py
models/labram/modeling_finetune.py
```

理解项目流水线时，优先看 adapter，而不是先钻进完整网络结构。

## 5. 微调并导出 Artifact

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
  -> collab/artifacts.py 保存 .npz
```

典型命令：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py   --model ifnet   --dataset BNCI2014004   --gpu 0
```

大模型按各自环境跑：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py --model mirepnet --dataset BNCI2014004 --gpu 0
conda run -n cbramod  python scripts/export/finetune_export.py --model cbramod  --dataset BNCI2014004 --gpu 1
conda run -n labram   python scripts/export/finetune_export.py --model labram   --dataset BNCI2014004 --gpu 2
```

只跑部分 subject/seed：

```bash
conda run -n mirepnet python scripts/export/finetune_export.py   --model ifnet   --dataset BNCI2014004   --subjects 0 1   --seeds 666   --gpu 0
```

## 6. Artifact 是什么

artifact 不是 checkpoint。它是模型对固定样本导出的标准化中间产物，保存位置由 `collab/artifacts.py` 管：

```text
results/artifacts/<dataset>/<model>/<subject>_<seed>_<split>.npz
```

每个 `.npz` 里有：

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

## 7. 测试时集成协同

入口：`scripts/fusion/run_ensemble.py`

它不加载模型，只读取 `test` artifact：

```text
scripts/fusion/run_ensemble.py
  -> artifacts.load_aligned(dataset, models, subject, seed, 'test')
  -> collab/ensemble.py
  -> eval/metrics.py
  -> results/metrics/<dataset>_ensemble.csv
```

核心方法在 `collab/ensemble.py`：

- `gate(...)`：大模型置信度门控，决定大模型和小模型平均概率的混合比例。
- `conf_weighted(...)`：所有模型按各自置信度加权。
- `voting(...)`：硬投票。

命令示例：

```bash
python scripts/fusion/run_ensemble.py   --dataset BNCI2014004   --models mirepnet ifnet eegnet adfcnn   --big mirepnet
```

## 8. 离线蒸馏协同

入口：`scripts/distill/run_distill.py`

这条线是单向的：teacher 已训练并导出 artifact，student 读取 teacher 的 logits/features 训练。teacher 不会更新。

流程：

```text
teacher 先 finetune_export.py 导出 train artifact
  -> run_distill.py 读取 teacher train artifact
  -> data.subject_split(...) 重新拿同一份训练/测试数据
  -> assert teacher['y'] == y_tr
  -> get_adapter(student)
  -> collab.distill.distill_student(...)
  -> 写 results/metrics/*.csv
```

核心训练函数：`collab/distill.py` 的 `distill_student(...)`。

基本损失：

```text
loss = CE(student, y)
     + lam_kd   * KL(student logits, teacher logits)
     + lam_feat * cosine_align(project(student feat), teacher feat)
```

典型命令：

```bash
# 1. 先导出 teacher artifact
conda run -n mirepnet python scripts/export/finetune_export.py   --model mirepnet   --dataset BNCI2014004   --gpu 0

# 2. 再蒸馏 student
conda run -n mirepnet python scripts/distill/run_distill.py   --dataset BNCI2014004   --teacher mirepnet   --student ifnet   --lam_kd 0.5   --lam_feat 0.5   --gpu 0
```

更规范的 YAML 驱动入口是 `experiments/run.py`：

```bash
conda run -n mirepnet python -m experiments.run configs/exp/distill_kd_within.yaml --report
```

相关文件：

- `experiments/protocols.py`：生成 within/loso 的 cell。
- `experiments/methods.py`：把 YAML condition 映射成 `distill_student` 参数。
- `experiments/run.py`：按 dataset/unit/seed/condition 循环运行。

## 9. LOSO 和跨被试协同

LOSO 是 Leave-One-Subject-Out：留一个被试做测试，其余被试训练。

数据入口：

```text
data.loso_split(dataset, test_subject)
```

常用脚本：

- `scripts/export/export_preds.py`：统一导出 within 或 LOSO artifact。LOSO 时 artifact model 名会变成 `<model>_loso`。
- `scripts/export/export_teacher_loso.py`：专门导出 `mirepnet_loso` teacher。MIRepNet 在 LOSO 中需要按 subject 分别 EA，再合并训练。
- `scripts/distill/run_loso_distill.py`：读取 `mirepnet_loso`，训练 student 做跨被试蒸馏。

示例：

```bash
conda run -n mirepnet python scripts/export/export_teacher_loso.py   --dataset BNCI2014004   --gpu 0

conda run -n mirepnet python scripts/distill/run_loso_distill.py   --dataset BNCI2014004   --student ifnet   --gpu 0
```

## 10. 真正双向蒸馏

这里要和 artifact 离线蒸馏区分开。

离线 artifact 蒸馏是：

```text
teacher 冻结 -> student 学 teacher
```

真正双向蒸馏是：

```text
两个模型在同一个训练过程中同时 forward、同时更新，并动态决定谁教谁
```

相关文件：

- `scripts/bidir/run_bidir_loso.py` -> `collab/bidirectional.py`
- `scripts/bidir/run_cramd_loso.py` -> `collab/mutual.py`
- `scripts/bidir/run_bdeeg_loso.py` -> `collab/bdeeg.py`

最直接命令：

```bash
conda run -n mirepnet python scripts/bidir/run_bidir_loso.py   --dataset BNCI2014001-4   --gpu 0   --epochs 100
```

`run_bidir_loso.py` 中：

```text
Uni    只做 B -> S
Bidir  做 B -> S 和 S -> B
```

`collab/bidirectional.py` 的核心逻辑：

```text
每个 batch:
  B = MIRepNet forward
  S = IFNet forward

  如果 B 对、S 错: B -> S KD
  如果 S 对、B 错: S -> B KD

  loss = CE_B + CE_S + routed KD terms
```

更完整的互学习实验：

```bash
conda run -n mirepnet python scripts/bidir/run_cramd_loso.py   --dataset BNCI2014001-4   --gpu 0
```

它会比较：

```text
G0_CE       两个模型各自 CE 训练，无蒸馏
G1_FixKD    大模型冻结，传统 B -> S
G3_SymDML   双向全样本互蒸馏
G5_Routed   B 对 S 错时 B -> S
G6_CRAMD    B -> S 和 S -> B 都做，但按互补样本路由
G8/G9       特征级双向对齐相关组
```

限制：真正双向蒸馏要求两个模型能在同一个 conda 环境、同一个 Python 进程中同时加载。目前最自然的是 `MIRepNet <-> IFNet`，因为都能在 `mirepnet` 环境里跑。CBraMod/LaBraM 这类依赖冲突模型更适合先走 artifact 解耦。

## 11. 特征融合协同

特征融合不是重新训练 backbone，而是在 frozen artifact 的 features 上训练轻量 head。

相关文件：

- `collab/fusion.py`：实现 `head_single`、`fusion_concat`、`fusion_gated`、`fusion_mutual`。
- `scripts/fusion/run_ft_fusion.py`：LOSO few-shot 场景下，在 test subject 的 K 个标注样本上训练轻量 head。
- `scripts/fusion/run_finetune_baseline.py`：对照实验，比较 frozen feature fusion 和真正 end-to-end fine-tune。
- `scripts/bidir/run_featbidir_fewshot.py`：few-shot 下的 feature-level bidirectional alignment。

核心问题是：融合是否比“只适配大模型 head”或“只适配小模型 head”更好。

## 12. 评估与统计

基础指标：`eval/metrics.py`

```text
evaluate(y_true, y_pred) -> acc, kappa
preds_from_logits(logits)
per_class(y, pred, num_classes)
```

正式统计：`eval/stats.py`

它会按 subject/fold 作为配对单位，先把 seeds 聚合成 per-unit mean，然后做：

```text
paired Wilcoxon
Holm correction
bootstrap CI
```

命令入口：

```bash
python -m eval 'results/metrics/*.csv' --baseline ifnet_base
```

理解实验结论时不要只看 raw mean，小提升要看 subject 级配对统计。

## 13. 推荐阅读顺序

建议按这个顺序读：

1. `README.md`
2. `config.py`、`configs/datasets/*.yaml`、`configs/models/*.yaml`
3. `data/split.py`
4. `data/eeg_dataset.py`
5. `models/base.py`
6. `models/__init__.py`
7. 一个小模型 adapter，例如 `models/ifnet/adapter.py`
8. 一个大模型 adapter，例如 `models/mirepnet/adapter.py`
9. `scripts/export/finetune_export.py`
10. `collab/artifacts.py`
11. `scripts/fusion/run_ensemble.py`、`collab/ensemble.py`
12. `scripts/distill/run_distill.py`、`collab/distill.py`
13. `experiments/run.py`、`experiments/protocols.py`、`experiments/methods.py`
14. LOSO/双向蒸馏脚本：`run_bidir_loso.py`、`run_cramd_loso.py`、`run_bdeeg_loso.py`
15. `eval/metrics.py`、`eval/stats.py`

网络结构文件可以最后读，因为它们解释的是单个模型内部怎么算特征，而不是项目怎么组织实验。

## 14. 最容易混淆的点

```text
checkpoint != artifact
```

- checkpoint：模型权重，可以继续训练/推理。
- artifact：模型对固定样本导出的 logits/features/y，用来协同和分析。

```text
离线蒸馏 != 双向蒸馏
```

- 离线蒸馏：teacher 先导出 artifact，student 单向学习，teacher 不更新。
- 双向蒸馏：两个模型同时训练，同时更新，动态互教。

```text
模型代码在仓库内 != 所有模型能在同一环境同时跑
```

代码已经 vendored 到 `models/`，但依赖仍可能冲突。因此 artifact hub 仍然是项目的核心设计。

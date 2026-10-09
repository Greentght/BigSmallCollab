# BigSmallCollab — 大小模型协同框架 (MI-BCI)

维护运动想象解码中的 **大模型实现 (MIRepNet / CBraMod / LaBraM / CodeBrain)**、
**小模型实现 (IFNet / EEGNet / ADFCNN)** 和协同实验工具。

CodeBrain 的独立复现已完成，但在当前被试内 few-shot 协议下接近机会水平；按项目当前决定，它不纳入活动协同 baseline 或 teacher 集合。实现和结果作为负结果记录保留，见 [CodeBrain 复现与结果记录](docs/codebrain_reproduction.md)。

## 核心思想：缓存产物 Hub（解耦不兼容的 conda 环境）

每个模型依赖各异（MIRepNet 的 numpy/mne pin vs LaBraM 的 timm0.4.12），无法共处一个
环境。所以：

1. **每个模型在自己的 env 里 finetune**，对每个 `(dataset, subject, seed, split)`
   导出标准化产物 `{logits, feats, y}`（`/data1/llx/BigSmallcollab/results/artifacts/`）。
2. **Hub 消费产物**做协同（任意 env，纯数组）：
   - **离线 KD + 特征对齐** — student 对着**冻结的缓存 teacher** feats/logits 训练，
     大模型无需在线。（测试时集成/融合线已归档，见 git tag `pre-consolidation`。）

**关键不变量**：同一 `(dataset, subject, seed, split)` 下所有模型样本顺序一致
（由 `data/split.py` 统一切分保证），否则按行对齐的集成/蒸馏会错位 —
`artifacts.load_aligned` 会用存储的 `y` 校验。

## 目录

```
data/      一处管数据: eeg_dataset(加载) · split(规范切分) · preproc(EA/通道padding/滤波) · channels(montage)
models/    一模型一文件夹(网络定义 + 适配器 co-located):
           base(ModelAdapter 契约 + 小模型基类 + registry)
           ifnet/ · eegnet/ · adfcnn/ · mirepnet/(mlm+lora/mmd) · cbramod/(criss-cross + adapter) · labram/(+optim_factory/montage) · codebrain/(EEGSSM + adapter)
           每个文件夹 = <net>.py + adapter.py;加模型 = 加一个文件夹
collab/    distill(离线KD+特征对齐) · bidirectional/mutual/bdeeg(双向线) · seed(统一播种) · artifacts(跨环境产物 hub)
eval/      stats(subject级配对 Wilcoxon+Holm+bootstrap CI,acc%优先) · metrics(acc/kappa/per_class)
experiments/ 实验入口: finetune/ · distill/ · fusion/ · bidir/ · mask/ drivers
config.py  数据/模型 yaml 加载 + 权重解析(weight_path)
scripts/   自检工具: check/(smoke/verify)
configs/   datasets/*.yaml(数据事实+统一split) · models/*.yaml(结构+finetune超参) · experiments/*.yaml(distill/fusion实验配方)
envs/      各模型 conda 环境说明
```

**完全自包含(2026-07-25):** 所有模型**代码**都 vendored 进框架,不再引用任何外部仓
(`sys.path.add_repo` 已全部移除):
- 数据层 `data/`(`eeg_dataset` `EEGDataset` / `preproc` EA+通道padding+滤波 / `channels` / `split`)
  与小模型(`models/{ifnet,eegnet,adfcnn}`)—— 逐位一致,见 `scripts/verify_foundation.py`。
- 大模型 **backbone** 与**微调代码**都在各自的 `models/<name>/` 里 —— `mirepnet`(`mlm` + PEFT
  `lora`/`mmd`)、`cbramod`(criss-cross transformer)、`labram`(`modeling_finetune` + `optim_factory`
  逐层 LR 衰减 + `montage`)、`codebrain`(作者公开 EEGSSM 源码子集 + 适配器)。三个既有大模型 build+forward 见 `scripts/verify_backbones.py`。
- 预训练**权重**(非代码)真文件统一存放在 `/data1/llx/pre_weight/`(稳定数据盘,
  与上游仓解耦——删掉 ~/MIRepNet 等不受影响)，
  由 `config.weight_path()` 解析,可用 `MIREPNET_WEIGHT` / `CBRAMOD_WEIGHT` /
  `LABRAM_WEIGHT` 环境变量覆盖(如指向新微调的 checkpoint)。CodeBrain 权重默认是
  `/data1/llx/BigSmallcollab/weights/codebrain.pth`,可用 `CODEBRAIN_WEIGHT` 覆盖；输入适配、官方权重版本和复现实验命令见
  [CodeBrain 复现与结果记录](docs/codebrain_reproduction.md)。

**存储位置：** 共享数据集独立存放在 `/data1/llx/<数据集名称>/`；本项目专用文件统一放在 `/data1/llx/BigSmallcollab/`，工作树只保存代码、配置和文档。

| 内容 | 路径 |
|---|---|
| 共享数据集 | `/data1/llx/BNCI2014001/`、`BNCI2014004/`、`BNCI2015001/`、`AlexMI/` |
| 新 14001 全场次宽带 NPY | `/data1/llx/BNCI2014001/broadband_0p1_75hz/` |
| 项目模型输入及协议缓存 | `/data1/llx/BigSmallcollab/cache/` |
| 实验结果、预测、教师缓存和训练 checkpoint | `/data1/llx/BigSmallcollab/results/` |
| CodeBrain 等外置权重 | `/data1/llx/BigSmallcollab/weights/` |
| 现有 MIRepNet／CBraMod／LaBraM 预训练权重 | `/data1/llx/pre_weight/` |
| Git LFS 本地对象 | `/data1/llx/BigSmallcollab/git_lfs/` |

完整目录规则见 [共享数据集与项目文件存储](docs/storage_layout.md)。

`experiments/storage.py` 将历史 `results/...`、`data_cache/...`、`weights/...` 路径映射到上述目录；
实验写入接口拒绝把数据或模型保存进项目工作树。

各大模型仍需在**自己的 conda 环境**里跑(依赖不兼容:MIRepNet 的 numpy/mne pin vs LaBraM 的
timm0.4.12 vs CBraMod 的 einops);框架靠 artifact hub 解耦——见下。

## 适配器契约 (`models/base.py`)

每个模型用一个 `ModelAdapter` 包装，对外统一：
- `preprocess(X_raw (N,C,1000)@250Hz) -> 模型输入`（各自拥有预处理）
- `build(num_classes) -> nn.Module`（含加载预训练）
- `forward(model, x) -> (feat[B,D], logits[B,C])`
- `finetune` / `infer` / `export` 由基类提供。

各模型预处理差异：MIRepNet → EA + 45ch pad；CBraMod → **EA + 45ch pad + 250→200Hz**（最终版，见 PROGRESS 07-01 终表）；
LaBraM → 250→200Hz resample + patchify `(ch,4,200)`（另需 `input_chans` 通道映射）；小模型 → 原样。

## 用法

```bash
# 1) 各模型在自己 env 里 finetune + 导出产物（train+test 两个 split）
conda run -n mirepnet python experiments/finetune/finetune.py --protocol fewshot --model ifnet    --dataset BNCI2014004
conda run -n mirepnet python experiments/finetune/finetune.py --protocol fewshot --model mirepnet --dataset BNCI2014004
conda run -n cbramod  python experiments/finetune/finetune.py --protocol fewshot --model cbramod  --dataset BNCI2014004 --gpu 1
conda run -n labram   python experiments/finetune/finetune.py --protocol fewshot --model labram   --dataset BNCI2014004 --gpu 1

# 2) 离线蒸馏（在 student env 里运行；teacher 产物须已导出）
conda run -n mirepnet python experiments/distill/run_distill.py \
    --config configs/experiments/distill_kd.yaml --gpu 1

# 3) artifact 融合（当前统一版）
conda run -n mirepnet python experiments/fusion/run_fusion.py \
    --config configs/experiments/fusion_concat_mlp.yaml --gpu 1
```

数据集配置只记录数据事实、seeds 和统一 split；模型在不同数据集/协议上的微调超参
写在 `configs/models/<model>.yaml` 的 `finetune.<dataset>.<protocol>` 下。源数据在
`/data1/llx/<DATASET>/`。`val_split` 是**测试**比例（0.3 = 70%校准/30%测试）。

CodeBrain 保留专用审计 runner `experiments/finetune/run_codebrain.py` 以便复现；当前不属于活动协同 baseline 或 teacher。已完成的三 seed 结果和负结果结论见 [CodeBrain 结果报告](docs/codebrain_results.md)。

## 冒烟自检

```bash
# 地基（数据+小模型逐位一致）
conda run -n mirepnet python scripts/verify_foundation.py
# 大模型 backbone（vendored 代码 + 权重 build+forward）
conda run -n mirepnet python scripts/verify_backbones.py --model mirepnet
conda run -n cbramod  python scripts/verify_backbones.py --model cbramod
conda run -n labram   python scripts/verify_backbones.py --model labram
# 适配器端到端
conda run -n mirepnet python scripts/smoke_test.py --models ifnet eegnet adfcnn mirepnet
```

## 实验与工具入口

experiments/ 是正式实验层。每个协同方法一个 YAML，配置文件位于
configs/experiments/：

- distill：distill_kd.yaml、distill_kd_masked.yaml、distill_mmd.yaml、distill_kd_mmd.yaml、distill_mi.yaml
- fusion：fusion_avg_prob.yaml、fusion_concat_mlp.yaml、fusion_gate_conf_acc.yaml

同一个 YAML 通过 pairs 为每个已选大模型/小模型组合分别配置 params 和 grid；
训练超参也可作为 params/grid 的一部分写入。datasets.<dataset> 可以在 pair 内覆盖参数，
优先级为模型默认值 → pair 参数 → dataset 覆盖 → grid 当前组合。Distill 会自动附加同训练配置的 student
scratch baseline（可用 include_baseline: false 关闭）；fusion 会自动评估
big_only 和 small_only（可用 include_controls: false 关闭）。

experiments/distill/run_distill.py — YAML 蒸馏入口
- 使用 --config configs/experiments/distill_<method>.yaml。
- method 是单一方法：kd / kd_masked / mmd / kd_mmd；Base 不单独建 YAML。
- distill_mi.yaml 仅运行 subject-wise few-shot，固定展开 Base、KD_all、CE_MI；CE+MI 的 lam_mi=0.1 是 pilot 值。
- 支持 fewshot / loso，并写出 <name>.resolved.yaml。

experiments/fusion/run_fusion.py — YAML 融合入口
- 使用 --config configs/experiments/fusion_<method>.yaml。
- method 是 avg_prob / concat_mlp / gate_conf_acc；controls 自动附加。
- avg_prob 支持 pair 级 big_temperature、small_temperature、big_weight。

两个 runner 的正式 CLI 只保留 --config --gpu --resume --force --fail-fast；旧逐参数 CLI
暂时保留用于历史复现，并标记为 deprecated。

`scripts/` 是工具箱，不承载正式实验矩阵。

**`scripts/` — 自检 / 冒烟**（改完代码先跑）
- `verify_foundation.py` — 数据层+小模型 vendoring 逐位一致
- `verify_backbones.py --model {mirepnet,cbramod,labram}` — 大模型 backbone build+forward
- `smoke_test.py` — 数据切分 + 适配器 forward 契约

**`experiments/finetune/` — 单模型微调 + 导出产物**（一切协同实验的共享前置：先把 teacher/student 产物缓存出来）
- 当前 LOSO baseline 以 [正式配置说明](docs/loso_baseline.md) 为准：参数在 `configs/models/*.yaml::finetune.<dataset>.loso`，来源在 `configs/datasets/*.yaml::loso`。004/5001 使用 `configs/protocols/loso.yaml`，001/001-4 使用 `configs/protocols/loso_001.yaml`；四组蒸馏使用 `configs/experiments/loso_distillation.yaml`。
- `finetune.py --model <m> --dataset <ds> --protocol fewshot` — session 内微调并导出标准产物。当前宽带 LOSO 任务使用上面的专用 baseline 入口。
- `finetune_teacher_mc.py` — teacher + MC-dropout 不确定度
- `finetune_teacher_loso.py` — LOSO 逐 fold teacher 微调
- `finetune_teacher_loso_subjoof.py` — LOSO subject-OOF 交叉拟合 teacher

**`experiments/bidir/`、`experiments/mask/` — 负结果复现**
- `experiments/bidir/` — 双向互蒸馏 / CR-AMD / BD-EEG / feature-level mutual（全线判 null，留作复现）
- `experiments/mask/run_wrong_sample.py` — wrong-sample 利用 E0-E5（closed；仅 E2 correct-only KD 存活）

**`scripts/legacy/` — 早期一次性编排脚本**（已归档到 tag `archive-legacy-scripts`）
- `run_*.sh` — 各实验线的规范启动器；取回:`git checkout archive-legacy-scripts -- scripts/legacy`

## 数据流与关键概念

```text
原始数据 -> data/split.py(规范切分) -> models/<name>/adapter.py(预处理/前向)
  -> experiments/finetune/*.py(微调+导出工件 .npz) -> experiments/*(编排) -> collab/*(算法) -> eval/*(统计) -> /data1/llx/BigSmallcollab/results/
```

几个最容易混淆的点：

- **checkpoint ≠ artifact**：checkpoint 是权重；artifact 是模型对固定样本导出的 `{logits, feats, y}`。
- **离线蒸馏 ≠ 双向蒸馏**：离线 = teacher 冻结、学生单向学工件；双向 = 两模型同进程同时更新互教（要求同 env）。
- **模型代码在仓库内 ≠ 能在同一 env 同时跑**：依赖冲突仍在（MIRepNet 的 numpy/mne pin vs LaBraM 的 timm0.4.12），所以 artifact hub 是核心设计。
- **实验配置职责**：`configs/datasets/*.yaml` 只放数据事实，`configs/models/*.yaml` 放模型默认参数，`configs/experiments/*.yaml` 放 distill/fusion 实验矩阵与 grid；统一 loader 负责校验、合并和展开。

## 范围

- **模型代码全部在框架内**（每模型一个 `models/<name>/` 文件夹，网络定义 + 适配器同处），可直接改/微调；预训练权重默认从 `/data1/llx/pre_weight/*.pth` 读取，也可用环境变量覆盖。
- 各大模型仍在各自 conda env 里 finetune + 导出产物；协同（蒸馏）在任意 env 消费产物。

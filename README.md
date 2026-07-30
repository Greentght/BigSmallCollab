# BigSmallCollab — 大小模型协同框架 (MI-BCI)

统一管理 **大模型 (MIRepNet / CBraMod / LaBraM)** 与 **小模型 (IFNet / EEGNet /
ADFCNN)** 在运动想象解码上的协同实验。模型代码已集成在本仓库的 `models/`
目录下；上游仓库不再作为运行时源码依赖。框架负责统一数据切分、各模型产出、
以及协同（集成 + 蒸馏）。

## 核心思想：缓存产物 Hub（解耦不兼容的 conda 环境）

每个模型依赖各异（MIRepNet 的 numpy/mne pin vs LaBraM 的 timm0.4.12），无法共处一个
环境。所以：

1. **每个模型在自己的 env 里 finetune**，对每个 `(dataset, subject, seed, split)`
   导出标准化产物 `{logits, feats, y}`（`results/artifacts/`）。
2. **Hub 消费产物**做协同（任意 env，纯数组）：
   - **测试时集成** — 置信度门控 (Gate, 主力) / 置信度加权 / 投票，只用 logits。
   - **离线 KD + 特征对齐** — student 对着**冻结的缓存 teacher** feats/logits 训练，
     大模型无需在线。

**关键不变量**：同一 `(dataset, subject, seed, split)` 下所有模型样本顺序一致
（由 `data/split.py` 统一切分保证），否则按行对齐的集成/蒸馏会错位 —
`artifacts.load_aligned` 会用存储的 `y` 校验。

## 目录

```
data/      一处管数据: eeg_dataset(加载) · split(规范切分) · preproc(EA/通道padding/滤波) · channels(montage)
models/    一模型一文件夹(网络定义 + 适配器 co-located):
           base(ModelAdapter 契约 + 小模型基类 + registry)
           ifnet/ · eegnet/ · adfcnn/ · mirepnet/(mlm+lora/mmd) · cbramod/(criss-cross + native head) · labram/(+optim_factory/montage)
           每个文件夹 = <net>.py + adapter.py;加模型 = 加一个文件夹
collab/    ensemble(gate/加权/投票) · distill(离线KD+特征对齐) · bidirectional/… · artifacts(跨环境产物 hub)
eval/      stats(subject级配对 Wilcoxon+Holm+bootstrap CI,acc%优先) · metrics(acc/kappa/per_class)
experiments/ config驱动 runner: protocols(within/loso) · methods(collab registry) · run
config.py  数据/模型 yaml 加载        paths.py  权重解析
weights/   预训练权重 symlink -> /data1/llx/pretrained_weights(*.pth, git忽略)
scripts/   命令行入口(按功能分类见下「脚本导航」);legacy/ 存放已归档的一次性 driver
configs/   datasets/*.yaml · models/*.yaml · exp/*.yaml(实验配方)
results/   artifacts/<ds>/<model>/<subj>_<seed>_<split>.npz · metrics/*.csv
envs/      各模型 conda 环境说明
```

**完全自包含(2026-07-25):** 所有模型**代码**都 vendored 进框架,不再引用任何外部仓
(`sys.path.add_repo` 已全部移除):
- 数据层 `data/`(`eeg_dataset` `EEGDataset` / `preproc` EA+通道padding+滤波 / `channels` / `split`)
  与小模型(`models/{ifnet,eegnet,adfcnn}`)—— 逐位一致,见 `scripts/check/verify_foundation.py`。
- 大模型 **backbone** 与**微调代码**都在各自的 `models/<name>/` 里 —— `mirepnet`(`mlm` + PEFT
  `lora`/`mmd`)、`cbramod`(criss-cross transformer)、`labram`(`modeling_finetune` + `optim_factory`
  逐层 LR 衰减 + `montage`)。三个大模型 build+forward 见 `scripts/check/verify_backbones.py`。
- 预训练**权重**(非代码)真文件统一存放在 `/data1/llx/pretrained_weights/`(稳定数据盘,
  与上游仓解耦——删掉 ~/MIRepNet 等不受影响);`weights/*.pth`(git 忽略)是指向它的 symlink,
  由 `paths.weight_path()` 解析,可用 `MIREPNET_WEIGHT` / `CBRAMOD_WEIGHT` /
  `LABRAM_WEIGHT` 环境变量覆盖(如指向新微调的 checkpoint)。

各大模型仍需在**自己的 conda 环境**里跑(依赖不兼容:MIRepNet 的 numpy/mne pin vs LaBraM 的
timm0.4.12 vs CBraMod 的 einops);框架靠 artifact hub 解耦——见下。

## 适配器契约 (`models/base.py`)

每个模型用一个 `ModelAdapter` 包装，对外统一：
- `preprocess(X_raw (N,C,1000)@250Hz) -> 模型输入`（各自拥有预处理）
- `build(num_classes) -> nn.Module`（含加载预训练）
- `forward(model, x) -> (feat[B,D], logits[B,C])`
- `finetune` / `infer` / `export` 由基类提供。

各模型预处理差异：MIRepNet → EA + 45ch pad；CBraMod/LaBraM → 250→200Hz resample +
patchify `(ch,4,200)` + µV/100（LaBraM 另需 `input_chans` 通道映射）；小模型 → 原样。

## 用法

```bash
# 1) 各模型在自己 env 里 finetune + 导出产物（train+test 两个 split）
conda run -n mirepnet python scripts/export/finetune_export.py --model ifnet    --dataset BNCI2014004
conda run -n mirepnet python scripts/export/finetune_export.py --model mirepnet --dataset BNCI2014004
conda run -n cbramod  python scripts/export/finetune_export.py --model cbramod  --dataset BNCI2014004 --gpu 1
conda run -n labram   python scripts/export/finetune_export.py --model labram   --dataset BNCI2014004 --gpu 1

# 2) 测试时集成（任意 env）
python scripts/fusion/run_ensemble.py --dataset BNCI2014004 \
    --models mirepnet cbramod labram ifnet adfcnn eegnet --big mirepnet cbramod labram

# 3) 离线蒸馏（在 student 的 env 里跑；teacher 产物须已导出）
conda run -n mirepnet python scripts/distill/run_distill.py \
    --dataset BNCI2014004 --teacher cbramod --student ifnet --lam_kd 0.5 --lam_feat 0.5
```

数据集：`BNCI2014004` (3ch/2类)、`BNCI2014001-4` (22ch/4类)，源数据在
`/data1/llx/<DATASET>/`。`val_split` 是**测试**比例（0.3 = 70%校准/30%测试）。

## 冒烟自检

```bash
# 地基（数据+小模型逐位一致）
conda run -n mirepnet python scripts/check/verify_foundation.py
# 大模型 backbone（vendored 代码 + 权重 build+forward）
conda run -n mirepnet python scripts/check/verify_backbones.py --model mirepnet
conda run -n cbramod  python scripts/check/verify_backbones.py --model cbramod
conda run -n labram   python scripts/check/verify_backbones.py --model labram
# 适配器端到端
conda run -n mirepnet python scripts/check/smoke_test.py --models ifnet eegnet adfcnn mirepnet
```

## 脚本导航 (`scripts/`)

`scripts/` 是命令行入口层（可复现的实验 driver），**按方案分成 7 个子目录**。前三个是共享
基础设施（日常主要用它们）；后四个是各条实验方案，其中 `bidir/` 与 `wrongsample/` 已判 null
（保留供复现，结论见 `PROGRESS.md` / memory）。每个脚本 `--help` 或文件首行 docstring 有详细说明。
被多个方案共用的脚本，放在它首次出现的基础目录里（如 `export/finetune_export.py`）。

**`check/` — 自检 / 冒烟**（改完代码先跑）
- `verify_foundation.py` — 数据层+小模型 vendoring 逐位一致
- `verify_backbones.py --model {mirepnet,cbramod,labram}` — 大模型 backbone build+forward
- `smoke_test.py` — 数据切分 + 适配器 forward 契约

**`export/` — 微调 + 导出产物**（一切协同实验的共享前置：先把 teacher/student 产物缓存出来）
- `finetune_export.py --model <m> --dataset <ds>` — 微调单个模型并导出标准产物（**主入口**）
- `export_preds.py` — 统一的逐样本预测/特征导出（within + LOSO）
- `export_teacher_mc.py` — teacher + MC-dropout 不确定度
- `export_teacher_loso.py` — LOSO 逐 fold teacher 微调
- `run_c_export.sh` / `run_d0_export.sh` — 批量补齐 BNCI2015001/AlexMI、D0 逐样本产物队列

**`bigmodel/` — 大模型原生适配 & 调参**（MIRepNet / CBraMod / LaBraM 的忠实 native pipeline）
- `cbramod_native_adapt.py` · `labram_native_adapt.py` — native 预处理下游适配；CBraMod 支持 `--protocol loso`
- `mirepnet_loso_adapt.py` — MIRepNet 端到端 LOSO 评估（per-subject EA + 45ch pad）
- `tune_mirepnet_loso.py` · `tune_cbramod_loso.py` — LOSO 场景聚焦网格调参（1 seed search + 3 seed confirm）
- `tune_cbramod_native.py` · `tune_labram_native.py` — 逐(数据集,split)超参调参
- `tune_cbramod_004_caronly.py` — CBraMod 在 BNCI2014004 的 CAR-only 精调

**`distill/` — 蒸馏方案**
- `run_distill.py` — 离线 KD + 特征对齐蒸馏（**主蒸馏入口**）
- `run_loso_distill.py` — LOSO 逐 fold 学生蒸馏

**`fusion/` — 集成 / 融合 / 路由（结果主线）**
- `run_ensemble.py` — 测试时集成（消费缓存产物，任意 env）
- `run_ft_fusion.py` — F+T 少样本特征融合主实验（结论已更正，见 `PROGRESS.md`）
- `run_balance_gate.py` — balance-gated 少样本选择（融合线的 corrected main line）
- `run_finetune_baseline.py` — 端到端微调基线（F+T 的关键对照）
- `run_r1_signal.py` — 学习式 logit 路由信号排查（负结果）

**`bidir/` — 双向互蒸馏方案（全线判 null，留作复现）**
- `run_bidir_loso.py` · `run_bidir_fewshot.py` · `run_cramd_loso.py` · `run_bdeeg_loso.py`
  — 各类双向互蒸馏（LOSO / 少样本 / CR-AMD / BD-EEG）
- `run_featbidir_fewshot.py` · `run_featbidir_within.py` — 特征级双向对齐
- `loso_pred_states.py` — LOSO 逐 fold 预测状态诊断工具

**`wrongsample/` — wrong-sample 方案（判 null）**
- `run_wrong_sample.py` — wrong-sample 利用 E0–E5（closed；仅 E2 correct-only KD 存活）

> 更早的一次性 `run_*.sh` 编排和 `analyze_*`/`aggregate_*` 分析脚本已归档在
> `scripts/legacy/`（见其 `README.md`），被 `experiments/`（config 驱动 runner）和
> `eval/`（统一配对统计）取代。

## 范围

- **模型代码全部在框架内**（每模型一个 `models/<name>/` 文件夹，网络定义 + 适配器同处），可直接改/微调；预训练权重通过 `weights/*.pth` symlink 或环境变量外置管理。
- 各大模型仍在各自 conda env 里 finetune + 导出产物；协同（集成/蒸馏）在任意 env 消费产物。
- 特征级门控融合 (`fusion_model.DualBranchFusion`) 仅在单 env 同进程下可用，作为可选 v2。

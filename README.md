# BigSmallCollab — 大小模型协同框架 (MI-BCI)

统一管理 **大模型 (MIRepNet / CBraMod / LaBraM)** 与 **小模型 (IFNet / EEGNet /
ADFCNN)** 在运动想象解码上的协同实验。

## 核心思想：缓存产物 Hub（解耦不兼容的 conda 环境）

每个模型依赖各异（MIRepNet 的 numpy/mne pin vs LaBraM 的 timm0.4.12），无法共处一个
环境。所以：

1. **每个模型在自己的 env 里 finetune**，对每个 `(dataset, subject, seed, split)`
   导出标准化产物 `{logits, feats, y}`（`results/artifacts/`）。
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
           ifnet/ · eegnet/ · adfcnn/ · mirepnet/(mlm+lora/mmd) · cbramod/(criss-cross + adapter) · labram/(+optim_factory/montage)
           每个文件夹 = <net>.py + adapter.py;加模型 = 加一个文件夹
collab/    distill(离线KD+特征对齐) · bidirectional/mutual/bdeeg(双向线) · seed(统一播种) · artifacts(跨环境产物 hub)
eval/      stats(subject级配对 Wilcoxon+Holm+bootstrap CI,acc%优先) · metrics(acc/kappa/per_class)
experiments/ 正式实验入口: config runner(protocols/methods/run) · distill/ bigmodel/ bidir/ mask/ drivers
config.py  数据/模型 yaml 加载        paths.py  权重解析
weights/   预训练权重 symlink -> /data1/llx/pretrained_weights(*.pth, git忽略)
scripts/   工具入口: check/ · export/ · legacy/(已归档/负结果复现实验)
configs/   datasets/*.yaml · models/*.yaml · exp/*.yaml(实验配方)
results/   artifacts/<ds>/<model>/<subj>_<seed>_<split>.npz · metrics/*.csv
envs/      各模型 conda 环境说明
```


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

# 2) 离线蒸馏（在 student 的 env 里跑；teacher 产物须已导出）
conda run -n mirepnet python experiments/distill/run_distill.py \
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

## 实验与工具入口

`experiments/` 是正式实验层。新增协同方案优先写 `configs/exp/*.yaml` 并走
`python -m experiments.run <yaml> --report`；如果暂时需要专门 driver，也放在
`experiments/<line>/`，不要再新增到 `scripts/`。

**`experiments/run.py` — config-driven 主入口**
- `protocols.py` — within / LOSO cell 生成
- `methods.py` — YAML condition 到 `collab.distill.distill_student` kwargs 的 registry
- `run.py` — config -> cells -> conditions -> metrics CSV -> optional report

**`experiments/bigmodel/` — 大模型原生适配 & 调参**
- `cbramod_adapt.py` · `labram_adapt.py` — 下游适配与调参驱动；CBraMod 支持 `--protocol loso`
- `mirepnet_loso_adapt.py` — MIRepNet 端到端 LOSO 评估（per-subject EA + 45ch pad）
- `tune_mirepnet_loso.py` · `tune_cbramod_loso.py` — LOSO 场景聚焦网格调参（1 seed search + 3 seed confirm）
- `tune_cbramod.py` · `tune_labram.py` — 逐(数据集,split)超参调参
- `tune_cbramod_004_caronly.py` — CBraMod 在 BNCI2014004 的 CAR-only 精调

**`experiments/distill/` — 蒸馏实验**
- `run_distill.py` — 离线 KD + 特征对齐蒸馏（历史主蒸馏入口；新实验优先沉到 YAML）
- `run_loso_distill.py` — LOSO 逐 fold 学生蒸馏
- `run_loso_subject_oof_kd.py` — LOSO subject-OOF KD 入口

> D0-onward 线（集成/融合/路由 `experiments/fusion/`、target-support `experiments/adapt/`、
> `eval/d0.py`）已于 2026-08-31 归档，可从 git tag `pre-consolidation` 恢复。

`scripts/` 是工具箱，不承载正式实验矩阵。

**`scripts/check/` — 自检 / 冒烟**（改完代码先跑）
- `verify_foundation.py` — 数据层+小模型 vendoring 逐位一致
- `verify_backbones.py --model {mirepnet,cbramod,labram}` — 大模型 backbone build+forward
- `smoke_test.py` — 数据切分 + 适配器 forward 契约

**`scripts/export/` — 微调 + 导出产物**（一切协同实验的共享前置：先把 teacher/student 产物缓存出来）
- `finetune_export.py --model <m> --dataset <ds>` — 微调单个模型并导出标准产物（**主入口**）
- `export_preds.py` — 统一的逐样本预测/特征导出（within + LOSO）
- `export_teacher_mc.py` — teacher + MC-dropout 不确定度
- `export_teacher_loso.py` — LOSO 逐 fold teacher 微调
- `export_teacher_loso_subjoof.py` — LOSO subject-OOF 交叉拟合 teacher

**`experiments/bidir/`、`experiments/mask/` — 负结果复现**
- `experiments/bidir/` — 双向互蒸馏 / CR-AMD / BD-EEG / feature-level mutual（全线判 null，留作复现）
- `experiments/mask/run_wrong_sample.py` — wrong-sample 利用 E0-E5（closed；仅 E2 correct-only KD 存活）

**`scripts/legacy/` — 早期一次性编排脚本**
- `run_*.sh` — 各实验线的规范启动器（路径已指向 `experiments/`）

## 范围

- **模型代码全部在框架内**（每模型一个 `models/<name>/` 文件夹，网络定义 + 适配器同处），可直接改/微调；预训练权重通过 `weights/*.pth` symlink 或环境变量外置管理。
- 各大模型仍在各自 conda env 里 finetune + 导出产物；协同（蒸馏）在任意 env 消费产物。
- 重跑流程与数据流说明见 [`REPRO.md`](REPRO.md)。

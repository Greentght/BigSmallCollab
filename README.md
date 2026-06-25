# BigSmallCollab — 大小模型协同框架 (MI-BCI)

统一管理 **大模型 (MIRepNet / CBraMod / LaBraM)** 与 **小模型 (IFNet / EEGNet /
ADFCNN)** 在运动想象解码上的协同实验。三个上游仓库被当作**只读模型源**引用
（不复制源码），框架负责数据切分、各模型产出、以及协同（集成 + 蒸馏）。

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
（由 `core/data.py` 统一切分保证），否则按行对齐的集成/蒸馏会错位 —
`artifacts.load_aligned` 会用存储的 `y` 校验。

## 目录

```
core/      paths(repo引用) · data(规范切分) · artifacts(产物存取) · config · metrics
adapters/  base(契约) · small(ifnet/eegnet/adfcnn) · mirepnet · cbramod · labram
collab/    ensemble(gate/加权/投票) · distill(离线KD+特征对齐)
scripts/   finetune_export · run_ensemble · run_distill · smoke_test
configs/   datasets/*.yaml · models/*.yaml
results/   artifacts/<ds>/<model>/<subj>_<seed>_<split>.npz · metrics/*.csv
envs/      各模型 conda 环境说明
```

## 适配器契约 (`adapters/base.py`)

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
conda run -n mirepnet python scripts/finetune_export.py --model ifnet    --dataset BNCI2014004
conda run -n mirepnet python scripts/finetune_export.py --model mirepnet --dataset BNCI2014004
conda run -n cbramod  python scripts/finetune_export.py --model cbramod  --dataset BNCI2014004 --gpu 1
conda run -n labram   python scripts/finetune_export.py --model labram   --dataset BNCI2014004 --gpu 1

# 2) 测试时集成（任意 env）
python scripts/run_ensemble.py --dataset BNCI2014004 \
    --models mirepnet cbramod labram ifnet adfcnn eegnet --big mirepnet cbramod labram

# 3) 离线蒸馏（在 student 的 env 里跑；teacher 产物须已导出）
conda run -n mirepnet python scripts/run_distill.py \
    --dataset BNCI2014004 --teacher cbramod --student ifnet --lam_kd 0.5 --lam_feat 0.5
```

数据集：`BNCI2014004` (3ch/2类)、`BNCI2014001-4` (22ch/4类)，源数据在
`/data1/llx/<DATASET>/`。`val_split` 是**测试**比例（0.3 = 70%校准/30%测试）。

## 冒烟自检

```bash
conda run -n mirepnet python scripts/smoke_test.py --models ifnet eegnet adfcnn mirepnet
conda run -n cbramod  python scripts/smoke_test.py --models cbramod
conda run -n labram   python scripts/smoke_test.py --models labram
```

## 范围

- 上游三仓库只读引用，不改其源码。
- v1 不做 teacher+student 端到端联合微调（需单 env）；用"冻结 teacher + 离线 KD"覆盖。
- 特征级门控融合 (`fusion_model.DualBranchFusion`) 仅在单 env 同进程下可用，作为可选 v2。

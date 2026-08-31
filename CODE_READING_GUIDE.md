# BigSmallCollab 代码阅读指南(2026-08-31 规整后)

一句话记住整个项目:

```text
原始数据 -> data 切分 -> models/adapter 统一模型 -> scripts/export 导出工件(artifact)
  -> experiments 编排实验 -> collab 提供算法 -> eval 统计 -> results/ 落盘
```

核心设计:**大/小模型依赖不兼容,所以大模型只在各自的 conda env 里训练并导出
标准化工件(logits/feats/y);所有协同/蒸馏实验在 mirepnet env 消费工件,
从不加载大模型。** 完整数据流见 [REPRO.md](REPRO.md) 第 0 节。

---

## 1. 目录树总览(当前实际结构)

```text
BigSmallCollab/
├── config.py / paths.py     配置加载 + 预训练权重路径解析
├── REPRO.md                 重跑手册(命令 + 数据流 + 验收标准)★ 跑实验先看它
├── CODE_READING_GUIDE.md    本文件
├── PROGRESS.md              实验日志(追加式,新结果写这里)
│
├── configs/
│   ├── datasets/*.yaml      5 个数据集:类别数/被试数/默认 seeds [666,667,668]/val_split=0.3
│   ├── models/*.yaml        每模型 env、epochs、lr、batch_size、weight_decay
│   └── exp/*.yaml           实验配方(config-driven runner 的输入,目前 distill_kd_within.yaml)
│
├── data/                    数据层(原始数据在 /data1/llx,DATA_ROOT env 可覆盖)
│   ├── eeg_dataset.py       按数据集规则加载 X/labels、选 session/截断/重采样/过滤类别
│   ├── split.py             确定性切分:subject_split(seeded) / loso_split —— 全项目唯一切分来源
│   ├── preproc.py           EA、通道补齐、bandpass/notch
│   └── channels.py          通道名与 scalp 位置
│
├── models/                  模型层(全部 vendored,无外部仓库依赖)
│   ├── base.py              ModelAdapter 统一接口:preprocess/build/forward + finetune/infer/export/mc_uncertainty
│   ├── __init__.py          get_adapter(name) 注册表
│   ├── ifnet/ eegnet/ adfcnn/   小模型(随机初始化训练)
│   ├── mirepnet/            大模型(EA + 45ch pad;mlm.py 网络)
│   ├── cbramod/             大模型(adapter.py = settled CAR-only 高分版)
│   └── labram/              大模型(250→200Hz patchify + input_chans 映射)
│
├── collab/                  协同算法库(可 import,不做命令行编排)
│   ├── artifacts.py         ★ artifact hub:save/load/load_aligned(逐行 y 对齐校验)
│   ├── distill.py           离线蒸馏核心:distill_student(CE+KD+特征对齐+DKD/proto/relational/pearson 变体)
│   ├── seed.py              统一全栈播种 set_seed(random/np/torch/cuda+cudnn)
│   ├── bidirectional.py  mutual.py  bdeeg.py   双向/互学习算法(experiments/bidir/ 调用)
│
├── experiments/             实验层(正式入口)
│   ├── run.py               config-driven runner:python -m experiments.run configs/exp/*.yaml
│   ├── protocols.py         within/loso cell 生成
│   ├── methods.py           YAML condition -> distill_student 参数 registry
│   ├── distill/             run_distill.py(KD/MMD/Combo/mask/dkd/eakd/adaptive/pearson 总入口)
│   │                        run_loso_distill.py / run_loso_subject_oof_kd.py / analyze_fewshot_pearson.py
│   ├── bigmodel/            cbramod/labram/mirepnet 的适配与调参(tune_*.py)
│   ├── bidir/               双向/CR-AMD/BD-EEG/feature-mutual 驱动(负结果复现)
│   └── mask/                run_wrong_sample.py(wrong-sample E0-E5)
│
├── scripts/                 工具层
│   ├── check/               自检:verify_foundation / verify_backbones / smoke_test
│   ├── export/              ★ 工件导出:finetune_export(主) / export_preds / export_teacher_loso(_subjoof) / export_teacher_mc
│   └── legacy/              run_*.sh 规范启动器(各实验线收口命令记录,路径已指向 experiments/)
│
├── eval/                    评估层
│   ├── metrics.py           acc/kappa/per-class
│   ├── stats.py             配对统计:种子先平均 -> subject/fold 配对 Wilcoxon + Holm + bootstrap CI
│   └── __main__.py          CLI:python -m eval '<csv glob>'
│
├── tools/                   汇总工具
│   ├── make_summary_xlsx.py CSV -> results/summary_*.xlsx
│   └── compare_repro.py     历史 vs 重跑对比(deterministic 逐行 / ci 置信区间)
│
├── weights/                 *.pth symlink -> /data1/llx/pretrained_weights/
└── results/(gitignored)     artifacts/ 工件缓存 · metrics/ 历史 CSV · metrics_repro/ 重跑 CSV
```

**已归档(D0-onward 线,git tag `pre-consolidation` 可恢复):**
`experiments/fusion/`、`experiments/adapt/`、`eval/d0.py`、`collab/{fusion,router,ensemble}.py`。

---

## 2. 配置与数据

- `config.py`: `load_dataset_config(name)` / `load_model_config(name)`。
- 注意 `val_split: 0.3` 表示**测试集占 30%**(70% 训练/校准),不是验证集。
- 种子统一 `[666, 667, 668]`,写在各 dataset YAML。
- `data/split.py` 是全项目**唯一**切分来源——所有模型必须走同一 split,否则 artifact
  按行对齐的蒸馏/协同会错位。`collab/artifacts.load_aligned` 会用 y 做逐行一致性护栏。
- 原始数据路径:`/data1/llx/<ds>/{X,labels}.npy`,用 `DATA_ROOT` env 覆盖。

## 3. 模型接入层(models/)

统一接口 `models/base.py:ModelAdapter`:

```text
preprocess(X_raw) -> 模型输入张量
build(num_classes) -> nn.Module(加载 weights/*.pth)
forward(model, x) -> (feat, logits)
```

小模型(ifnet/eegnet/adfcnn)吃原始 `(B,C,T)`,随机初始化;大模型各有预处理:
MIRepNet = per-subject EA + 45 通道补齐;CBraMod/LaBraM = 250→200Hz + patchify,
CBraMod 用 `adapter.py`(settled CAR-only 高分版,`--model cbramod` 即走这里)。
读模型时先读 adapter,别钻网络结构。

## 4. 工件层(artifact hub)—— 跨 env 解耦的核心

```text
results/artifacts/<dataset>/<model>[|_loso]/<key>_<seed>_<split>.npz
  {logits (N,C) f32, feats (N,D) f32, y (N,) i64}
```

- **artifact ≠ checkpoint**:模型每次从 `weights/*.pth` 重建,只落工件和 CSV。
- 命名:within = `<model>/<subject>_<seed>_<split>.npz`;
  loso = `<model>_loso/<fold>_<seed>_<split>.npz`;特殊键 `mirepnet_loso`、
  `mirepnet_loso_subjoof` —— **改名会破坏所有消费者**。
- 导出脚本在 `scripts/export/`(`finetune_export.py` 是主入口),在**各自模型的 env** 里跑。
- `ARTIFACT_ROOT` env 可换工件根目录(重跑时指向 `results/artifacts_v0` 复用历史教师工件)。

## 5. 协同算法库(collab/)

- `distill.py:distill_student(...)` 是离线蒸馏核心:
  `L = CE + lam_kd·KL + lam_feat·(1-cos)` 及 DKD/prototype/relational/pearson 变体;
  teacher 冻结为常数,只传 `feat_t/log_t`。
- `seed.py:set_seed` 是唯一播种实现(distill.py 以 `_set_seed` 别名导出,双向线脚本依赖该别名)。
- `bidirectional.py / mutual.py / bdeeg.py` 是**双向**算法(两模型同进程同时更新),与离线蒸馏不同。

## 6. 实验层(experiments/)

读任何 driver 的顺序:`parse_args/YAML → config → protocol/split → artifact/adapter → collab → eval.metrics → CSV`。

- **`run.py`** = config-driven 主入口(矩阵实验优先 YAML):`python -m experiments.run configs/exp/<name>.yaml [--gpu N] [--seed S] [--report]`。
  `protocols.py` 生成 (dataset×unit×seed) cell;`methods.py` 把 condition 名映射成
  distill 参数(baseline/kd/feat/combo/proto/dkd + masked sugar)。
- **`distill/`** — 蒸馏实验:`run_distill.py` 是历史主入口(mask/dkd/eakd/adaptive/pearson
  各 flag 都在它身上);LOSO 版走 `run_loso_distill.py`、`run_loso_subject_oof_kd.py`。
- **`bigmodel/`** — 大模型适配(`*_adapt.py`)与网格调参(`tune_*.py`,
  1-seed search + 3-seed confirm,断点续跑)。
- **`bidir/` + `mask/`** — 负结果复现线,命令以 `scripts/legacy/run_*.sh` 为准。

## 7. 评估层(eval/)与工具(tools/)

- `python -m eval 'results/metrics/*.csv' [--baseline X]` — 配对 Wilcoxon + Holm + CI。
  **小提升只看这个,不看 raw mean**(项目纪律:固定种子 + 被试级配对 + Holm 才能下结论)。
- 重跑验收:`tools/compare_repro.py`(确定性家族逐行 bit-identical;审计家族 CI 内)。
- 汇总:`tools/make_summary_xlsx.py --hist <glob> --repro <glob>` → `results/summary_*.xlsx`。

---

## 8. 跑实验命令速查(重跑时按此执行)

通用约定:长任务 `setsid nice -n 19 conda run -n <env> python ... > logs/<task>.log 2>&1 < /dev/null &`;
重跑时 `export REPRO_OUT=results/metrics_repro`(不覆盖历史);
教师工件若要复用历史缓存,学生蒸馏加 `export ARTIFACT_ROOT=<旧工件目录>` 即可;或先 `mv results/artifacts results/artifacts_v0`。

### 0) 自检(跑任何东西前)

```bash
conda run -n mirepnet python scripts/check/smoke_test.py --models ifnet eegnet adfcnn mirepnet
conda run -n mirepnet python scripts/check/verify_backbones.py --model mirepnet
conda run -n cbramod  python scripts/check/verify_backbones.py --model cbramod
conda run -n labram   python scripts/check/verify_backbones.py --model labram
```

### 1) 基础蒸馏 / MMD / COMBO(within,MIRepNet 教师)

```bash
# 教师工件(mirepnet env):
conda run -n mirepnet python scripts/export/finetune_export.py --model mirepnet --dataset BNCI2014004 --gpu 2
# 蒸馏(mirepnet env,学生 ifnet;feat=MMD 对齐,combo=KD+feat):
conda run -n mirepnet python experiments/distill/run_distill.py \
  --dataset BNCI2014004 --teacher mirepnet --student ifnet --lam_kd 0.5 --lam_feat 0.5 --gpu 2
# 或 config 驱动(4 条件 base/KD/KD_masked/Combo):
conda run -n mirepnet python -m experiments.run configs/exp/distill_kd_within.yaml --report --gpu 2
```
对比锚点:原 MIRepNet 仓库 CSV(`/home/lixinli/MIRepNet/result/`)。

### 2) few-shot / K-shot

```bash
conda run -n mirepnet python experiments/distill/run_distill.py \
  --dataset BNCI2014001-4 --teacher mirepnet --student ifnet \
  --fewshot_pearson --shots 5 10 20 --lam_pearson 0.5 --gpu 2
```

### 3) EEGNet / ADFCNN 学生

```bash
conda run -n mirepnet python experiments/distill/run_distill.py \
  --dataset BNCI2014004 --teacher mirepnet --student eegnet  --gpu 2   # 或 adfcnn
```

### 4) CBraMod 复现 / 调参 / 协议消融(cbramod env)

```bash
bash scripts/legacy/run_cbramod_paper5.sh native70 "3 5 6 8 2"          # 5 数据集复现(native70)
conda run -n cbramod python experiments/bigmodel/tune_cbramod.py --phase all --gpus 2 3 5   # within 调参
conda run -n cbramod python experiments/bigmodel/tune_cbramod_004_caronly.py   # 004 CAR-only 精调
conda run -n cbramod python experiments/bigmodel/tune_cbramod_loso.py --phase all --gpus 1 8 --threads 4  # ★LOSO 收尾(07-29 未完成,断点续跑)
# 协议/预处理消融旋钮(cbramod_adapt.py):--head linear|mlp --norm_method car|none --scale_divisor 1 --band b50|b75n60
```

### 5) CBraMod 作教师蒸馏

```bash
conda run -n cbramod  python scripts/export/export_teacher_mc.py --model cbramod --dataset BNCI2014004 --gpu 2
conda run -n mirepnet python experiments/distill/run_distill.py --dataset BNCI2014004 --teacher cbramod --student ifnet --gpu 2
```

### 6) LaBraM 复现 / 调参(labram env)

```bash
bash scripts/legacy/run_labram_paper5.sh 2
conda run -n labram python experiments/bigmodel/tune_labram.py --phase all --gpus 2 3 5
```

### 7) Mask / Confidence / Adaptive / EA-KD / DKD(命令以对应 run_*.sh 为准)

```bash
bash scripts/legacy/run_mask_ablation.sh            # mask(教师正确样本掩码)
bash scripts/legacy/run_mask_ablation_eegnet.sh     # mask × EEGNet 学生
bash scripts/legacy/run_mask_ablation_B_cbramod.sh  # mask × CBraMod 教师
bash scripts/legacy/run_dkd_ablation.sh             # DKD(base/KD_all/KD_masked/DKD_all/DKD_tmask)
bash scripts/legacy/run_adaptive_entropy_kd.sh      # 先 export_teacher_mc,再 --adaptive(MC 熵加权)
bash scripts/legacy/run_eakd_within.sh              # EA-KD
bash scripts/legacy/run_eakd_combo_matrix.sh        # EA-KD × Combo 矩阵
bash scripts/legacy/run_relational_ablation.sh      # 相似度保持蒸馏
```

### 8) LOSO 跨被试蒸馏

```bash
bash scripts/legacy/run_loso_full.sh                # export_teacher_loso -> run_loso_distill(001-4/004)
bash scripts/legacy/run_loso_ext_0014.sh            # LOSO reliability + pred-states(001-4)
bash scripts/legacy/run_loso_ext_004.sh             # 同上(004)
# subject-OOF(先导出再蒸馏):
conda run -n mirepnet python scripts/export/export_teacher_loso_subjoof.py --dataset BNCI2014001-4 --gpu 2
conda run -n mirepnet python experiments/distill/run_loso_subject_oof_kd.py --dataset BNCI2014001-4 --gpu 2
```

### 9) 双向 / CR-AMD / BD-EEG

```bash
bash scripts/legacy/run_bidir_full.sh               # 双向 routed 蒸馏(001-4/004)
bash scripts/legacy/run_cramd_full.sh               # CR-AMD(001-4)
bash scripts/legacy/run_bdeeg_parallel.sh           # BD-EEG 9-fold 并行(001-4)
```

### 10) Pearson logit-distance(仅 4 类 001-4)

```bash
conda run -n mirepnet python experiments/distill/run_distill.py \
  --dataset BNCI2014001-4 --teacher mirepnet --student ifnet --fewshot_pearson --shots 5 10 20 --gpu 2
conda run -n mirepnet python experiments/distill/analyze_fewshot_pearson.py \
  results/metrics/BNCI2014001-4_fewshot_pearson_mirepnet_to_ifnet.csv
```

### 验收 / 对比 / 汇总(跑完后)

```bash
python tools/compare_repro.py --hist 'results/metrics/<fam>_*.csv' \
  --repro 'results/metrics_repro/<fam>_*.csv' --mode deterministic   # ✅ 家族:逐行 bit-identical
python tools/compare_repro.py --hist '...' --repro '...' --mode ci --tol 0.5   # ❌ 家族:CI 内
python -m eval 'results/metrics_repro/<fam>_*.csv'                  # 配对统计报告
python tools/make_summary_xlsx.py --hist 'results/metrics/*.csv' --repro 'results/metrics_repro/*.csv'
```

---

## 9. 归档(怎么找回 D0-onward 代码)

```bash
git tag -l                     # pre-consolidation / consolidation
git checkout pre-consolidation -- experiments/fusion experiments/adapt eval/d0.py collab/fusion.py
```

## 10. 推荐阅读顺序

1. `REPRO.md` 第 0 节(数据流)→ `README.md` → 本指南 §1 目录树
2. `configs/datasets/*.yaml` → `data/split.py` → `data/eeg_dataset.py`
3. `models/base.py` → `models/__init__.py` → 一个小模型 adapter(`models/ifnet/adapter.py`)→ 一个大模型 adapter(`models/mirepnet/adapter.py`)
4. `scripts/export/finetune_export.py` → `collab/artifacts.py`
5. `experiments/run.py` + `protocols.py` + `methods.py` → `collab/distill.py`
6. `eval/stats.py` → `tools/compare_repro.py`

## 11. 最容易混淆的点

- **checkpoint ≠ artifact**:checkpoint 是权重;artifact 是模型对固定样本导出的 logits/feats/y。
- **离线蒸馏 ≠ 双向蒸馏**:离线 = teacher 冻结、学生单向学工件;双向 = 两模型同进程同时更新互教(要求同 env)。
- **模型代码在仓库内 ≠ 能在同一 env 同时跑**:依赖冲突仍在,所以 artifact hub 是核心设计。
- **✅/❌ 家族**:✅(within 蒸馏、mask、dkd、loso 学生)期望重跑 bit-identical;
  ❌(大模型调参、bidir/CR-AMD/BD-EEG、LOSO 大模型)期望 CI 内——不要拿 raw diff 判失败。
- **新实验优先 YAML**:矩阵实验写 `configs/exp/*.yaml` + `methods.py` registry;专门 driver 放 `experiments/<line>/`;导出/自检才放 `scripts/`。

# 大小模型协同实验结果与数据上下文整理

检查时间：2026-07-28。依据当前本地仓库、`PROGRESS.md`、已落盘结果，以及 `/data1/llx/*` 数据文件。

这份文档专门把几件容易混在一起的事情拆开：

1. 现在到底得到了什么结果。
2. 每个结果是在什么训练/微调场景下得到的。
3. 每个数据集实际用了哪个 session、哪些 trial、哪些被试。

## 总结论

现在最诚实的结论是：

- 原始目标：**在大小模型都允许端到端微调的情况下，找到一个协同方法超过二者各自独立端到端微调的最好结果**。这个目标目前还没有完成。
- 稳健正结果：只在**冻结特征 / 缓存 artifact** 场景成立。也就是 backbone 不动，只用缓存好的 features/logits，在测试被试的 K 个标注 trial 上训练轻量 head。
- 在这个冻结场景下，把 `fusion` 作为一个候选加入支撑集 CV 选池：
  - `cv_sel2 = CV 选 {head_big, head_small}`
  - `cv_sel3 = CV 选 {head_big, head_small, fusion}`
  - `cv_sel3` 跨 4 个数据集、24 个 cell 稳定比 `cv_sel2` 高约 `+0.5~+0.6` 个 accuracy point。
- 但是 A 微调基线说明：一旦允许单模型端到端微调，`best_ft = max(ft_big, ft_small)` 比冻结 fusion 更强。所以这个正结果不能说是普适 SOTA，只能说是**冻结/缓存/低成本 hub 场景下的稳健小增益**。

## 实验线索总表

| 实验线 | 场景 | 数据集 | 模型 | 划分/适应方式 | 结果状态 | 主要文件 |
|---|---|---|---|---|---|---|
| D0 headroom 诊断 | 缓存 logits/features，within 和 LOSO | BNCI2014001-4, BNCI2014004 | 大：MIRepNet, CBraMod native；小：IFNet, EEGNet, ADFCNN | within = 每被试 downstream session 内 70/30；LOSO = 留一被试测试，其余被试训练 | oracle union headroom 真实存在，大约 +9~17，但跨被试静态 route/fuse 拿不到 | `results/d0/decision_report.md`, `results/headroom_map.csv` |
| R1 logit router | LOSO 缓存 logits 上训练 gate | 旗舰：BNCI2014001-4, MIRepNet x IFNet | MIRepNet, IFNet | 嵌套 subject-LOSO 训练 gate | 负结果：静态 gate 比基线更差，同被试校准只有 +0.4 且 ns | `collab/router.py`, `experiments/fusion/run_r1_signal.py`, `PROGRESS.md` |
| KD / 离线蒸馏 | student 对冻结 teacher artifact 训练 | 主要 BNCI2014001-4, BNCI2014004；部分 BNCI2015001 | teacher: MIRepNet/CBraMod；student: IFNet/EEGNet/ADFCNN | within 70/30 或 LOSO，依脚本而定 | 机制性负结果，没有稳健主正结果 | `experiments/distill/*`, `results/metrics/*distill*` |
| 双向 / 互蒸馏 | 两个模型同进程共同训练 | 主要 MIRepNet x IFNet | MIRepNet, IFNet | LOSO 或 few-shot | 负/不稳定；feature-level mutual 没有超过 concat fusion | `scripts/legacy/bidir/*`, `results/metrics/*bidir*`, `*cramd*`, `*bdeeg*` |
| wrong-sample 利用 | 利用 teacher-correct / teacher-wrong 样本 | BNCI2014001-4, BNCI2015001 | teacher: MIRepNet/CBraMod；student: IFNet | 多为 within split | 基本 closed/null；只有普通 correct-only KD-like 信号残留 | `scripts/legacy/wrongsample/run_wrong_sample.py`, `results/metrics/wrong_sample_*` |
| F+T 初始 feature fusion | 冻结 LOSO features，测试被试 K-shot | BNCI2014001-4, BNCI2014004 | 大：MIRepNet/CBraMod；小：IFNet/EEGNet/ADFCNN | 在 K 个 support trial 上训练 fusion 线性头，剩余 trial 评估 | 初始 “12/12 win” 是基线错误，只比了 `head_big` | `results/metrics/ft_generality.csv` |
| 修正后的 balance-gated selection | 冻结 LOSO features，测试被试 K-shot | BNCI2014001-4, BNCI2014004, BNCI2015001, AlexMI | 大：MIRepNet/CBraMod；小：IFNet/EEGNet/ADFCNN | `cv_sel2` 支撑集 CV 选大/小 head；`cv_sel3` 支撑集 CV 选大/小/fusion | 正结果但范围有限：24 cell 中 K20 +0.51，K30 +0.58 | `experiments/fusion/run_balance_gate.py`, `results/metrics/balance_gate.csv` |
| A 端到端微调基线 | 单模型在 K support 上真实端到端微调 | 记录中的旗舰：BNCI2014001-4, MIRepNet x IFNet | MIRepNet, IFNet | 重训 LOSO base，deepcopy 后在 K support 上微调整个模型 | 记录结论：best single FT 比 frozen fusion 高约 +3.9 / +4.8 | `experiments/fusion/run_finetune_baseline.py`, `PROGRESS.md`；当前 CSV 不完整 |

## 协议定义

先澄清一个术语：现在的 few-shot 主线不是一个独立的“从零 few-shot 数据集划分”。它是：

> 先按 LOSO 训练/导出每个模型的 held-out subject features/logits，再在这个 held-out subject 的测试池里抽 K 条标注 trial 做 few-shot 适应。

所以准确叫法应该是 **LOSO + target-subject K-shot adaptation**。
不是“只跑了 LOSO、没有 few-shot”，也不是“普通 within-subject few-shot”。

但这也意味着：如果最终论文目标是“只在 few-shot 调参协议下做协同”，那这些 LOSO-base K-shot 结果只能当诊断/临时结果，不能当最终主结果。最终主实验应该用 few-shot 协议下已经调好的单模型参数，重新生成同一协议下的大/小模型输出，再做协同比较。

### within-subject 70/30

实现位置：`data.subject_split(dataset, subject, val_split=0.3, seed=...)`。

流程：

- 先用 `data_mode='session3'` 读取该被试的 downstream pool。
- 这里的 session3 是代码变量名，不是所有数据集真实都有一个叫 session 3 的 session。具体对应关系：BNCI2014001-4 = session_E；BNCI2014004 = loader 里写死的 p1:p2 block；BNCI2015001 = session_A；AlexMI = session=0, run=0 且 drop 掉 rest。
- 再做 stratified `train_test_split(..., test_size=0.3, random_state=seed)`。
- 训练/校准集 = 70%，测试集 = 30%。
- 这个协议用于早期 per-subject finetune/export、KD、ensemble 等实验。

注意：这里的 `val_split=0.3` 在项目里表示**测试比例**，不是验证集比例。

### LOSO

实现位置：`data.loso_split(dataset, test_subject)`。

流程：

- fold `t` 的测试集 = 被试 `t` 的 downstream pool。
- 训练集 = 其他所有被试的 downstream pool。
- fold 层面没有随机切分。
- LOSO artifact 路径：

```text
results/artifacts/<dataset>/<model>_loso/<subject>_<seed>_{train,test}.npz
```

artifact 里只保存：

- `logits`
- `feats`
- `y`

没有保存 raw row id。因此如果要追溯“第几条 trial”，需要按照下面的数据加载规则重建。逐被试 raw row 范围也单独落在 docs/dataset_selected_rows.csv。

### LOSO 测试被试上的 K-shot

用于：

- `experiments/fusion/run_ft_fusion.py`
- `experiments/fusion/run_balance_gate.py`
- `experiments/fusion/run_finetune_baseline.py`

流程：

- 先拿 LOSO held-out subject 的 test pool。
- 用 `StratifiedShuffleSplit(train_size=K, random_state=seed)` 从这个 pool 里抽 K 条标注 trial 作为 support。
- 剩余 trial 作为 eval。
- `run_balance_gate.py`：K = 20/30，seeds = 666/667/668，draws = 5。
- `run_ft_fusion.py`：K = 10/20/30，seeds = 666/667/668，draws = 5。
- `run_finetune_baseline.py` 默认：K = 20/30，seeds = 666/667/668，draws = 3；但记录中的 A 旗舰 run 是 GPU 受限设置。

重点：

- LOSO 的 held-out subject test pool 是下面表里的连续/筛选后 trial 池。
- K-shot 的 support/eval trial **不是连续的“第几条到第几条”**，而是在每个被试 test pool 里按类别分层随机抽出来的非连续索引。
- 因此如果只问“这个被试用了哪部分样本”，看下面每个数据集的 selected rows。
- 如果问“某个 seed/K/draw 的 K 条 support 到底是哪几条”，需要用 StratifiedShuffleSplit 按 seed 重建。

## 主要数值结果

### D0 诊断

来自 `results/d0/decision_report.md`：

- BNCI2014001-4 LOSO，MIRepNet x IFNet：
  - big = 48.37
  - small = 41.31
  - oracle union = 64.72
  - headroom = +16.36
- BNCI2014004 LOSO，MIRepNet x IFNet：
  - big = 76.77
  - small = 74.35
  - oracle union = 85.87
  - headroom = +9.10
- CBraMod x small 的部分 LOSO cell 也有很大 oracle headroom，但 CBraMod 本身有时比小模型弱。

解释：

- 大小模型确实会犯不同的错误。
- 如果 oracle 知道每个样本谁对，就能获得很大提升。
- 真实困难是：没有标签时，如何稳定判断每个样本该信谁。

### R1 logit router

来自 `PROGRESS.md`：

- 旗舰 cell：BNCI2014001-4，MIRepNet x IFNet，LOSO。
- big / small / avg ensemble = 48.37 / 41.31 / 48.59。
- 静态跨被试 soft gate = 46.71，显著差于基线。
- 同被试 2-fold 校准 = 50.10，只比 `max(big, avg ensemble)` 高 +0.40，ns。
- oracle union = 64.72。

解释：

- logits 里有一点同被试信息。
- 但这个信息跨被试不稳定，不能支持静态 gate。
- D0 的大部分 oracle headroom 不是简单 logit router 能拿到的。

### 修正后的冻结特征选池结果

来自 `results/metrics/balance_gate.csv`。

设置：

- 4 数据集：
  - BNCI2014001-4
  - BNCI2014004
  - BNCI2015001
  - AlexMI
- 2 个大模型：
  - MIRepNet
  - CBraMod native
- 3 个小模型：
  - IFNet
  - EEGNet
  - ADFCNN
- 一共 24 cell。
- K = 20 / 30。

结果：

| K | `cv_sel3 - cv_sel2` | 正 cell 数 | `bal_gate - cv_sel2` | 正 cell 数 |
|---|---:|---:|---:|---:|
| 20 | +0.51 | 20/24 | +0.59 | 19/24 |
| 30 | +0.58 | 21/24 | +0.71 | 19/24 |

按数据集平均的 `cv_sel3 - cv_sel2`：

| 数据集 | 平均增益 |
|---|---:|
| AlexMI | +0.60 |
| BNCI2014001-4 | +0.93 |
| BNCI2014004 | +0.31 |
| BNCI2015001 | +0.34 |

方法含义：

- `head_big`：冻结大模型特征，在 K 个 support labels 上训练线性头。
- `head_small`：冻结小模型特征，在 K 个 support labels 上训练线性头。
- `fusion`：拼接 `[big_feat, small_feat]`，在 K 个 support labels 上训练线性融合头。
- `cv_sel2`：只在 `head_big` / `head_small` 里用 support-CV 选择。
- `cv_sel3`：在 `head_big` / `head_small` / `fusion` 里用 support-CV 选择。

这个就是目前最稳的正结果。但它只说明：

> 在冻结/缓存特征场景下，把 fusion 加入候选池有稳定小增益。

它不说明：

> fusion 或 `cv_sel3` 能超过真实端到端微调的单模型。

### A 端到端微调基线

来自 `PROGRESS.md` 和 memory 记录。

设置：

- 旗舰 cell：BNCI2014001-4，MIRepNet x IFNet，LOSO。
- 先在非 held-out subjects 上训练 LOSO base。
- 对每个测试被试，deepcopy base。
- 在同样的 K support trials 上端到端微调整个单模型。
- 记录中的 run：9 被试 x 2 seeds，`base_epochs=40`，`ft_epochs=30`。

结果：

| K | `ft_big` | `ft_small` | frozen `fusion` | frozen `head_big` | frozen `head_small` |
|---|---:|---:|---:|---:|---:|
| 20 | 51.4 | 52.9 | 50.0 | 43.3 | 45.7 |
| 30 | 53.7 | 54.2 | 51.9 | 44.7 | 47.7 |

解释：

- `best_ft = max(ft_big, ft_small)`。
- K20：`best_ft` 比 frozen fusion 高约 +3.9，5/9 被试，ns。
- K30：`best_ft` 比 frozen fusion 高约 +4.8，8/9 被试，p = 0.008。
- `ft_big > frozen head_big`：9/9 被试，p = 0.0039。

所以：

> 如果允许端到端微调单模型，单模型微调 baseline 更强。

复现 caveat：

- 当前 `results/metrics/finetune_baseline.csv` 只剩 1 行：
  - BNCI2014001-4
  - MIRepNet x IFNet
  - subject 0
  - seed 666
  - K30
  - draw0
- 9 被试完整 A 汇总目前保存在 `PROGRESS.md` 和 `/home/lixinli/.claude/projects/-home-lixinli-BigSmallCollab/memory/ft-fusion-positive.md`，没有完整落在当前 CSV 里。

## 主 LOSO / K-shot 实验使用的数据池

下面所有 row index 都是 `/data1/llx/<dataset>/X.npy` 里的**零基 raw row index**，除非特别说明。

artifact 的 row 顺序等于 loader 筛选后的顺序。

### BNCI2014001-4

来源：

```text
/data1/llx/BNCI2014001/X.npy
shape = (5184, 22, 1001)
```

设置：

- loader dataset name：`BNCI2014001-4`
- 任务：4 类 MI
  - `feet`
  - `left_hand`
  - `right_hand`
  - `tongue`
- 只用 `session_E` 作为 downstream pool。
- `session_T` 不进入这些主 LOSO / K-shot 实验的测试池。
- 每个 trial 从 1001 samples 截断到 1000。
- 9 被试。
- 每被试选中 288 trials。
- 每类 72 trials。
- 每个 selected session 有 6 个 run，每 run 48 trials，每类 12 trials。

对 0-index subject `s`：

- raw subject block = `[576*s, 576*s+575]`
- selected downstream/test pool = `[576*s+288, 576*s+575]`
- selected pool 里的 run `r` = `[576*s+288+48*r, 576*s+335+48*r]`

| 0-index subject | raw subject id | 使用 session | raw rows | n |
|---:|---:|---|---|---:|
| 0 | 1 | session_E | 288-575 | 288 |
| 1 | 2 | session_E | 864-1151 | 288 |
| 2 | 3 | session_E | 1440-1727 | 288 |
| 3 | 4 | session_E | 2016-2303 | 288 |
| 4 | 5 | session_E | 2592-2879 | 288 |
| 5 | 6 | session_E | 3168-3455 | 288 |
| 6 | 7 | session_E | 3744-4031 | 288 |
| 7 | 8 | session_E | 4320-4607 | 288 |
| 8 | 9 | session_E | 4896-5183 | 288 |

LOSO fold `t`：

- train = 上表里除了 subject `t` 以外的所有 selected rows。
- test = subject `t` 的 selected rows。

### BNCI2014004

来源：

```text
/data1/llx/BNCI2014004/X.npy
shape = (6520, 3, 1126)
```

设置：

- loader dataset name：`BNCI2014004`
- 任务：2 类 MI
  - `left_hand`
  - `right_hand`
- loader 现在从 `/data1/llx/BNCI2014004/meta004.csv` 正常读取 metadata。
- 默认使用真实 `session_3`。旧代码用硬编码 `p1:p2`，对应的正是 `meta004.csv` 里的 `session_3`。
- `session_4` 没有进入当前 downstream few-shot 实验，除非显式设置 `data_mode="session4"` 或环境变量切换。
- 每个 trial 截断到 1000 samples。

| 0-index subject | phase1 rows，不进 downstream few-shot | 使用的 `session3` rows | leftover rows，不用 | selected n | class counts |
|---:|---|---|---|---:|---|
| 0 | 0-399 | 400-559 | 560-719 | 160 | 80/80 |
| 1 | 720-1119 | 1120-1239 | 1240-1399 | 120 | 60/60 |
| 2 | 1400-1799 | 1800-1959 | 1960-2119 | 160 | 80/80 |
| 3 | 2120-2539 | 2540-2699 | 2700-2859 | 160 | 80/80 |
| 4 | 2860-3279 | 3280-3439 | 3440-3599 | 160 | 80/80 |
| 5 | 3600-3999 | 4000-4159 | 4160-4319 | 160 | 80/80 |
| 6 | 4320-4719 | 4720-4879 | 4880-5039 | 160 | 80/80 |
| 7 | 5040-5479 | 5480-5639 | 5640-5799 | 160 | 80/80 |
| 8 | 5800-6199 | 6200-6359 | 6360-6519 | 160 | 80/80 |

LOSO fold `t`：

- train = 其他 subjects 的 selected `session3` rows。
- test = subject `t` 的 selected `session3` rows。

### BNCI2015001

来源：

```text
/data1/llx/BNCI2015001/X.npy
shape = (5600, 13, 2561)
```

设置：

- loader dataset name：`BNCI2015001`
- 任务：2 类 MI
  - `feet`
  - `right_hand`
- 默认只用 `session_A`。
- 环境变量 `MI2015001_SESSION` 可以覆盖，但当前默认/配置是 `session_A`。
- 原始采样率 512 Hz，loader 重采样到 250 Hz。
- 时间长度裁到能被 125 整除，最多 1000。
- 12 被试。
- 每被试 selected 200 trials。
- 每类 100 trials。

| 0-index subject | raw subject id | 使用 session | raw rows | n | class counts |
|---:|---:|---|---|---:|---|
| 0 | 1 | session_A | 0-199 | 200 | 100/100 |
| 1 | 2 | session_A | 400-599 | 200 | 100/100 |
| 2 | 3 | session_A | 800-999 | 200 | 100/100 |
| 3 | 4 | session_A | 1200-1399 | 200 | 100/100 |
| 4 | 5 | session_A | 1600-1799 | 200 | 100/100 |
| 5 | 6 | session_A | 2000-2199 | 200 | 100/100 |
| 6 | 7 | session_A | 2400-2599 | 200 | 100/100 |
| 7 | 8 | session_A | 2800-2999 | 200 | 100/100 |
| 8 | 9 | session_A | 3400-3599 | 200 | 100/100 |
| 9 | 10 | session_A | 4000-4199 | 200 | 100/100 |
| 10 | 11 | session_A | 4600-4799 | 200 | 100/100 |
| 11 | 12 | session_A | 5200-5399 | 200 | 100/100 |

raw metadata 里 subjects 8-11 还有 `session_C`，但当前 loader 默认不用。

### AlexMI

来源：

```text
/data1/llx/AlexMI/X.npy
shape = (480, 16, 1537)
```

设置：

- loader dataset name：`AlexMI`
- 原始标签：
  - `feet`
  - `right_hand`
  - `rest`
- 当前任务只用 2 类：
  - `right_hand`
  - `feet`
- `rest` 被 drop。
- metadata 里只有 session `0`、run `0`。
- 原始采样率 512 Hz。
- loader 重采样到 250 Hz：1537 -> 750。
- 然后把前 250 samples 拼到后面，让长度变成 1000。
- 8 被试。
- 每被试原始 60 trials。
- drop `rest` 后，每被试 selected 40 trials。
- 每类 20 trials。

| 0-index subject | raw subject id | raw block | drop rest 后 selected n | class counts |
|---:|---:|---|---:|---|
| 0 | 1 | 0-59 | 40 | 20/20 |
| 1 | 2 | 60-119 | 40 | 20/20 |
| 2 | 3 | 120-179 | 40 | 20/20 |
| 3 | 4 | 180-239 | 40 | 20/20 |
| 4 | 5 | 240-299 | 40 | 20/20 |
| 5 | 6 | 300-359 | 40 | 20/20 |
| 6 | 7 | 360-419 | 40 | 20/20 |
| 7 | 8 | 420-479 | 40 | 20/20 |

因为 drop 了 `rest`，最后 selected rows 不是连续区间。精确 non-rest raw ranges 如下：

| raw subject id | selected non-rest raw ranges |
|---:|---|
| 1 | 0-3, 5-11, 14, 19, 21-25, 28-29, 31-35, 38, 40-41, 43, 45-47, 49, 51-55, 58-59 |
| 2 | 61-63, 66-68, 70-71, 75-78, 80-85, 87-88, 90-91, 93-96, 98, 101-102, 104-105, 107-110, 112-113, 116-117, 119 |
| 3 | 121-123, 126-128, 130-131, 135-138, 140-145, 147-148, 150-151, 153-156, 158, 161-162, 164-165, 167-170, 172-173, 176-177, 179 |
| 4 | 181-183, 186-188, 190-191, 195-198, 200-205, 207-208, 210-211, 213-216, 218, 221-222, 224-225, 227-230, 232-233, 236-237, 239 |
| 5 | 240-245, 248-249, 251-252, 256-257, 261-262, 264-272, 274-275, 277, 279, 282-286, 288, 290-292, 294, 296-298 |
| 6 | 302, 304, 306-308, 310-316, 318-321, 323-324, 327-328, 332-334, 336-339, 341-344, 346-347, 349, 351, 353-355, 357-358 |
| 7 | 360-361, 364-366, 368, 370-373, 375-376, 379-380, 382-385, 387-388, 390-392, 395-403, 405-408, 410, 412, 414, 416 |
| 8 | 420-422, 424-425, 427-429, 431-432, 434, 436-437, 439-442, 446-447, 449, 451-452, 454-458, 460, 463-465, 467, 469, 471-472, 474, 476-479 |

## 如何重建某个 K-shot draw 的具体 support/eval raw rows

artifact 里没有保存 support/eval indices。要重建，需要：

1. 按上面的表先得到 held-out subject 的 selected raw row list。
2. 读取 artifact 里的 `y`。
3. 用同样的 `StratifiedShuffleSplit` 生成 support/eval local indices。
4. 用 local indices 映射回 raw rows。

代码模板：

```python
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# y 是某个 held-out subject 的 artifact y。
# raw_map 是 artifact-local row -> raw X.npy row 的映射。
# 对 BNCI2014001-4 / BNCI2014004 / BNCI2015001，raw_map 通常是连续 selected rows。
# 对 AlexMI，raw_map 是 drop rest 后的非连续 selected rows。

sss = StratifiedShuffleSplit(n_splits=5, train_size=K, random_state=seed)
splits = list(sss.split(np.zeros(len(y)), y))

support_local, eval_local = splits[draw]
support_raw = raw_map[support_local]
eval_raw = raw_map[eval_local]
```

注意：

- `run_balance_gate.py` 用 `draws=5`。
- `run_ft_fusion.py` 用 `draws=5`。
- `run_finetune_baseline.py` 默认 `draws=3`。
- 如果要完全追踪论文里的某个数字，必须同时指定：
  - dataset
  - big model
  - small model
  - held-out subject / fold
  - seed
  - K
  - draw

## 模型配置

来自 `configs/models/*.yaml`。

| 模型 | 角色 | env | 默认训练配置 |
|---|---|---|---|
| MIRepNet | big / foundation | `mirepnet` | epochs 10, lr 0.001, batch 8, weight_decay 1e-6, emb_size 256, depth 6 |
| CBraMod native | big / foundation | `cbramod` | batch 16；lr/epochs/dropout/weight_decay/band 由 adapter 内置 per-dataset 表决定 |
| IFNet | small | `mirepnet` | epochs 100, lr 0.001, batch 16, weight_decay 0.01 |
| EEGNet | small | `mirepnet` | epochs 100, lr 0.001, batch 32, weight_decay 1e-4 |
| ADFCNN | small | `mirepnet` | epochs 100, lr 0.001, batch 32, weight_decay 1e-4 |

A 端到端微调基线的记录 run 使用了 `base_epochs=40`，这和 MIRepNet 默认 `epochs=10` 不同。这个 caveat 已经写在 `PROGRESS.md` 里：它可能让 frozen `head_big` 绝对值偏低，但 `ft` vs `fusion` 使用同一个 base，所以这个比较本身仍然相对公平。

## 现在应该怎么讲这个故事

最稳妥的叙事是：

1. D0 证明大小模型之间有真实 oracle 协同空间，但它是被试特异的，简单 logit 空间拿不到。
2. KD、logit router、双向互蒸馏、wrong-sample 等方向没有形成稳健正结果，主要作为机制性负结果。
3. 冻结 LOSO features + 测试被试 K-shot 适应时，单独 fusion 不可靠；强弱悬殊时会稀释强模型。
4. 但把 fusion 作为候选加入支撑集 CV 选池，`cv_sel3` 能稳定小幅超过 `cv_sel2`。
5. 这个正结果只适用于冻结/缓存/低成本 hub 场景。
6. 若允许真实端到端微调单模型，当前记录显示 `best_ft` 更强。因此原始目标还没有达成。


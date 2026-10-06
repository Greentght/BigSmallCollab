# MIRepNet / CBraMod 五种数据设定的 LOSO 复现方案

制定日期：2026-10-01。对应 [实验规格 YAML](../configs/reproductions/loso_five_datasets_v1.yaml)、[47 折清单](loso_five_datasets_folds.csv)、[trial 清单](../configs/reproductions/manifests/loso_five_datasets_v1_trials.csv) 和 [来源快照](../configs/reproductions/manifests/loso_five_datasets_v1_sources.json)。本轮产物是实验方案与数据清单；YAML 是新规格，现有训练调度器尚未读取这个 schema。

## 数据范围与划分

五个设定对应四个原始数据集。`BNCI2014001` 是左右手二分类；`BNCI2014001-4` 是同一原始缓存的四分类，二者均为 22 通道。001-4 中的 `4` 表示类别数。

| 设定 | 类别 / 原生 EEG 通道 | 所用场次 | 被试 / 总 trials | 每折训练 / 测试 |
|---|---|---|---:|---:|
| 001 | 左手、右手 / 22 | `session_T`，loader `sessionT` | 9 / 1296 | 1152 / 144 |
| 001-4 | 脚、左手、右手、舌 / 22 | `session_T`，loader `sessionT` | 9 / 2592 | 2304 / 288 |
| 004 | 左手、右手 / 3 | `session_3`，loader `session3`，对应参考仓 `3test` | 9 / 1400 | S2：1280 / 120；其他：1240 / 160 |
| 5001 | 脚、右手 / 13 | `session_A` | 12 / 2400 | 2200 / 200 |
| AlexMI | 脚、右手 / 16，去除 rest | session `0`、run `0` | 8 / 320 | 280 / 40 |

每折完整留出一位被试，其余被试的所选场次全部 trials 用于训练。两模型共用同一份标签、trial 顺序、测试被试和 source UID。划分由测试被试决定，seed 只影响初始化、训练 shuffle 和随机算子。`val_split`、Fewshot 的 30% 抽样和 run 划分均不参与这版 LOSO。

固定 seeds 为 **666、667、668**。每个 dataset×model×subject×seed 都从同一个预训练 checkpoint 重新初始化，分类头按二类或四类重新构建，全参数微调。

标签编码固定为：001/004 `left_hand=0, right_hand=1`；001-4 `feet=0, left_hand=1, right_hand=2, tongue=3`；5001/Alex `feet=0, right_hand=1`。参考仓 5001 的数字编码与当前缓存不同，因此按类名解释输出，避免直接比较数字含义。

## 窗口与来源

001/001-4、004 取缓存 epoch 前 1000 点 @250 Hz。5001 保留当前加载器的 512→250 Hz `scipy.signal.resample`，再取前 1000 点。AlexMI 保留当前 `mne.filter.resample(up=125, down=256)`，取 750 点后重复开头 250 点，得到模型使用的 1000 点。

所以协议称为 **canonical 4 秒输入**：前四种设定来自约 4 秒采集信号；AlexMI 是 **3 秒信号＋1 秒重复补齐**，必须在表头/说明注明，不能把它写成真实采集 4 秒。时间零点先定义为缓存 epoch 起点；实际 cue 相对时间没有完整生成记录时，记录为未核实。

主实验沿用现有缓存，来源快照记录 `.npy`、metadata、预训练权重和关键代码的 SHA256。已有文档和幅值支持 µV 量级，但数组没有单位、通道名、滤波历史或 event 时间等自描述字段，不能仅凭幅值认定完整生成流程。主表定位为当前缓存与固定协议的复现。若要重建可追溯宽带缓存，需要独立协议版本，重新跑受影响结果；不要把重建缓存的结果和旧 004 混在一起。

trial 清单保留 `(source_dataset, raw_cache_row)`，以及 task、subject、session、run、类名、编码和筛选后 local index。001 与 001-4 的左右手 trials 通过相同 source UID 对齐，识别它们的数据重叠；旧 `(subject, local index)` 仍保留作兼容字段。5001 显式锁定 `MI2015001_SESSION=session_A`，不让环境变量静默改变场次。

## 主实验参数与输入

主实验沿用已完成 004 的 recipe；这是预先固定的基线，不依据其他外层测试被试成绩选参数。训练参数分别来自当前项目的各 dataset LOSO 分支，信号处理在新规格中显式覆盖。

| 项目 | MIRepNet | CBraMod |
|---|---|---|
| 参数来源 | `mirepnet.yaml` 各 dataset 的 LOSO | `cbramod.yaml` 各 dataset 的 LOSO |
| epochs | 001-4：20；其余：10 | 五种设定均 50 |
| optimizer / LR | Adam / 1e-3 | AdamW / 1e-4，eps 1e-8 |
| batch / weight decay | 8 / 1e-6 | 64 / 0.05 |
| label smoothing | 0 | 0.1 |
| dropout | 保留模型原实现 | head 0.1 |
| LR schedule | 每 epoch cosine，min LR 0，无 warmup | 每 epoch cosine，min LR 0，无 warmup |
| 输入 | `[B,45,1000]` @250 Hz | `[B,C_native,4,200]` @200 Hz |
| 频率处理 | 8–30 Hz | 0.3–75 Hz＋60 Hz notch |
| 空间 / 幅值处理 | 按被试分别 EA，再映射到 45 通道 | 原生通道，EA=false、CAR关闭、scale=1 |
| 分类头 | 保留现有 MIRepNet head，输出 2/4 类 | Flatten＋Dropout＋Linear，输出 2/4 类 |

MIRepNet 的顺序为：canonical window → `data.preproc.bandpass` 的四阶、双向 Butterworth 8–30 Hz → 每个训练被试单独 EA → 人工 2D 位置的逆距离插值到 45 通道。它是插值，不是补零；已有目标通道直接复制，其他目标通道按距离加权。进入 adapter 时使用 `skip_preprocess=True`，防止混合 8/11/7 位训练被试后再次统一 EA。

MIRepNet 测试被试使用其全部无标签 trials 估计自己的 EA 协方差，这版明确记为 **transductive LOSO**。CBraMod 主 recipe 不估计测试集的共享统计。两者保留各自的输入处理需求；论文若要声称严格不使用目标数据分布，需要单列 MIRepNet EA 关闭或其他明确的源数据方案，另给协议名。

CBraMod 顺序为：canonical window → 当前 adapter 的 250→200 Hz 重采样 → 四阶双向 Butterworth 0.3–75 Hz → Q=30 的双向 60 Hz IIR notch → 原生通道 reshape。001/001-4 输入 `[B,22,4,200]`，004 `[B,3,4,200]`，5001 `[B,13,4,200]`，Alex `[B,16,4,200]`。

这里的 CBraMod 滤波是显式协议覆盖，和已完成 004 一致；当前 LOSO YAML 自身的频率参数为 null，不能说所有数据的原 LOSO YAML 已包含这些滤波。主 recipe 保留 **每 epoch cosine**，这样才与旧 004 CBraMod 结果的数值训练流程一致。

## EEG-FM 参考配方的补充对照

参考仓 `config/` 只有 004、5001 JSON；CBraMod 的 `full` 均是 Fewshot 30%。001 的四分类配方在 README 中，AlexMI 没有本地参考配方，MIRepNet 也不在该仓模型配置中。

以下 CBraMod 补充组使用同样的 LOSO folds、666/667/668 seeds、canonical 4 秒输入和 Flatten head，迁移完整参考 recipe。共同参数：AdamW，LR 1e-3，20 epochs，dropout 0.5，label smoothing 0，5 epoch 从 0 开始线性 warmup，再逐 step cosine 到 min LR 1e-6，eps 1e-8，EA=false。

| 设定 | 配方来源 | batch / weight decay | normalization | 迁移说明 |
|---|---|---|---|---|
| 001 | README 四分类配方外推到左右手子集 | 16 / 0.1 | CAR | 标记为类别子集迁移 |
| 001-4 | README 的 `BNCI2014001 full` | 16 / 0.1 | CAR | LR 为 1e-3；项目 Fewshot 的 3e-4 并非该 README 数值 |
| 004 | `BNCI2014004.json/CBraMod/full` | 16 / 0.01 | None | 原 5 秒重复补齐改为共用 4 秒 |
| 5001 | `BNCI2015001.json/CBraMod/full` | 8 / 0.01 | CAR | 使用全部源被试 trials，保留本次标签映射 |
| AlexMI | 无本地参考配方 | — | — | 先完成主基线，不为其虚构 benchmark 配方 |

补充组是完整 recipe 对照：001/001-4/5001 同时改变了 CAR 和训练参数，差异不能只归因于 LR、epoch 或 optimizer。外层 test 不用于从这几组挑一个“最好 recipe”作为正式成绩。

参考训练器生成逐 step LR 表却在 epoch 末更新，新补充组在每次 optimizer update 前应用 LR 表，明确这是按 warmup/cosine 配置意图修正执行时机。布尔参数读 YAML/JSON 的真实 bool，避免 `type=bool` 将字符串 `False` 解析为 True。

## 训练、选择与指标

第一轮固定训练轮数，在最后一轮完成后评估外层测试被试一次。无独立 validation，也无早停；不使用 `best_val_score` 或最高测试 epoch 作为结果。参考仓返回最终轮模型，因此与其比较应采用 `final_metrics`。

后续如需调参，必须单列 nested tuning：在每个外层 fold 的源被试中，9 人数据用 7 人训练、1 人验证；5001 用 10 人训练、1 人验证；Alex 用 6 人训练、1 人验证，或进一步做内部被试 LOSO。确定配置后重新初始化，用全部 8/11/7 位源被试训练，再评估外层被试。现在已经公开看过 004 测试分数，后续配置探索应完整记录，不能称其测试集从未被查看。

主指标保存 Accuracy、Balanced Accuracy、Kappa，另存 macro F1 和混淆矩阵。二分类可报告 AUROC（正类=1）；001-4 采用多类概率的 macro one-vs-rest AUROC，不能套用现有 binary AUROC。每个 seed 先对该设定的被试等权平均，再计算三个 seed 均值的均值与 sample std。被试间波动另列，不能与跨 seed 标准差混写。

模型比较按同一 dataset×subject×seed 配对。需要显著性检验时，先在每个被试内平均 seed，以被试为单位，不将 seed 当独立被试。五设定分别报告；001/001-4 共享原始 trials，不能把两行当作独立数据集证明来累计样本量。

## 工作量与执行顺序

主实验：`(9+9+9+12+8)×2×3 = 282` 个训练单元。004 已完成 MIRepNet、CBraMod 各 27，若来源哈希、trial 顺序、权重、head、处理与调度一致，可复用 **54** 个，剩 **228** 个。

补充 CBraMod 参考 recipe：`(9+9+9+12)×3 = 117` 个单元。主实验加补充总计 **399** 个；复用旧 004 主组后新增 **345** 个。已跑的 004 EEG-FM 参数组 seeds 0/1/2 属于历史附加结果，不计入新的同 seed 配对组。

执行顺序：

1. 固定来源快照、raw-row trial 清单、标签和 47 折；核对两个模型拿到相同样本。
2. 通用化独立 004 runner，读 dataset 的被试/类别/通道与各自参数；先验证每种输入和分类头、一个优化更新、学习率与 artifact 保存。该阶段结果不计入正式表。
3. 核对并索引已完成的 004 主结果。先以 seed 666 完成另外四个设定两模型的所有被试，再补 667、668；优先 001-4、001、5001，最后 Alex。这个顺序便于先确认四分类与高通道输入、再检查小样本行为，不据成绩改主参数。
4. 对 001/001-4 的 22 通道、5001 的 13 通道和 Alex 的 16 通道先验证 CBraMod batch 64 显存；失败时明确记录新的 microbatch/梯度累积实现版本，按实际样本数加权 loss，每个有效 batch 只更新一次 optimizer 和逐 step LR，末尾不足 batch 同样按实际样本数处理。CBraMod 使用 GroupNorm/LayerNorm，但 dropout 随机数消费顺序与浮点累加仍会使同 seed 的整 batch 和累积结果不完全一致，需要记录实现版本。耗时需要实测，不能直接用 004 三通道训练速度估算。
5. 完成三个 seeds 后检查所有权重/manifest/结果，计算逐被试、逐 seed 与总表。
6. 再按固定的参考 recipe 完成 117 个 CBraMod 补充单元，给出同 seed 配对对照与处理差异。

## 入口改造与结果落盘要求

当前 `run_loso004_reproduction.py` 只有 004、9 位被试、3 通道、二分类，不能直接改 dataset 参数跑五种设定。新入口应扩展这套独立 runner，不能直接复用存在问题的旧 LOSO 入口：`finetune_teacher_loso.py` 只读取模型 defaults，会将 001-4 的 20 epochs 错用为 10；旧统一入口会丢失被试 ID 后混合 EA；CBraMod adapter 的训练 schedule 仍在 epoch 末更新。

新目录按 `protocol/dataset/model/recipe/subject/seed` 区分，避免与历史 `mirepnet_loso` 等缓存混用。每单元保存完整 resolved config、训练/测试被试与 source UID、数据与预训练权重 hash、代码与环境版本、逐 epoch train loss/实际 LR、最终 checkpoint、标签/预测/logits/probabilities/features 与指标。断点续跑先验证完成单元的 manifest 与配置一致，汇总时使用累计全部 seeds，避免部分 seed 恢复时遗漏已完成结果。

这些新增记录对新训练单元必需。旧 004 没有保存完整逐 epoch loss/LR 历史和当时的环境快照；复用时保留其既有证据，并在 import manifest 中明确记为缺失，不事后编造历史，也不把当前环境快照当作旧训练环境。

现有 004 主结果：[MIRepNet summary](../results/reproductions/loso_benchmark_session3_4s_v1/mirepnet/summary.json)（77.1991%），[CBraMod summary](../results/reproductions/loso_benchmark_session3_4s_v1/cbramod/summary.json)（71.1883%）；历史参考参数 seeds 0/1/2：[CBraMod summary](../results/reproductions/loso_eegfm_cbfull_session3_4s_v1/cbramod/summary.json)（71.1343%）。这些为被试等权均值，用于识别既有可复用产物，不作为其他数据调参目标。

## 本地证据

- 数据任务与源场次：[dataset YAML](../configs/datasets/)、[loader](../data/eeg_dataset.py)、[LOSO split 与 legacy UID](../data/split.py)、[既有 raw rows 审计](dataset_selected_rows.csv)。
- 参数与输入：[MIRepNet YAML](../configs/models/mirepnet.yaml)、[CBraMod YAML](../configs/models/cbramod.yaml)、[预处理](../data/preproc.py)、[通道与 2D 位置](../data/channels.py)、[004 独立 runner](../experiments/finetune/run_loso004_reproduction.py)。
- 参考配方：[EEG-FM README](/home/lixinli/EEG-FM-Benchmark/README.md:88)、[004 full JSON](/home/lixinli/EEG-FM-Benchmark/config/BNCI2014004.json:182)、[5001 full JSON](/home/lixinli/EEG-FM-Benchmark/config/BNCI2015001.json:182)。参考仓没有 MIRepNet 配方，且这些 CB 配方均非 LOSO 最优参数声明。

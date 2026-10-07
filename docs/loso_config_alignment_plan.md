# LOSO 配置对齐计划

日期：2026-10-05。依据 [复现审计](/home/lixinli/BigSmallCollab/docs/loso_reproduction_audit_20261005.md)。本文件和配套 YAML 是待执行规格；本轮仅制定计划。

## 目标与首轮范围

先处理可以与论文直接比较的 CBraMod 教师和 EEGNet CE 基线，覆盖 001-4、004、5001，共 30 个 LOSO 被试折。优先定位 CBraMod 的 001-4 差距。

首轮目标是证明数据来源、实际模型输入、初始化、优化器、学习率轨迹和最终轮计分可追溯，并说明哪些配置来自参考 Cross、哪些从 Fewshot 迁移。准确率接近论文不是通过条件。作者完整 Cross 配置及生成环境尚未取得时，结果标记为参考仓库配置对齐或配方迁移评估。

001 二分类和 AlexMI 在第二阶段扩展；MIRepNet、IFNet、ADFCNN 进入后续共同数据协议审查。本计划的首轮工作量不包含这些扩展和 KD 重跑。

## 阶段 0：冻结配置证据与依赖

1. 保存本地参考仓库 commit、关键文件 SHA256、README 命令、JSON 原值、parser 默认值及解析后的实际参数，形成 `reference_resolution.json`。
2. 对 JSON 的真实布尔值和文字命令的 `False→True` 行为分别记录。主配方使用真实布尔 False；文字命令的解析只做证据核对，不自动启用 EA、梯度裁剪或 WD schedule。无法据此判断论文原运行行为。
3. 固定独立数据生成环境，记录 MOABB、MNE、SciPy、NumPy、PyTorch 版本。参考 requirements 只有版本下界，不能假定某一 MOABB 版本就是作者使用的版本。
4. 核实 CBraMod 官方权重获取来源、架构和 SHA256；strict load 成功、backbone 和分类头优化器覆盖、初始化与最终权重均写入 manifest。原权重哈希可用于旧结果溯源，但作者权重同一性仍需对应文件或哈希。
5. 为 EEGNet 补齐模型、adapter、runner、配置和依赖代码哈希。新输出保存初始化权重、RNG 状态、实际学习率/WD 轨迹和训练历史。

交付：锁定环境、参考参数解析表、源码/权重清单。缺少论文原始版本或 Cross 参数时填写 unresolved，不能补写推测值。

## 阶段 1：重建可追溯的数据来源

从参考仓库的数据生成逻辑建立独立缓存。以下是来源处理，模型还需要执行自己的滤波。

| 设定 | 来源滤波 | 场次对应 | 通道／类别 | 所选 trials |
|---|---|---|---|---:|
| 001-4 | MOABB 0.1–75 Hz | 0train ↔ session_T | 22／4 | 2592 = 9×288 |
| 004 | MOABB 0–120 Hz | 3test ↔ session_3 | 3／2 | 1400；S2=120，其余160 |
| 5001 | MOABB 0.1–75 Hz | 0A ↔ session_A | 13／2 | 2400 = 12×200 |

记录原始文件校验和、采样率、事件 tmin/tmax、事件采样位置、通道顺序、单位和全部滤波参数。不要复用预处理历史不明的 `.npy` 作为新协议来源。

新 UID 由原始数据身份、subject、session、run、事件位置/序号和类别名称确定，避免使用重建后数组行号作为唯一身份。生成旧缓存 raw_row 到新 UID 的映射，核查是否一一对应。若事件身份无法证明或有歧义，标记不能配对，不能宣称同 trials，也不能把两组差值直接归因于某一参数。

001 参考生成阶段已裁成前 1000 点，需忠实记录这个步骤；004 保留约 4.5 秒来源 epoch；5001 保留原生 512 Hz 的来源 epoch。新模型处理直接从来源采样率重采样到模型采样率，避免固定经过 250 Hz 中间缓存。

验收：subject/session/run 与原始记录一致；类别均衡、数量和通道正确；UID 唯一；每个外层折训练和测试被试严格不交叉；单位和事件起点有生成记录；采样率和 padding 明确。

## 阶段 2：001-4 来源桥接，保持旧训练配方

在同一冻结实现、环境、初始化、batch 顺序和标签编码下，分别使用旧缓存与重建来源，模型及训练保持旧实验的真实配方：

- CBraMod：旧迁移配方，20 epoch、AdamW、LR=1e-3、batch=16、WD=0.1、head dropout=0.5、CAR、4 秒、step warmup/cosine。
- EEGNet：旧 CE 配方，100 epoch、AdamW、LR=1e-3、batch=32、WD=1e-4、dropout=0.25、4 秒、仅旧入口显式处理、epoch cosine。

这里的新来源也通过同一旧模型处理链，不能同时改成新优化器或新 EEGNet 滤波。两组均按旧 fold 独立重置 RNG，使用 seed 666；每组 9 个外层折、两个模型，共 36 个单元。对旧输出的复用必须同时通过源码、输入、配置、初始状态、batch 顺序等完整指纹核对；EEGNet 历史源码哈希不全，不能默认复用，预算按重新配对训练计算。

这些配置事先固定，外层测试只用于报告桥接差异。重建来源改变的是来源处理链整体，不据此宣称单独滤波或 epoch 起点产生了多少收益。

## 阶段 3：锁定参考配置并正式重跑基线

固定阶段 1 的来源数据，用独立命名配置执行下表。CBraMod 的参数来源依然是 Fewshot；EEGNet 的 Cross 示例只明确对应 001-4，向 004/5001 的使用也标为参数迁移。

| 参数 | EEGNet：Cross 示例及向其它数据集迁移 | CBraMod：001-4 | CBraMod：004 | CBraMod：5001 |
|---|---|---|---|---|
| 初始化 | 随机初始化 | 同一核验预训练文件，每折新模型 | 相同 | 相同 |
| 优化器 | Adam | AdamW | AdamW | AdamW |
| LR | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| Batch | 32 | 16 | 16 | 8 |
| Epoch | 100 | 20 | 20 | 20 |
| Weight decay | 0.01 | 0.1 | 0.01 | 0.01 |
| Dropout | 0.5 | head 0.5；backbone 原结构 | 相同 | 相同 |
| Label smoothing | 0 | 0 | 0 | 0 |
| Class weights | 开，按源训练标签计算 | 关 | 关 | 关 |
| 模型频率 | 8–32 Hz | 0.3–75 Hz | 0.3–75 Hz | 0.3–75 Hz |
| Notch | 60 Hz，Q=30 | 相同 | 相同 | 相同 |
| CAR | 开 | 开 | 关 | 开 |
| EA | 关 | 关 | 关 | 关 |
| 模型采样率 | 250 Hz | 200 Hz | 200 Hz | 200 Hz |
| 时间长度 | 5 秒，来自当前 CLI 默认 | 4 秒 | 5 秒 | 4 秒 |
| LR warmup / min LR | 5 epoch / 1e-6 | 相同 | 相同 | 相同 |

EEGNet 的 001-4 五秒输入为参考来源先裁出的真实四秒加重复开头一秒。004 五秒输入含约四秒半信号加重复补齐；不能标注为五秒真实采集。EEGNet 在 5001 的五秒从原始五秒来源重采样，末端点按参考代码处理。处理采样点数和实际重复长度以运行记录为准。

参考模型处理顺序固定为：`来源 epoch → 重采样到模型 fs → 裁窗/重复补齐 → 4 阶 Butterworth filtfilt → notch filtfilt → EA（关闭）→ CAR（按表）→ float32 Tensor`。001 的来源裁 1000 点发生在这条模型链之前。

EEGNet 对齐参考的非对称 padding、F1=8/D=2/F2=16、kernel=64/16、BN、ELU、pool=4/8 和线性分类头；显式传 dropout。用解析维度或 eval dummy 推导分类头，避免 train dummy 改写 BN/RNG。相对参考不额外增加 maxnorm。

CBraMod 对齐 12 层、d_model=200、native channels、flatten 读出，全参数微调；classifier 维度由实际 patch 数计算，允许 4 或 5 个一秒 patch。旧 adapter 硬编码 1000 点和 4 patch，需要在后续实施时通用化。

学习率采用参考代码的实际轨迹：生成 step 表，第一 epoch 使用 base LR，在 epoch 结束后依据累计 global_step 设置下一 epoch。把逐 step 修正模式作为独立配置，不能混用。LR warmup=5 与此前 KD 的 CE-only warmup10 含义不同。

正式参考组采用 seeds 0/1/2。保留参考的 seed 初始化范围和被试顺序：每个 dataset/model/seed worker 内按被试顺序串行，记录 fold 间连续 RNG 和参考额外模型初始化的消耗；并行按 dataset/model/seed 分配。参考每 epoch 迭代测试 DataLoader，iterator 创建即使不 shuffle 也可能消耗全局 RNG；本计划只在最终轮测试，因此预检必须核对这部分随机数消耗。不能证明一致时记录随机轨迹对齐 unresolved，不通过准确率反向挑选随机流。原实验的每折独立重置模式用于来源桥接和后续共同四秒实验。上述初始化范围对齐本身不证明完整随机轨迹相同。

正式组共 `2 models × 30 folds × 3 seeds = 180` 个单元。三个数据集的参考参数来源不同，报告逐项保留 source_task_mode 和 evidence_level。

## 阶段 4：可选调度诊断与源被试验证

作者实际 Cross 参数缺失时，先完成固定参考配方的报告；不根据外层测试差值挑最高的配方。

如需研究 CBraMod 001-4 的调度因素，事先固定其它设置，比较 `epoch_end_global_step_table` 与 `before_optimizer_step` 两种模式；对九折使用同一诊断 seed、同一每折初始状态和 batch 顺序，最多 18 个单元。该比较用于解释两种方法，不依据 outer test 选择最终配置。

真正调参使用嵌套被试验证：对每个外层测试被试，内部仅使用其余源被试，按被试 ID 确定三个内部验证折并记录。先固定 WD、dropout 和调度，候选为 LR=[1e-4,3e-4,1e-3] × epoch=[20,50]，每个候选重新初始化，按内部被试等权 BCA 选择，固定 tie-break；选完重新初始化、训练全部源被试，只评估一次外层测试。

001-4 的六候选、三个内折、九个外折、一个调参 seed，共 162 个内部训练单元；正式最终配置的三 seeds 另需 27 个 CBraMod 单元。调参结果单独标记 `source_validated_loso`，不冒充作者 Cross 原配置。扩大到 004/5001 或增加 WD/dropout 候选另计工作量。

## 阶段 5：共同四秒协议及 KD 的接续条件

用于大模型—小模型比较的共同四秒组使用同一新来源、trial UID、测试被试和四秒时间范围；各模型保留自己的采样率和预处理。明确固定四秒，采用 step LR 更新、独立 fold 初始化，使用 666/667/668，建立新的 CE 基线。这与参考组的 EEGNet 五秒/004 CBraMod 五秒保持独立协议身份。

两模型、三个首轮数据集、三 seeds 的共同四秒组最多另需 180 个单元；不能因均叫 LOSO 就复用不同窗口、RNG 或调度的参考组结果。只有完整指纹相同的单元才可共享。

共同四秒组同时改变窗口、调度和 RNG 规则，其与参考组的差值不能解释为纯窗口或纯 seed 效应。以后若单独研究 seed 0/1/2 与 666/667/668，需要固定其它因素的额外配对实验，不自动计入当前预算。

扩展到五种设定时，001 二分类由新的 001 四分类来源筛左右手，保持共享 UID；AlexMI 的重复补齐和无直接论文对照身份继续记录。任何缓存变化都需要检查 MIRepNet、IFNet、ADFCNN 及其 CE 对照是否还能完整复用。

教师或模型输入变化后，源训练 logits/features 缓存重新导出；cache key 必须含来源、UID、类映射、窗口、预处理、checkpoint、feature 模式和 seed。新 KD 对照使用同协议的新 CE，初始化和 batch 顺序与该 CE 配对。五秒 CBraMod 的 feature 维度变化必须单列，不能套用旧四秒投影或旧缓存。此前四阶段 KD 的重跑在基线验收后另列计划。

## 工作量、资源与验收

| 项目 | 训练单元数 | 本轮计划属性 |
|---|---:|---|
| 数据/参数解析与功能预检 | 另计少量短任务 | 首轮先行 |
| 001-4 两来源、两模型、seed666 | 36 | 首轮诊断，按不复用预算 |
| 三数据集、两模型、seeds0/1/2 | 180 | 首轮正式对齐 |
| 首轮完整训练合计 | 216 | 不含预检 |
| CBraMod 001-4 调度诊断 | 最多18 | 可选，单列 |
| CBraMod 001-4 嵌套验证 | 162内部＋27最终 | 可选，单列 |
| 首轮三数据集共同四秒基线 | 最多180 | KD 接续阶段 |

不预估未经测量的训练小时数。实施时先测典型单元，把模型训练、数据处理、教师导出、排队分别计时，同时报告累计训练时间和墙钟时间。GPU0 排除；未来根据空闲 GPU 1–7 分配，避免影响其它任务。参考 RNG 组按 dataset/model/seed 并行，桥接和共同四秒组可按独立 fold 并行。

预检需核对同输入下参考与新 wrapper 的处理后数组、eval logits、optimizer 参数覆盖、第一轮和 warmup 后 LR/WD 轨迹。数值比较固定 dtype、版本和容差，记录差异来源，不能仅以 shape 正确作为通过。

最终验收需有完整单元数量、源码/配置/输入/权重指纹、全部 epochs、一次 final 测试、预测结果、无被试交叉、准确率/BCA/Kappa、逐被试和三 seed 统计。Excel 使用显式 dataset_id 映射填列，分别注明 recipe、时长、padding、参数来源和 seed；通过条件不包含“达到论文 Accuracy”。

输出目录为 `/data1/llx/BigSmallCollab_results/reproductions/loso_config_alignment_v2/{profile}/{dataset}/{model}/seed_Y/subject_XX`；缓存放 `/data1/llx/data_cache/eegfm_alignment_v2/`。来源与模型输入的处理记录随缓存保存，旧实验不写入新协议目录。执行入口为 `experiments/finetune/run_loso_config_alignment.py`；目录于 2026-10-07 按用户要求外置，数据和模型不再保存到项目工作树。

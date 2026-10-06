# LOSO 复现审计：CBraMod 与 EEGNet

审计日期：2026-10-05。对照本地 `/home/lixinli/EEG-FM-Benchmark`、实际训练 manifest/history/checkpoint，以及论文 arXiv:2601.17883v2 的表 XV、XVII、XIX。此次审计没有启动训练或修改实验代码。

## 结论

当前结果是采用参考被试划分的固定配方 LOSO 实验，尚不能称为论文的严格数值复现。已确认有配方来源、学习率执行、模型专属预处理和窗口的差异；没有发现 CBraMod 整个 backbone 被误冻结、四分类头类别数错误或遗漏 CBraMod 显式滤波的证据。剩余准确率差距不能仅归因于“参数没调好”，也不能仅凭平均数接近 linear probing 就推断冻结。

用户引用的两个 004 数值实际上对应当前原始 Excel 的 001 二分类列。纠正后，EEGNet 的 004 差距为 -1.17 pp，而非 -7.06 pp；最大的差距仍是 CBraMod 在 001 四分类上的 -11.92 pp。

## 成绩对应与更正

本地可读报告为 [loso_distillation_accuracy_3seed.xlsx](/home/lixinli/BigSmallCollab/results/loso_distillation_accuracy_3seed.xlsx)。表头依次为 `student, method, 004, 001, 001-4, alexmi, 15001`。本地当前没有用户提到的 `loso_distillation_accuracy_3seed_colored.xlsx`；下表从原始工作簿和对应模型 summary 重算，不依赖颜色或截图。

| 数据集 | 模型 | 论文 Accuracy (%) | 当前 Accuracy (%) | 当前－论文 (pp) |
|---|---|---:|---:|---:|
| BNCI2014001 四分类 | CBraMod 全参数微调 | 53.03 | 41.11 | -11.92 |
| BNCI2014001 四分类 | EEGNet CE | 44.97 | 44.44 | -0.53 |
| BNCI2014004 | CBraMod 全参数微调 | 75.45 | 71.23 | -4.22 |
| BNCI2014004 | EEGNet CE | 76.38 | 75.21 | -1.17 |
| BNCI2015001 | CBraMod 全参数微调 | 63.47 | 60.14 | -3.33 |
| BNCI2015001 | EEGNet CE | 63.40 | 61.92 | -1.48 |

论文来源：[arXiv v2 PDF](https://arxiv.org/pdf/2601.17883v2)。论文正文明确使用三个随机种子的最终 epoch 成绩；MI LOSO 每个被试使用一个 session。当前结果同样是先在每个 seed 内对测试被试等权平均，再取三个 seed 的均值。所比较的数据集在每个测试被试内类别平衡，Accuracy 与 Balanced Accuracy 相等，不存在这两种指标混用造成的差距。

当前真实 seed 均值与标准差（标准差针对三个 seed 的被试平均数，ddof=1）：

| 数据集／模型 | seed 666 | seed 667 | seed 668 | mean ± SD (%) |
|---|---:|---:|---:|---:|
| 001-4 CBraMod | 40.7407 | 41.7824 | 40.8179 | 41.1137 ± 0.5804 |
| 001-4 EEGNet | 44.1744 | 44.6373 | 44.5216 | 44.4444 ± 0.2409 |
| 004 CBraMod | 70.2546 | 71.4120 | 72.0139 | 71.2269 ± 0.8941 |
| 004 EEGNet | 75.9491 | 74.3750 | 75.3009 | 75.2083 ± 0.7911 |
| 5001 CBraMod | 60.7917 | 59.2500 | 60.3750 | 60.1389 ± 0.7975 |
| 5001 EEGNet | 62.1250 | 62.0417 | 61.5833 | 61.9167 ± 0.2917 |

当前 seeds 是 666/667/668，参考命令是 0/1/2。每折 RNG 重置方式也不同，因此即使参数全部对齐，也不能期待逐折逐点完全相同。

## CBraMod：实际训练检查

001-4 的 41.1137% 来自 `eegfm_full_recipe_transferred_to_loso4s`，实际参数为 AdamW、LR=1e-3、batch=16、weight decay=0.1、20 epoch、分类头 dropout=0.5、label smoothing=0、5 epoch LR warmup、EA 关闭、CAR 开启。

- 对全部 27 个 fold-seed 的最终 `model.pt` 与预训练文件逐张量比较：每份 209 个 backbone 张量中 208 个变化，唯一未变为分类前向未使用的 `mask_encoding`。可以排除整个 backbone 冻结为 linear probing。
- 优化器包含 `model.parameters()`；每 epoch `model.train()`，正常反传并 `optimizer.step()`。证据：[训练循环](/home/lixinli/BigSmallCollab/experiments/finetune/run_loso_five_datasets.py:298)。
- 27 份训练历史均有 20 epoch；平均 CE loss 在 epoch 1/5/10/15/20 分别为 1.3866/1.3900/0.9580/0.5199/0.2614，说明网络真实学习了源训练数据。低训练损失本身不能证明跨被试泛化成功。
- 分类头为 `Flatten → Dropout(0.5) → Linear(17600,4)`，与参考 22 通道、4 秒、200 Hz、flatten 读出时相同。[本地 adapter](/home/lixinli/BigSmallCollab/models/cbramod/adapter.py:39)、[参考 Loader](/home/lixinli/EEG-FM-Benchmark/models/FM/CBraMod/Loader_CBraMod.py:36)。
- 权重在替换 `proj_out` 前加载，`load_state_dict` 默认 strict=True；实际预训练 SHA256 为 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`，与 source snapshot、实际 manifest 一致。[加载代码](/home/lixinli/BigSmallCollab/models/cbramod/adapter.py:32)。参考仓库期望的 `models/pretrained_models/CBraMod.pth` 当前不存在，无法进一步证明与论文使用权重字节级相同。
- 当前 CBraMod runner、adapter、backbone、Transformer 源码哈希与冻结 source snapshot、蒸馏 teacher cache 中的记录一致，读取当前源码可解释当时训练行为。[源码快照](/home/lixinli/BigSmallCollab/configs/reproductions/manifests/loso_five_datasets_v1_sources.json:17)。
- 实际执行 `0.3–75 Hz`、60 Hz notch；001/001-4/5001 迁移组执行 CAR，004 norm=None。[信号处理](/home/lixinli/BigSmallCollab/models/cbramod/adapter.py:158)。因此不存在 CBraMod 忘记执行显式频率筛选这一已证实问题。

## CBraMod：未对齐的配方与运行语义

本地参考仓库仅提供 004、5001 两个 JSON；其中 CBraMod 配置均为 Fewshot。README 的 001 CBraMod 命令也明确为 Fewshot。未找到 CBraMod Cross/LOSO 的完整真实运行参数或历史日志，论文正文也没有给出对应 LR、batch、epoch 的配置表。因而不能把当前迁移配方称为论文 LOSO 官方超参。

证据：[README Fewshot](/home/lixinli/EEG-FM-Benchmark/README.md:88)、[004 JSON](/home/lixinli/EEG-FM-Benchmark/config/BNCI2014004.json:192)、[5001 JSON](/home/lixinli/EEG-FM-Benchmark/config/BNCI2015001.json:192)、[迁移配方声明](/home/lixinli/BigSmallCollab/configs/reproductions/loso_five_datasets_v1.yaml:175)、[运行元信息](/home/lixinli/BigSmallCollab/experiments/finetune/run_loso_five_datasets.py:219)。

学习率的实际执行有明确差别。参考代码生成逐 step 的 schedule 数组，但只在 epoch 结束后根据累计 `global_step` 设置下一个 epoch 的 LR；第一 epoch 全程使用基准 LR。当前迁移组在每个 optimizer step 前应用 schedule，从 LR=0 做 5 epoch warmup。这是已经明确记录的调度修正，但不等同于逐字运行参考实现。[参考更新位置](/home/lixinli/EEG-FM-Benchmark/utils/trainer.py:305)、[当前更新位置](/home/lixinli/BigSmallCollab/experiments/finetune/run_loso_five_datasets.py:328)。

参考 JSON 的 `reproduce_command` 还包含 `--apply_EA False` 等参数，而 CLI 使用 `type=bool`。逐字运行字符串 `False` 实际会得到 True，可能开启 EA、梯度裁剪、WD schedule、bias/norm 不衰减等。本地迁移按配置的布尔 False 执行。不能据此推断论文运行也受该问题影响，需要论文原日志；README 的 001 命令没有传这些 False 参数。[参考 CLI](/home/lixinli/EEG-FM-Benchmark/run_finetuning.py:151)、[参考 JSON 命令](/home/lixinli/EEG-FM-Benchmark/config/BNCI2015001.json:225)。

参数和处理方案确实影响当前准确率：001-4 主配方（LR=1e-4、50 epoch、batch=64、dropout=0.1、无 CAR 等）为 34.9151%，Fewshot 完整配方迁移后为 41.1137%，增加 6.1986 pp。这是多项设置一起改变的结果，不能归因于单独的 LR，也不能证明剩余 11.92 pp 都可通过调参消除。

## EEGNet：配置与实现差异

检查了全部 141 个 baseline manifest 和完整历史。实际配置均为 100 epoch、batch=32、LR=1e-3、AdamW、WD=1e-4、每 epoch cosine、无 warmup，最终轮一次评估。模型 adapter 未传 dropout，实际使用模型默认 0.25。[当前 adapter](/home/lixinli/BigSmallCollab/models/eegnet/adapter.py:9)、[训练入口](/home/lixinli/BigSmallCollab/experiments/finetune/run_loso_small_baselines.py:85)。

参考 README 的 EEGNet Cross 示例仅针对 BNCI2014001 四分类。它不是所有数据集的通用最佳参数：

| 项目 | 当前 EEGNet | 参考 README Cross 001 示例的实际语义 |
|---|---|---|
| LR / batch / epoch | 1e-3 / 32 / 100 | 1e-3 / 32 / 100 |
| optimizer / weight decay | AdamW / 1e-4 | Adam / 0.01 |
| dropout | 0.25 | CLI 默认 0.5 |
| 显式模型预处理 | canonical 4 秒后转 Tensor | 8–32 Hz + CAR |
| LR schedule | 每 epoch cosine，无 warmup，eta_min=0 | step schedule 表，epoch 末按累计 step 更新；默认 warmup=5、min_lr=1e-6 |
| 时间长度 | 显式 4 秒 | 命令未写；当前 parser 默认 5 秒，会重复补齐 |
| notch | 不额外执行 | 命令未写；当前 parser 默认 60 Hz |

[参考 README](/home/lixinli/EEG-FM-Benchmark/README.md:65)、[参考 CLI 默认](/home/lixinli/EEG-FM-Benchmark/run_finetuning.py:103)、[参考预处理 Loader](/home/lixinli/EEG-FM-Benchmark/models/DL/EEGNet/Loader_EEGNet.py:62)。004/5001 JSON 中 EEGNet 配方也是 Fewshot，不能拿它们直接当论文对应 LOSO 超参。

当前 `_SmallAdapter.preprocess` 仅执行 `torch.as_tensor`，没有附加 EEGNet 专属 8–32 Hz 与 CAR。[当前预处理](/home/lixinli/BigSmallCollab/models/base.py:148)。但现有缓存滤波历史没有完整生成记录，因此只能确认这个显式步骤缺失，不能断言输入缓存绝对未滤波，也不能断言已发生重复滤波。

核心 EEGNet 结构基本相同：F1=8、D=2、F2=16，时间 kernel=64/16，spatial depthwise conv、BN、ELU、pool=4/8、线性分类头。虽然本地类名是 `ResidualEEGNet`，其前向没有残差支路。已读取 checkpoint 核对实际卷积和分类头张量形状。

仍有两个小的实现差异：本地偶数 kernel 使用对称 padding，分别多出一个时间点，参考使用非对称 padding 保持长度；本地推导 flatten 维度时以 train 模式做 dummy forward，更新 BN 一次并消耗 dropout RNG。两边都没有实际实现 maxnorm，不能把 maxnorm 当相对参考的缺项。[本地模型](/home/lixinli/BigSmallCollab/models/eegnet/residual_eegnet.py:16)、[参考模型](/home/lixinli/EEG-FM-Benchmark/models/DL/EEGNet/Model_EEGNet.py:29)。这些差异没有证据证明是性能差的主因。

溯源限制：EEGNet 模型、adapter、小模型 runner、YAML 没有纳入原冻结 snapshot 的模型源码哈希。因此可以确认当前代码与 manifest、历史、checkpoint 形状一致，但不能声称对训练时所有 EEGNet 源码做了完整哈希验证。

## 数据和计分协议

已实际核对 trial manifest 的 raw row 与源 meta 的 subject/session；所选 trials、类别数和类别均衡情况如下：

| 设定 | 场次 | 被试数 | 总 trials | 每被试测试 trials |
|---|---|---:|---:|---|
| 001-4 | session_T，对应参考 0train | 9 | 2592 | 288 |
| 004 | session_3，对应参考 3test | 9 | 1400 | S2=120，其余160 |
| 5001 | session_A，对应参考 0train | 12 | 2400 | 200 |

当前完整留一被试、其余全部所选被试训练，与参考 split 规则一致。模型训练和评估的类别编码一致；001-4 和 5001 与参考标签数字只是类别置换，未发现置换不一致导致的指标错误。最终 epoch 评估也与参考返回 final model 的行为一致。[当前分折和评估](/home/lixinli/BigSmallCollab/experiments/finetune/run_loso_five_datasets.py:645)、[参考分折](/home/lixinli/EEG-FM-Benchmark/utils/dataset_split.py:14)。

参考数据生成阶段也有滤波：001 与 5001 使用 `MotorImagery(fmin=0.1, fmax=75)`，004 使用 `MotorImagery(fmin=0, fmax=120)`。这是模型前处理之前的来源处理，不能只对齐 adapter 中的滤波参数就宣称输入相同。[001 数据生成](/home/lixinli/EEG-FM-Benchmark/datasets/BNCI2014001/Preprocess_Dataset.py:19)、[5001 数据生成](/home/lixinli/EEG-FM-Benchmark/datasets/BNCI2015001/Preprocess_Dataset.py:23)、[004 数据生成](/home/lixinli/EEG-FM-Benchmark/datasets/BNCI2014004/Preprocess_Dataset.py:23)。

窗口和信号来源仍有差别或未解决事项：

- 001-4 模型实际 4 秒，与 CBraMod README 明确的 4 秒一致。
- 004 原始 epoch 约 4.5 秒；参考 CBraMod Fewshot JSON 指定补齐到 5 秒，当前统一裁成真实前 4 秒。不能把这两组当相同输入。Cross 的实际参数未取得。
- 5001 原 epoch 5 秒，但本地参考 CBraMod Fewshot JSON 实际写 `time_length=4.0`，因此当前 4 秒与这个配置一致。当前先 512→250 Hz 再裁窗和 250→200 Hz，参考直接 512→200 Hz；重采样实现与处理顺序不同。不能从论文数据集描述的 5 秒推断 CBraMod Cross 一定用了 5 秒。[5001 JSON](/home/lixinli/EEG-FM-Benchmark/config/BNCI2015001.json:223)。
- 源缓存缺少完整事件起点、已有滤波和单位生成记录；现有证据仅支持微伏量级。当前没有参考历史 pkl 或原始运行日志可直接逐 trial 比对。[源快照](/home/lixinli/BigSmallCollab/configs/reproductions/manifests/loso_five_datasets_v1_sources.json:139)。

## 后续修正优先级

1. 先建立可追溯的 MOABB 宽带缓存，并锁定版本、事件 epoch 起点、单位、通道顺序、session 和 trial UID；保留现有结果作为旧配方。
2. 增加单独命名的 EEGNet benchmark 配方，显式写入 8–32 Hz、CAR、dropout、optimizer/WD、窗口、notch 和学习率执行模式，避免依赖 CLI 默认值。001 Cross 示例可核对，其它数据集仍需确认真正的 LOSO 配置。
3. 优先处理 CBraMod 001-4：在同一可追溯输入下比较参考代码的 LR 轨迹与当前修正轨迹，核实公开预训练权重身份，再通过外层训练被试内部的被试验证选择 LR、epoch、WD 和 dropout。缺作者 Cross 参数时应记录为重新配置的 LOSO 评估，而不是已证明严格复现。
4. 004 的 5 秒兼容组保持独立标记；不能把结果合并进统一 4 秒主组。先把教师及 CE 基线对齐，再讨论是否重跑依赖这些教师/基线的 KD 结果。
5. 所有新配方重新初始化，在内部源被试验证上选择；外层测试成绩不用于追求论文数字或选择最佳 checkpoint。

这次审计已经定位确定差异并排除若干实现故障，但尚未证明 CBraMod 剩余差距的单一因果来源。不能用“调参数”跳过数据来源和实际 Cross 配方的核对。

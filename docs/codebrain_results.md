# CodeBrain 复现结果与负结果记录

> **当前项目决定（2026-09-29）：** CodeBrain 不纳入活动的 BigSmallCollab 协同 baseline 或 teacher 集合。完成参数搜索后的正式三 seed 结果仍接近机会水平，且总体低于 MIRepNet 和 CBraMod。保留实现、配置与结果仅用于复现追溯和负结果报告。最新完整结果见 [tuned D2 run 报告](../results/codebrain/codebrain_tuned_d2_cpu_20260929/final_report.md)；下文早期 run 结果作为历史记录保留。

本报告按已生成的比较与消融产物整理，未筛选被试、seed 或指标。表内展示值四舍五入至三位小数；原始全精度数值见对应 CSV。准确率与平衡准确率以百分比表示，配对差值的单位为百分点；Kappa 无量纲。

## 按更新后的当前 YAML 重跑（2026-09-27 至 28）

此部分对应 run ID `codebrain_yaml_rerun_20260927`，与后面的旧版固定参数结果分开。完整配置快照、逐被试配置、split manifest、权重加载记录、训练 history、checkpoint 和测试预测均保存在该 run 目录。

- 覆盖 4 个数据集、38 个被试、seed 666/667/668，共 114 次完整训练。训练/测试划分为 30%/70%；前三个 BNCI 数据集各训练 20 epoch，AlexMI 训练 50 epoch；固定使用最后一轮 checkpoint。
- 当前 YAML 的设置为 LR `1e-3`、weight decay `0.05`、batch size `64`、dropout `0.3`、min LR `1e-4`、按 batch 更新的 cosine scheduler、gradient clipping `5`、label smoothing `0`、输入 scale divisor `100`。模型配置 SHA256：`599d7fa0ddedd8d3d32bcc4f1db606fd38640792d057e5ed550183ae733b7a3b`。
- 这次确实执行了全量微调：作者预训练 encoder 的 269/269 项权重均 strict load，missing/unexpected keys 都为 0。总参数 29,409,002，其中 29,408,802 可训练；仅保留官方 `mask_encoding` 的 200 个参数固定。CodeBrain checkpoint 不含当前任务分类头；本项目适配实现将任务头重新初始化为三层 MLP `[17600, 800, 200, 2]`。
- 输入按配置重采样到 200 Hz、保留原生通道及顺序，并按固定 µV 尺度除以 100；每个 4 秒 trial 转为四个 1 秒 patch。
- 训练样本数小于 batch size 64，因此每个 epoch 只有一次优化器更新。history 中有训练 loss 明显升高的轮次；这提示当前配方的优化过程不稳定，但单凭这些结果不能确定坍塌的单一原因。

三 seed 平均结果按 subject × seed 等权汇总。`塌缩` 表示测试预测只包含一个类别；CodeBrain 一共 **112/114（98.25%）** 个 subject-seed 单元发生单类预测。

| 数据集 | 模型 | 被试数 | Accuracy | Balanced Accuracy | Kappa | 单类预测 |
|---|---|---:|---:|---:|---:|---:|
| BNCI2014001 | CodeBrain | 9 | 49.945 | 50.000 | 0.000 | 27/27 |
| BNCI2014001 | MIRepNet | 9 | 80.198 | 80.228 | 0.604 | 0/27 |
| BNCI2014001 | CBraMod | 9 | 68.133 | 68.153 | 0.363 | 0/27 |
| BNCI2014004 | CodeBrain | 9 | 50.033 | 50.033 | 0.001 | 26/27 |
| BNCI2014004 | MIRepNet | 9 | 81.548 | 81.548 | 0.631 | 0/27 |
| BNCI2014004 | CBraMod | 9 | 76.433 | 76.433 | 0.529 | 0/27 |
| BNCI2015001 | CodeBrain | 12 | 50.179 | 50.179 | 0.004 | 35/36 |
| BNCI2015001 | MIRepNet | 12 | 80.952 | 80.952 | 0.619 | 0/36 |
| BNCI2015001 | CBraMod | 12 | 68.869 | 68.869 | 0.377 | 0/36 |
| AlexMI | CodeBrain | 8 | 50.000 | 50.000 | 0.000 | 24/24 |
| AlexMI | MIRepNet | 8 | 67.560 | 67.560 | 0.351 | 0/24 |
| AlexMI | CBraMod | 8 | 54.464 | 54.464 | 0.089 | 1/24 |
| **总体** | **CodeBrain** | **38** | **50.051** | **50.064** | **0.001** | **112/114** |
| **总体** | **MIRepNet** | **38** | **78.095** | **78.102** | **0.562** | **0/114** |
| **总体** | **CBraMod** | **38** | **67.454** | **67.458** | **0.349** | **1/114** |

按相同 UID 对齐后，将每个被试的三个 seed 平均，再做配对 subject bootstrap（10,000 次，取自 comparison 汇总）。总体 Accuracy 差值为：CodeBrain−MIRepNet `-28.04` 个百分点（95% CI `[-32.59, -23.24]`，胜/平/负 `1/0/37`）；CodeBrain−CBraMod `-17.40` 个百分点（95% CI `[-22.30, -12.38]`，胜/平/负 `3/2/33`）。seed 666 首轮单独结果为 CodeBrain 49.984%、MIRepNet 77.737%、CBraMod 65.954%；CodeBrain 在该 seed 的 38 个被试中有 37 个单类预测。

所有 342/342 个模型测试工件都存在且有效；UID、标签和 split policy 对齐不匹配为 0，30/30 个配对汇总项完整。该配置下 CodeBrain 接近机会水平并出现大范围单类预测，因此可保留作负结果或架构对照，当前不适合作为有效的大模型教师基线。

重跑产物：[resolved 配置](../results/codebrain/codebrain_yaml_rerun_20260927/config_resolved_all.json)、[逐单元结果](../results/codebrain/codebrain_yaml_rerun_20260927/aggregate/aggregate_cells_pretrained_seed666-667-668.csv)、[汇总指标](../results/codebrain/codebrain_yaml_rerun_20260927/comparison/summary_by_dataset.csv)、[配对统计与 bootstrap CI](../results/codebrain/codebrain_yaml_rerun_20260927/comparison/paired_comparisons.csv)、[UID 对齐检查](../results/codebrain/codebrain_yaml_rerun_20260927/comparison/alignment_checks.csv)、[逐模型运行明细](../results/codebrain/codebrain_yaml_rerun_20260927/comparison/run_metrics.csv)。

以下章节记录更早一版固定参数方案的结果，不能与本次更新 YAML 的结果混为一组。

## 上一版固定参数方案：三个 seed 的主结果

主结果覆盖 seed 666、667、668。每个被试先对三个 seed 求均值，再对被试等权汇总；因此总体不是按测试 trial 数加权。`n` 为被试数。

| 数据集 | 模型 | n | Accuracy | Balanced Accuracy | Kappa |
|---|---|---:|---:|---:|---:|
| BNCI2014001 | MIRepNet | 9 | 80.198 | 80.228 | 0.604 |
| BNCI2014001 | CBraMod | 9 | 68.133 | 68.153 | 0.363 |
| BNCI2014001 | CodeBrain | 9 | 49.542 | 49.532 | -0.009 |
| BNCI2014004 | MIRepNet | 9 | 81.548 | 81.548 | 0.631 |
| BNCI2014004 | CBraMod | 9 | 76.433 | 76.433 | 0.529 |
| BNCI2014004 | CodeBrain | 9 | 56.515 | 56.515 | 0.130 |
| BNCI2015001 | MIRepNet | 12 | 80.952 | 80.952 | 0.619 |
| BNCI2015001 | CBraMod | 12 | 68.869 | 68.869 | 0.377 |
| BNCI2015001 | CodeBrain | 12 | 51.984 | 51.984 | 0.040 |
| AlexMI | MIRepNet | 8 | 67.560 | 67.560 | 0.351 |
| AlexMI | CBraMod | 8 | 54.464 | 54.464 | 0.089 |
| AlexMI | CodeBrain | 8 | 52.232 | 52.232 | 0.045 |
| **总体** | **MIRepNet** | **38** | **78.095** | **78.102** | **0.562** |
| **总体** | **CBraMod** | **38** | **67.454** | **67.458** | **0.349** |
| **总体** | **CodeBrain** | **38** | **52.531** | **52.529** | **0.051** |

### 38 被试配对 Accuracy 差值

差值定义为 CodeBrain 减去对照模型。区间为按被试重采样的 95% bootstrap CI（10,000 次）；胜/平/负按每个被试的配对差值计数。每个数据集内先将该被试的三个 seed 求均值，再参与配对统计。

| 范围 | 对照 | Δ Accuracy | 95% CI | 胜/平/负 | 被试数 |
|---|---|---:|---:|---:|---:|
| BNCI2014001 | MIRepNet | -30.656 | [-38.467, -22.442] | 0/0/9 | 9 |
| BNCI2014001 | CBraMod | -18.592 | [-28.713, -9.168] | 1/0/8 | 9 |
| BNCI2014004 | MIRepNet | -25.033 | [-31.614, -18.452] | 0/0/9 | 9 |
| BNCI2014004 | CBraMod | -19.918 | [-27.910, -11.948] | 0/1/8 | 9 |
| BNCI2015001 | MIRepNet | -28.968 | [-35.437, -21.806] | 0/0/12 | 12 |
| BNCI2015001 | CBraMod | -16.885 | [-24.683, -9.067] | 1/1/10 | 12 |
| AlexMI | MIRepNet | -15.327 | [-27.381, -3.274] | 2/0/6 | 8 |
| AlexMI | CBraMod | -2.232 | [-6.845, 2.530] | 4/0/4 | 8 |
| **总体** | **MIRepNet** | **-25.564** | **[-30.014, -20.848]** | **2/0/36** | **38** |
| **总体** | **CBraMod** | **-14.923** | **[-19.440, -10.379]** | **6/2/30** | **38** |

总体 CodeBrain Accuracy 为 52.531%，MIRepNet 为 78.095%，CBraMod 为 67.454%。相对两项对照的总体配对 CI 均低于零。

## 上一版固定参数方案：Seed 666 首轮结果

以下只反映 seed 666 的首轮结果，与上面的三 seed 主表分开报告；每个被试只有一个 seed。首轮覆盖 38 个被试，结果如下。

| 数据集 | 模型 | n | Accuracy | Balanced Accuracy | Kappa |
|---|---|---:|---:|---:|---:|
| BNCI2014001 | MIRepNet | 9 | 79.538 | 79.542 | 0.591 |
| BNCI2014001 | CBraMod | 9 | 65.787 | 65.813 | 0.316 |
| BNCI2014001 | CodeBrain | 9 | 47.855 | 47.843 | -0.043 |
| BNCI2014004 | MIRepNet | 9 | 82.176 | 82.176 | 0.644 |
| BNCI2014004 | CBraMod | 9 | 75.066 | 75.066 | 0.501 |
| BNCI2014004 | CodeBrain | 9 | 56.812 | 56.812 | 0.136 |
| BNCI2015001 | MIRepNet | 12 | 80.536 | 80.536 | 0.611 |
| BNCI2015001 | CBraMod | 12 | 67.798 | 67.798 | 0.356 |
| BNCI2015001 | CodeBrain | 12 | 51.310 | 51.310 | 0.026 |
| AlexMI | MIRepNet | 8 | 66.518 | 66.518 | 0.330 |
| AlexMI | CBraMod | 8 | 53.125 | 53.125 | 0.062 |
| AlexMI | CodeBrain | 8 | 50.893 | 50.893 | 0.018 |
| **总体** | **MIRepNet** | **38** | **77.737** | **77.738** | **0.555** |
| **总体** | **CBraMod** | **38** | **65.954** | **65.960** | **0.319** |
| **总体** | **CodeBrain** | **38** | **51.707** | **51.704** | **0.034** |

seed 666 的总体配对 Accuracy：CodeBrain−MIRepNet 为 -26.030 个百分点（95% CI [-31.552, -19.953]，胜/平/负 4/0/34）；CodeBrain−CBraMod 为 -14.247 个百分点（95% CI [-19.513, -8.882]，胜/平/负 6/1/31）。

## 预训练消融：seed 666

此消融在相同 seed、划分和训练规则下比较预训练初始化与同架构随机初始化。差值为 pretrained 减 random；区间按被试 bootstrap。以下列出 Accuracy、Balanced Accuracy 和 Kappa 的配对均值、区间及胜/平/负。

| 范围 | 指标 | Pretrained | Random | Δ（95% CI） | 胜/平/负 |
|---|---|---:|---:|---:|---:|
| BNCI2014001 | Accuracy (%) | 47.855 | 51.595 | -3.740 [-9.244, 1.430] | 3/0/6 |
| BNCI2014001 | Balanced Accuracy (%) | 47.843 | 51.612 | -3.769 [-9.159, 1.336] | 3/0/6 |
| BNCI2014001 | Kappa | -0.043 | 0.032 | -0.076 [-0.187, 0.026] | 3/0/6 |
| BNCI2014004 | Accuracy (%) | 56.812 | 53.208 | 3.604 [-4.730, 14.683] | 3/1/5 |
| BNCI2014004 | Balanced Accuracy (%) | 56.812 | 53.208 | 3.604 [-4.828, 14.583] | 3/1/5 |
| BNCI2014004 | Kappa | 0.136 | 0.064 | 0.072 [-0.097, 0.296] | 3/1/5 |
| BNCI2015001 | Accuracy (%) | 51.310 | 51.488 | -0.179 [-2.619, 2.440] | 5/1/6 |
| BNCI2015001 | Balanced Accuracy (%) | 51.310 | 51.488 | -0.179 [-2.619, 2.440] | 5/1/6 |
| BNCI2015001 | Kappa | 0.026 | 0.030 | -0.004 [-0.051, 0.049] | 5/1/6 |
| AlexMI | Accuracy (%) | 50.893 | 54.018 | -3.125 [-9.375, 3.571] | 2/2/4 |
| AlexMI | Balanced Accuracy (%) | 50.893 | 54.018 | -3.125 [-9.375, 3.571] | 2/2/4 |
| AlexMI | Kappa | 0.018 | 0.080 | -0.063 [-0.188, 0.071] | 2/2/4 |
| **总体** | **Accuracy (%)** | **51.707** | **52.453** | **-0.746 [-3.744, 2.668]** | **13/4/21** |
| **总体** | **Balanced Accuracy (%)** | **51.704** | **52.457** | **-0.753 [-3.746, 2.708]** | **13/4/21** |
| **总体** | **Kappa** | **0.034** | **0.049** | **-0.015 [-0.075, 0.053]** | **13/4/21** |

所有总体消融 CI 均包含零；这组 seed 666 结果没有显示预训练初始化稳定优于同架构随机初始化。

## 覆盖、坍塌和解释范围

- 三 seed 汇总包含 4 个数据集、38 个被试、3 个模型和 3 个 seed，共 342 个模型测试工件；342/342 均存在且有效，缺失 0、无效 0，UID、标签和 `split_policy` 对齐不匹配数为 0。
- seed 666 首轮比较覆盖 114/114 个模型测试工件；缺失、无效及对齐不匹配均为 0。
- 三 seed 的 CodeBrain 预训练模型有 0/114 个单类预测坍塌；seed 666 随机初始化消融有 2/38 个坍塌，均在 BNCI2014004。完整三 seed 基线中，CBraMod 有 1/114 个坍塌（AlexMI 1/24），MIRepNet 为 0/114。
- 该历史 run 使用项目被试内协议：训练数据占 30%、测试数据占 70%；BNCI2014001/4004/2015001 训练 20 epoch，AlexMI 训练 50 epoch，均使用预先固定的末轮 checkpoint。论文 SHU-MI 的数据划分和评估协议不同，不能把其分数与本表直接比较。
- 按作者披露的信息，CodeBrain 预训练数据来源为 TUEG；作者未公开逐记录的预训练清单。因此只能表述为“根据披露的来源未发现与这四套下游数据重叠”，不能声称已逐记录验证无重叠。

**历史结论：**早期配置下 CodeBrain 的表现低于 MIRepNet 与 CBraMod；seed 666 预训练对照也未显示稳定正收益。后续调参后的正式结果见本文开头链接。根据当前项目决定，CodeBrain 不作为活动协同 baseline 或 MI 教师；这些实验仅作为架构复现与负结果记录。

## 原始汇总文件

- 三 seed 主汇总：[summary_by_dataset.csv](../results/codebrain/comparison/summary_by_dataset.csv)、[summary_overall.csv](../results/codebrain/comparison/summary_overall.csv)、[paired_comparisons.csv](../results/codebrain/comparison/paired_comparisons.csv)、[per_seed_summary.csv](../results/codebrain/comparison/per_seed_summary.csv)。
- 首轮 seed 666：[comparison report](../results/codebrain/comparison_seed666/comparison_report.md)、[paired comparisons](../results/codebrain/comparison_seed666/paired_comparisons.csv)，以及逐 run 预测数/坍塌记录。
- 随机初始化消融：[report.json](../results/codebrain/pretrain_ablation_seed666/report.json)、[report.md](../results/codebrain/pretrain_ablation_seed666/report.md)、[runs.csv](../results/codebrain/pretrain_ablation_seed666/runs.csv)。
- 对齐与逐 run 明细见三 seed 的 [comparison_report.md](../results/codebrain/comparison/comparison_report.md) 和 [run_metrics.csv](../results/codebrain/comparison/run_metrics.csv)。

# 大模型向小模型蒸馏的五设定 LOSO 实验

实验规格：[loso_distillation_v1.yaml](../configs/reproductions/loso_distillation_v1.yaml)。数据协议沿用 `loso_five_settings_canonical4s_v1`，实验输出独立保存到 `results/distill/loso_five_settings_kd_feature_warmup10_v1`。执行顺序固定为 **logits-KD → KD＋feature → warmup10-KD → warmup10-KD＋feature**，每个阶段完成全部教师、学生、数据设定和 seed 后进入下一阶段。

## 数据、模型和教师来源

沿用同一份 [trial 清单](../configs/reproductions/manifests/loso_five_datasets_v1_trials.csv)、[来源哈希快照](../configs/reproductions/manifests/loso_five_datasets_v1_sources.json) 和 [47 折清单](loso_five_datasets_folds.csv)。每个外层 fold 完整留出一位被试，其他被试的全部所选 trials 用于教师训练和学生训练。五种设定为 `BNCI2014001`、`BNCI2014001-4`、`BNCI2014004`、`BNCI2015001`、`AlexMI`，seed 为 **666、667、668**。类别顺序、场次筛选和 canonical 输入窗口保持原协议；AlexMI 输入仍为 3 秒采集信号加重复开头 1 秒，001 与 001-4 共享部分原始 trials。

两位教师为 MIRepNet、CBraMod，三个学生为 IFNet、EEGNet、ADFCNN。每种数据设定执行全部六个有方向的组合：MIRepNet→IFNet、MIRepNet→EEGNet、MIRepNet→ADFCNN、CBraMod→IFNet、CBraMod→EEGNet、CBraMod→ADFCNN。每个学生对两个教师分别训练，使用单教师的 logits 和特征。

| 教师 | 使用的已完成 LOSO recipe | 输入处理 |
|---|---|---|
| MIRepNet | `project_loso_mi8_30_subject_ea_idw45`，全部五种设定 | 8–30 Hz、按源被试分别 EA、45 通道逆距离插值 |
| CBraMod | 前四种设定 `eegfm_full_recipe_transferred_to_loso4s` | 200 Hz、0.3–75 Hz、60 Hz notch；001、001-4、5001 使用 CAR，004 不做 CAR |
| CBraMod / AlexMI | `project_loso_filtered_native4s_flatten` | 200 Hz、0.3–75 Hz、60 Hz notch，保留原生通道 |

每个教师 checkpoint 必须对应当前外层留出被试与同一个 seed，并已排除该被试全部 trials。蒸馏阶段教师 `eval()`、冻结参数，不重新微调。缓存只包含该 fold 的源训练 trials，使用 `(source_dataset, raw_cache_row)` 对齐学生样本与类别顺序。允许教师和学生保留各自输入处理，同一个 UID 指向同一个原始 trial。

教师缓存是**教师在其已参与训练的源 trials 上生成的 in-sample 目标**。这版不做源被试 out-of-fold 教师目标；其结果解释为常规源数据蒸馏，不能表述成对源被试也做了交叉拟合。

MIRepNet 已完成教师基线评估使用目标被试全部无标签 trials 的 EA，因此历史教师分数仍属于 transductive LOSO。源训练缓存的 EA 按各源被试分别估计；蒸馏学生不使用目标被试标签、EA、共享统计或适配步骤。学生保持自己的原始基线输入：IFNet 使用 4–16 Hz 和 16–40 Hz 双分支滤波，EEGNet/ADFCNN 使用 canonical 原生通道输入。

## 四个依次执行的消融组

CE 基线复用已经完成的 `scratch_supervised_loso_v1`，计分使用固定最后一轮，三位学生均为 100 epochs。以下四组分别从相同 seed 对应的新随机初始化开始训练；后一阶段不接着前一阶段权重训练。

| 顺序 / 配置名称 | 第 1–10 epoch | 第 11–100 epoch | 总 epochs |
|---|---|---|---:|
| 1 / `logits_kd` | CE＋0.5 KD | CE＋0.5 KD | 100 |
| 2 / `kd_feature` | CE＋0.5 KD＋0.5 feature | CE＋0.5 KD＋0.5 feature | 100 |
| 3 / `warmup10_kd` | CE | CE＋0.5 KD | 100 |
| 4 / `warmup10_kd_feature` | CE | CE＋0.5 KD＋0.5 feature | 100 |

**10 warmup 指前 10 个完整 epoch 只训练 CE，第 11 个 epoch 起开启蒸馏。**总训练轮数仍为 100；它不改变学习率 warmup，也不是先训练 10 epoch 再训练 100 epoch。开启时直接采用表中的固定权重，不另加权重渐增。CE 权重固定为 1.0，温度固定为 2.0；这版全部 trials 都参与 KD，不加 teacher-correct mask、MI、原型损失或额外样本筛选。

设学生 logits 为 `z_s`，冻结教师 logits 为 `z_t`，类别数为 K。每 batch 的损失为：

```text
L_CE = mean_i CrossEntropy(z_s[i], y[i])
L_KD = T² × mean_i sum_k p_t[i,k] × (log p_t[i,k] − log p_s[i,k])
p_t = softmax(z_t / T), p_s = softmax(z_s / T), T = 2
L_feature = mean_i (1 − cosine(P(f_s[i]), f_t[i]))
L_total = L_CE + lambda_KD × L_KD + lambda_feature × L_feature
```

KL 使用 batch mean，即先对每个 trial 的类别维求和，再对 trials 求平均；不能把类别维也平均而让二分类与四分类的 KD 权重静默不同。教师 logits 与特征均 `detach`；cosine 使用明确的小正数 epsilon 防止零向量导致 NaN。warmup 期间两个 lambda 均为 0。

## Feature 对齐

使用 adapter 已暴露的原生分类前特征；CBraMod 使用 backbone 的 `C×4×200` 全部 token 展平、分类头 dropout 之前的表示。MIRepNet 使用 256 维 pooled 表示。教师特征维度如下，001 与 001-4 相同维度但分别有自己的四分类 / 二分类 checkpoint：

| 教师 / 数据设定 | 001 | 001-4 | 004 | 5001 | AlexMI |
|---|---:|---:|---:|---:|---:|
| MIRepNet | 256 | 256 | 256 | 256 | 256 |
| CBraMod Flatten | 17600 | 17600 | 2400 | 10400 | 12800 |

学生特征维度在构建模型后实际读取并记录。以可训练线性层 `P: student_dim → teacher_dim` 对齐维度，对投影结果和教师特征计算 cosine 距离。教师特征不降维、不均值池化、不取 logits 冒充特征。投影层和学生共同训练，推理时丢弃投影层，仅使用学生分类头。高通道 CBraMod 的投影层可能增加数百万训练参数，须另存学生参数量、投影层参数量、训练耗时和显存；部署学生大小只计算学生本体。这组检验的是向教师完整原生表示对齐的效果。

## 学生训练与配对

所有训练参数首先核对已完成学生基线的 manifest，不以当前 YAML 覆盖历史实际参数。预期设置如下：

| 学生 | optimizer / LR | batch | weight decay | epochs / 调度 |
|---|---|---:|---:|---|
| IFNet | AdamW / 1e-3 | 16 | 0.01 | 100 / 每 epoch cosine |
| EEGNet | AdamW / 1e-3 | 32 | 1e-4 | 100 / 每 epoch cosine |
| ADFCNN | AdamW / 1e-3 | 32 | 1e-4 | 100 / 每 epoch cosine |

CE 不使用 label smoothing；所有组输入处理、dropout、批次大小、优化器和学习率执行时机与基线一致。每个 dataset×student×fold×seed 先按基线构建顺序设 seed 并初始化学生，保存初始化权重 hash；两个教师与四个阶段复用该初始状态。特征投影层在学生初始化之后单独创建，保存并恢复创建前的全局随机状态，避免它影响学生 dropout 或后续随机过程。DataLoader 使用与基线一致的独立 `torch.Generator().manual_seed(seed)`、shuffle、`num_workers=0`，保存生成器状态，保证相同 seed 的训练 trial 批次顺序一致。

warmup10 组前 10 epoch 应在训练数据、学生初始权重、批次顺序、随机状态和优化器上与 CE 基线一致；投影层在 warmup 内不更新、也不消耗改变学生随机过程的算子。第 11 epoch 开始的差异才来自 KD / feature。基线初始化与 shuffle 不一致时停止该单元，修复后重新开始，不将其当作配对对照。

不依据外层测试准确率选择教师 seed、checkpoint、temperature、损失权重或训练轮数。后续调参需要另立版本，限定在外层源被试内部的被试验证；已经查看过的教师外层成绩应如实注明。当前教师 recipe 是预先固定的已有 recipe，没有按每个外层折的测试成绩动态择优。

## 工作量、执行与恢复

五设定共有 `9＋9＋9＋12＋8 = 47` 折。每阶段 `47 折 × 3 seeds × 2 教师 × 3 学生 = 846` 个训练单元；四阶段合计 **3384** 个。教师源训练缓存 `47 × 3 × 2 = 282` 份，只需导出一次，四阶段和三个学生共同读取。已有 CE 基线 423 个训练单元直接作为对照，不重复训练。

执行入口为 `experiments/distill/run_loso_distillation.py`，顺序启动器为 `experiments/distill/run_loso_distillation_suite.sh`。先校验现有产物与缓存，再启动 `logits_kd`；全部 846 个有效完成单元汇总后进入 `kd_feature`，随后运行两组 warmup10。阶段内按照固定数据设定、教师、学生、fold、seed 顺序执行。核对共享服务器作业属于 `lixinli` 后选择可用资源，使用持久会话和实时日志。耗时根据实际已完成单元估算，不直接将 IFNet 单折速度外推到所有学生与 feature 大投影层。

正式启动前核对 trial / source hashes、fold 留出被试、训练 UID 相等且测试 UID 不相交、checkpoint hash、类别顺序、标签、logits / features shape 与有限值、教师评估模式、学生初始状态 hash、批次 UID 序列。各教师×学生组合执行一次有限梯度和优化更新审计，包括四分类与 feature 投影；确认教师没有梯度、学生和投影层梯度有效、warmup 前 10 epoch 蒸馏权重为零。这些审计不作为正式成绩。

每 5 个 epoch 原子写入恢复 checkpoint，保存学生和投影层参数、optimizer、scheduler、已完成 epoch、history、Python / NumPy / Torch CPU / CUDA RNG、DataLoader generator 状态及完整配置和输入 hash。恢复时校验所有来源与配置相同，从最后完整保存 epoch 继续；完成的单元通过产物与 hash 校验后跳过。持久进程状态和日志明确记录启动、失败、恢复与结束时间，避免对话结束导致任务中断。错误采用 fail-fast，先保留错误证据，再处理和断点恢复。

## 结果与报告

每个 dataset×teacher→student×stage×fold×seed 保存 resolved config、来源和代码 hash、训练 / 测试 UID、教师 checkpoint 与训练缓存 hash、学生初始化 hash、最终学生权重、辅助投影层权重、逐 epoch CE / KD / feature / total loss 和实际 LR、恢复状态、最终测试标签 / logits / probabilities / 预测 / 特征、Accuracy、Balanced Accuracy、Kappa、macro F1、混淆矩阵与耗时。二分类 AUROC 正类为 1；四分类使用 macro one-vs-rest AUROC。

测试被试仅在固定第 100 epoch 结束后评估一次，正式成绩采用最终模型。每 seed 先对所有被试等权平均，再报告三个 seed 的均值与 sample std；完整被试和三个 seeds 未齐时只能报告进度或注明部分结果，不与完成组作正式排名。同步保存相对于同 dataset×student×fold×seed CE 基线的 Accuracy / BAcc / Kappa 差值，先在被试内平均三个 seeds，再以被试为单位比较 KD 与基线、feature 与纯 KD、warmup 与无 warmup。001 / 001-4 的重叠来源单独说明。

主表按数据设定逐行报告 CE、两位教师、四组蒸馏的学生成绩与配对增益。阶段完整后更新该阶段汇总，不从部分完成的 folds 推断总体改善。结果目录使用独立协议名，并记录每位教师固定 recipe；AlexMI 的 CBraMod fallback 和 MIRepNet 教师历史 transductive EA 均在最终报告注明。

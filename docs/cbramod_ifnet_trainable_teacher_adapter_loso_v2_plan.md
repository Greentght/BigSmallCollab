# CBraMod → IFNet：可训练教师侧 adapter LOSO 实验方案

日期：2026-11-03。版本：`CBRAMOD_IFNET_TRAINABLE_TEACHER_ADAPTER_LOSO_V2`。

状态：设计文档。已核对现有 V1 配置、feature tap、loss 和结果；本版未创建训练入口、未执行 smoke、未排队或训练。以下新增设置是固定 pilot 起点，不是已验证的最优参数。

本版首批只执行第 1 批两组 FEATURE；第 2 批 DiffKD 为预先说明的后续设计，不随首批自动启动。

## 1. 问题和结论边界

第一批回答：把现有固定随机线性投影换成可训练的小型 MLP，并让它与 IFNet 联合训练，最终 IFNet 是否改善？

第二批回答：同一个已学习、已冻结的 adapter 下，DiffKD 是否优于直接 FEATURE？

教师侧 adapter 在教师完整特征与学生监督目标之间学习映射。它不插入教师原分类路径，也不改变教师的原三通道输入。最终推理只加载 IFNet。

固定随机线性 P 与可训练 MLP 同时在结构、非线性和可学习性上不同。第一批比较的是两条映射方案，不能把分数差单独归因于“可训练”或“非线性”。若以后需要分离这些因素，可另加一个 `Linear(2400,512,bias=False)`，用旧 P 初始化并仅把权重改为可训练；该控制不在本轮首批预算中。

这是查看过同一数据集既有目标结果后的探索性后续研究。固定本版超参数不代表整个研究过程从未看过目标成绩。

## 2. 固定数据和配对协议

| 项目 | 设置 |
|---|---|
| 数据集、任务 | BNCI2014004 / 004，session_3，左右手二分类 |
| 模型 | CBraMod → IFNet |
| seeds / 目标被试 | 666、667、668；每 seed 9 折 |
| 每折划分 | 7 源训练被试 D_train + 1 源反馈被试 D_feedback + 1 目标被试 D_test |
| split | 复用现有 UID/split manifest，不重新随机划分 |
| 教师 | 现有完整特征 T2S V1 的实际 T_CE、T_MCC；各自已从同折 T0 适配 10 epoch |
| 拟合范围 | adapter、学生以及第二批 prior 的所有梯度只使用 D_train |
| D_feedback | 固定时点只读诊断，不选超参数、epoch、checkpoint 或方法 |
| D_test | 本批全部条件、全部三 seed 的训练终点锁定后统一评测 |

预处理逐项继承被复用运行的 resolved recipe。CBraMod 保持其三通道模型输入与 `apply_EA=false`；IFNet 保持原生通道、双频带和无 EA。两者均不使用目标被试统计进行本版适配。所有 trial 按原 UID 配对。

严格复用的是 `cbramod_t2s_full_feature_loso_v1` 实际用于学生训练的教师及其完整 targets，不重新构建更早的历史教师。新旧教师若 hash 不同，不能拼接为同一个条件。

## 3. 特征与 adapter 的实际定义

```text
冻结 CBraMod backbone 输出 [B,3,4,200]
    → reshape [B,2400]，保留全部通道和时间片元素
    → 可训练 Linear(2400,512)
    → GELU
    → 可训练 Linear(512,512)
    → 无仿射 LayerNorm(512)
    → 教师侧监督向量 z_A

IFNet 原生分类头前特征 h_S [B,512]
    → 无仿射 LayerNorm(512)
    → 学生监督向量 z_S

IFNet 原分类路径仍是 h_S → 原分类头 → logits → CE
```

两个 Linear 均使用 bias。adapter 共 1,491,968 个可训练参数，fp32 参数约 6 MB；没有 BN、Dropout、通道平均、patch 平均或额外学生 projector。隐藏维数取 512，避免再额外加入 256 维瓶颈。LayerNorm 无 affine，eps=1e-5，仅位于辅助分支。

adapter 使用 PyTorch Linear 的默认初始化，并记录实际初始权重 hash。初始化在独立 CPU RNG/fork_rng 中进行，派生种子由 `experiment_id/stage/seed/fold` 确定。同一 seed/fold 下，CE 与 MCC 教师的 adapter 从同一个实际初态开始；初始化不消耗学生初始化、数据顺序或 Dropout 随机流。

教师仍用原 `[B,3,4,200] → reshape2400 → 原 Dropout/Linear` 分类。训练 adapter 不改变教师骨干或分类头，因此本批不会产生新的教师适配成绩。

## 4. 第 1 批：联合训练 adapter + FEATURE

对同 UID 的冻结教师特征 f_T 和学生特征 h_S：

\[
z_A=LN(A_\theta(\operatorname{sg}(f_T))),\quad z_S=LN(h_S),
\]

\[
L_{feat}=\frac{1}{B\cdot512}\sum_i\|z_{S,i}-z_{A,i}\|_2^2,
\qquad L=CE(logits_S,y)+0.5L_{feat}.
\]

教师 f_T 停止梯度；z_A 不 detach，否则 adapter 无法训练。MSE 两侧都保留梯度，分别更新 IFNet 和 adapter。每 batch 只前向 IFNet 一次，同时读取 h_S 和 logits，避免额外 BN 更新或 Dropout 消耗。

首批注册表：

| 条件名 | 教师 / 映射 | 训练状态 | 来源 |
|---|---|---|---|
| IFNet_CE | 无教师 | 仅 IFNet CE | 复用严格匹配的 V1 CE reference |
| FIXED_P_CE_FEATURE | T_CE + 旧固定 P | 仅 IFNet | 复用 V1 CE_T2S_FEATURE |
| FIXED_P_MCC_FEATURE | T_MCC + 旧固定 P | 仅 IFNet | 复用 V1 MCC_T2S_FEATURE |
| JOINT_A_CE_FEATURE | T_CE + 本版 MLP | IFNet 与 adapter 联合更新 | 新训练 |
| JOINT_A_MCC_FEATURE | T_MCC + 本版 MLP | IFNet 与 adapter 联合更新 | 新训练 |

所有新 IFNet 都从与对应 CE reference 完全相同的未训练初态 I_s 开始，包括原分类头和 BN 状态。不能加载已训练 CE 学生作为 warmup。两组使用相同 UID 顺序和学生 Dropout 随机流。

### 固定优化配置

| 项目 | IFNet | adapter |
|---|---|---|
| 训练终点 | epoch100 | epoch100 |
| optimizer | AdamW | 独立 AdamW |
| lr / weight decay | 1e-3 / 0.01 | 1e-3 / 0.01 |
| betas / eps | (0.9,0.999) / 1e-8 | 同左 |
| scheduler | epoch-end cosine，T_max=100、eta_min=0 | 同左 |
| batch | 16 | 同一 batch |
| precision | fp32 | fp32 |
| 梯度裁剪 | 沿用原学生无裁剪 | adapter 参数范数上限 1.0 |
| foreach / fused | False / False | False / False |

教师在本批全程 eval/frozen。学生保留原生 train 模式、BN 和 Dropout；无新增数据增强、logits KD、AID、Keep 或其他辅助 loss。从 epoch1 启用 feature，取固定 epoch100，不按反馈或目标挑 checkpoint。

### 必须解释的局限

联合 MSE 允许 adapter 追随学生来降低 loss，不能保证教师知识被保留或传给学生。LayerNorm 约束单 trial 的尺度，但不能保证跨 trial 不塌缩。本版将这种情况作为需要测量的结果，不临时增加重建头、分类头、参考学生或其他损失。

## 5. 第 1 批记录与预定比较

固定 epoch1/10/50/100，在 D_train 与 D_feedback 记录：

- 学生 BA、Accuracy、CE、预测类别数量；目标只在本批正式训练全部完成后评测。
- 未加权/加权 MSE；adapter 输出与学生辅助特征的方差、有效秩和类间/类内距离。
- 同一固定诊断 batch 的 adapter 目标相对 epoch1 的变化，观察监督目标是否大幅漂移。
- feature 项对 adapter、h_S、IFNet stem 的梯度范数；学生 CE 与 feature 梯度的夹角。
- 原生 h_S 与归一化 z_S 分别诊断，避免只看辅助分支的尺度和几何。

IFNet 的约束线性层即使 eval 也可能 renorm；诊断必须保存/恢复模型状态与 RNG，检查诊断前后 hash 一致。

主要比较（各 9 个配对目标被试、每被试先平均三个 seed）：

1. JOINT_A_CE_FEATURE − FIXED_P_CE_FEATURE。
2. JOINT_A_MCC_FEATURE − FIXED_P_MCC_FEATURE。

次要比较为两组相对 IFNet_CE，以及 JOINT_A_MCC_FEATURE − JOINT_A_CE_FEATURE。每项报告 BA/Accuracy 差 pp、10000 次被试 bootstrap 95% CI、逐被试差及 W/T/L；tie 阈值 1e-6 pp。bootstrap 固定 seed20261103，不把27个seed-fold当27个独立被试。

MSE 降低不作为成功标准。最终 IFNet 改善才支持本具体 adapter pipeline；即使改善，也不单独证明非线性或可学习性的贡献。

## 6. 第 2 批：冻结学到的 adapter 后，成对检验 DiffKD

本批需单独决定执行；不随第 1 批自动启动。采用第 1 批固定 epoch100 的 A*_CE / A*_MCC，不按反馈或目标选择 adapter。

顺序固定为：

1. 冻结教师及对应 A*，只用 D_train 导出 `z_T=LN(A*(f_T))`。
2. 用这个固定 z_T 重新训练对应去噪器，固定5000更新，再冻结。
3. 分别从同一个未训练 I_s 训练 FEATURE 与 DIFFKD 两个全新 IFNet，各100epoch。

正式条件：

| 条件名 | 教师目标 | 最终学生目标 |
|---|---|---|
| FROZEN_A_CE_FEATURE | T_CE + A*_CE | CE + 0.5 MSE(z_S,z_T) |
| FROZEN_A_MCC_FEATURE | T_MCC + A*_MCC | 同左 |
| FROZEN_A_CE_DIFFKD | T_CE + A*_CE | CE + 0.5 MSE(D_phi(z_S,epsilon),z_T) |
| FROZEN_A_MCC_DIFFKD | T_MCC + A*_MCC | 同左 |

第二批 FEATURE 必须重新训练。第一批学生面对变化的 adapter，不能拿来充当固定 A* 的 DiffKD 对照。第二批最终学生不加载第一批学生、optimizer 或 BN 状态。

prior 继续采用现有512维向量方案：输入 `[z_t,time32] → Linear128 → SiLU → Linear128 → SiLU → Linear512`，无标签/通道条件；DDPM1000步、linear beta1e-4→0.02；AdamW lr3e-4、wd1e-4、batch=min(128,N_train)、5000更新、clip1、fp32、无早停/EMA/BN/Dropout。学生去噪固定 DDIM eta0，250→200→150→100→50→0。

旧固定 P 的 prior 不能复用，因为它学习的是另一种目标特征分布。最终学生阶段 A* 与 prior 均固定，五步去噪对学生输入保留 autograd。两个新教师的 prior 共用初态、UID抽样和噪声随机流；两个 DiffKD 学生共用独立的噪声表。

第二批主要比较同教师的 FROZEN_A_DIFFKD − FROZEN_A_FEATURE；其次相对 CE 学生。教师选择、超参数和 epoch 不根据第一批目标成绩修改。本批是查看过第一批结果后开展的探索性两阶段 pipeline，不能声称整体未见过目标结果。

A* 含第 1 批学生联合训练轨迹的信息，因此第二批不属于“没有学生预阶段的纯教师压缩”。全部前阶段成本计入报告。

## 7. 更新与冻结关系

| 阶段 | 更新 | 冻结 |
|---|---|---|
| 第1批 JOINT_A_FEATURE | 新 IFNet + 对应 adapter | 已有 T_CE/T_MCC |
| 第2批 target 导出 | 无 | 教师 + A* |
| 第2批 prior 拟合 | 对应去噪器 | 教师 + A* + 全部学生 |
| 第2批最终 FEATURE / DIFFKD | 全新 IFNet | 教师 + A* + prior |
| 推理 | 无 | 只调用最终 IFNet 原推理路径 |

## 8. 复用、验收与预算

正式实现前逐折核验：实际教师权重/targets hash、完整 feature `[N,2400]`、UID/标签、7/1/1、source/preprocessing manifest、CE baseline 的 I_s/BN/UID顺序/Dropout/optimizer/scheduler。复用当前 T2S V1 的实际教师，缺失产物列清单并停止，不触发隐式重训。

V1 固定 P 带有确定性恢复记录时，结合 `restored_original_sha256`、实际矩阵和恢复核验记录判断身份；不能仅因 NPZ 归档字节 hash 变化就判定原实验缺失或触发重训。

只缓存固定教师原始2400维特征。第1批 adapter 输出随训练改变，不能把初始化时导出的 adapter targets 当作全程训练目标。

固定单折、train-only smoke 验收：

- 一个学生 forward 正确取 h_S/logits；MLP输入2400、输出512。
- 仅 feature backward 时 adapter 与 IFNet stem 均有有限非零梯度；教师没有梯度且 hash 不变。
- 第1批 z_A 不 detach；第2批 A* 无梯度，prior无参数梯度但学生输入有梯度。
- feature_weight=0 时跳过辅助分支，不消耗额外学生 RNG，退化为同一 CE 轨迹。
- checkpoint/resume 恢复 S/A/prior 的 optimizer/scheduler、独立 RNG、sampler cursor，下一步一致。
- 正式 prior 从规定初态训练5000更新；smoke 权重不进入正式结果。
- 最终只加载 IFNet，复现保存的目标 logits。

| 工作 | 第1批 | 第2批额外新增 |
|---|---:|---:|
| 新 IFNet 100epoch 轨迹 | 54（2条件×3seed×9折） | 108（4条件×3seed×9折） |
| 与学生联合训练的 MLP | 54 | 0；读取第1批 A* |
| 新 prior / 每个更新数 | 0 | 54 / 5000 |
| 新教师微调/适配 | 0，合法复用已有产物 | 0 |

若两批均完成，总计162条新学生100epoch轨迹、54个prior；不能声称与旧固定P版本总训练预算相同。第1批固定P/CE旧参考只有严格身份匹配才复用。执行前再根据同机实际历史耗时估算GPU时间，不在设计阶段编造时间。

每GPU worker数值库和Torch intra/inter-op线程均1，DataLoader workers0，进入共享队列，不改变其他任务资源设置。

## 9. 独立入口和保存材料

拟新增（本次未创建）：

- `configs/experiments/loso/cbramod_ifnet_trainable_adapter_v2.yaml`
- `experiments/distill/run_cbramod_ifnet_trainable_adapter_loso.py`
- `collab/cbramod_ifnet_trainable_adapter.py`

采用新experiment_id和外部run目录，保留V1源码、结果与产物。通过experiments.storage解析新cache/weights/results；物理特征、adapter、教师引用、模型权重与resume均在 `/data1/llx/BigSmallcollab/` 下。报告在checkout真实 `results/distill/cbramod_ifnet_trainable_adapter_v2/`。

保存resolved config、源码/环境/commit/status、输入/split/UID hash、教师与I_s来源、adapter初态/终态与hash、两种optimizer状态和独立RNG、阶段history/梯度诊断、逐trial logits与小完成记录。第二批额外保存A*目标缓存与prior。推理仅需IFNet，不据此删除研究仍需复用的adapter或教师。

Excel分别报告Accuracy/BA，明示三seed与九被试，按项目student/method及匹配teacher/CE规则着色；首批JOINT与第二批FROZEN条件保持独立名称。单批完成性不足则标缺失，不填0或不完整宏平均。本次不自动登记exp_log。

结论只评价实际完成的配置：固定P vs联合MLP、固定已学习adapter下FEATURE vsDIFFKD，分别报告；不把MSE下降、维数一致、教师自身BA或单seed波动当作方法有效的证据。

# CBraMod → IFNet：双侧可训练 adapter 的 LOSO 实验方案

日期：2026-11-04。版本：`CBRAMOD_IFNET_DUAL_ADAPTER_FEATURE_LOSO_V3`。

状态：设计文档。已读取当前 V1/V2 特征映射、V2 runner/config 及其已完成结果；尚未实现 V3、执行 smoke、提交队列或训练。下述新增设置是预先固定的实验选择，未验证最优性。本次只制定 FEATURE 方案。

## 1. 本轮检验的问题

在冻结同一个 CE 适配 CBraMod 教师的前提下，两侧各加一个可训练 adapter，映射到共同的 512 维空间，是否比只训练教师侧 adapter 更有利于最终 IFNet？

共同空间中的两个表示都可以移动，不能预先称其中一个为固定的“教师监督目标”。本版同时测量对齐程度、表示是否保留跨 trial 差异，以及最终 IFNet 的表现。

本轮教师统一使用现有 **T_CE**：同折 T0 经纯 CE 继续适配 10 epoch 的 CBraMod。MCC 直接作用于分类概率形成的类别混淆矩阵；其梯度可以间接改变骨干特征，但它不直接完成教师—学生特征匹配。本轮不使用 T_MCC。

此前已经看过本数据集的多轮目标结果，因此这是探索性后续实验。锁定本版参数不等于整个研究从未看过目标成绩。

## 2. 已完成参考及本轮条件

从 `results/distill/cbramod_ifnet_trainable_adapter_v2/metric_summary.csv` 核对到以下参考；均为 004/session_3、3 seeds、每 seed 9 个目标被试、7/1/1 协议。这里 Accuracy 与 BA 的实际汇总数值相同，仍分别标明指标。

| 本版报告名称 | 实际来源名称 | Accuracy (%) | BA (%) |
|---|---|---:|---:|
| IFNet_CE | V2 汇总的 IFNet_CE | 74.10 | 74.10 |
| FIXED_P_FEATURE | V1 CE_T2S_FEATURE | 73.90 | 73.90 |
| TEACHER_ADAPTER_FEATURE | V2 JOINT_A_CE_FEATURE | 73.36 | 73.36 |

这些参考只解释当前设计的起点，不作为新结果。此前其他标准 LOSO 或其他划分下的 CE 数字不能替换这里的配对基线。

正式条件如下；新增两组，其余严格核验后复用：

| 条件 | 教师侧 | 学生侧辅助分支 | 新增训练 |
|---|---|---|---|
| IFNet_CE | 无 | 无 | 复用 CE |
| FIXED_P_FEATURE | V1 固定 P + LN | LN(h_S) | 复用 V1 |
| TEACHER_ADAPTER_FEATURE | V2 可训练 A_T + LN | LN(h_S) | 复用 V2 |
| FIXED_P_STUDENT_ADAPTER_FEATURE | 同一个 V1 固定 P + LN | 新可训练 A_S + LN | 是，稳定教师目标对照 |
| DUAL_ADAPTER_FEATURE | 与 V2 同初态的可训练 A_T + LN | 同一个新可训练 A_S + LN | 是，主条件 |

主要比较是 `DUAL_ADAPTER_FEATURE − TEACHER_ADAPTER_FEATURE`，检验在现有教师侧 adapter 上增加学生侧 adapter 的整体作用。

`FIXED_P_STUDENT_ADAPTER_FEATURE − FIXED_P_FEATURE` 检验固定教师目标时增加学生侧 adapter 的作用。`DUAL_ADAPTER_FEATURE − FIXED_P_STUDENT_ADAPTER_FEATURE` 同时改变了教师映射的结构、非线性、初始化和可学习性，不能把这个差值单独归因于“教师 adapter 可训练”。

## 3. 数据、教师及划分

| 项目 | 固定设置 |
|---|---|
| 数据 | BNCI2014004 / 004，session_3，左右手二分类 |
| 原始源版本 | `/data1/llx/BNCI2014004/wide_fre_band/` |
| 教师 / 学生 | CBraMod / IFNet |
| seeds / folds | 666、667、668；每 seed 9 折 |
| 每折 | 7 源训练被试 D_train + 1 源反馈被试 D_feedback + 1 目标被试 D_test |
| split 与配对 | 复用 V1/V2 的实际 UID/split manifest；师生依原始 trial UID 对应 |
| 教师 | V1 实际用于 V2 的 T_CE 及完整特征 targets；骨干与原分类头全冻结 |
| 输入处理 | 继承各自历史 resolved recipe；CBraMod 三通道、apply_EA=false；IFNet 原生通道、双频带、无 EA |
| 梯度与任何拟合 | 仅 D_train |
| D_feedback | 预定时点只读诊断；不选择参数、epoch、checkpoint 或条件 |
| D_test | 全部新条件、全部三 seed 终点锁定后统一评测 |

教师完整特征为原骨干 `[B,3,4,200]`，按原有元素顺序 reshape 为 `[B,2400]`。保留全部通道和 patch 元素。IFNet h_S 为原分类头实际读取的 `[B,512]` 向量。

预处理参数以旧运行 resolved config 和输入缓存 manifest 为准，不用当前 YAML 默认值覆盖。教师、学生均不使用目标被试的跨 trial 统计完成本版训练或预处理；辅助 LayerNorm 按单 trial 计算。

## 4. 两个 adapter

```text
冻结 CBraMod 的 f_T [B,3,4,200]
  → reshape [B,2400]
  → A_T: Linear(2400,512) → GELU → Linear(512,512)
  → 无仿射 LayerNorm(512)
  → z_T [B,512]

IFNet 原生 h_S [B,512]
  → A_S: Linear(512,512) → GELU → Linear(512,512)
  → 无仿射 LayerNorm(512)
  → z_S [B,512]

IFNet 原分类分支：h_S → 原分类头 → logits → CE
```

两个 adapter 的 Linear 均有 bias；LayerNorm eps=1e-5，无 affine。adapter 无 BN、Dropout、额外分类头或 residual 分支。共同输出维度固定 512。

| 模块 | 参数量 |
|---|---:|
| 教师侧 A_T | 1,491,968，沿用 V2 结构 |
| 学生侧 A_S | 525,312 |
| 双 adapter 合计 | 2,017,280 |

这两百万参数属于训练辅助模块，额外参数和计算成本须报告，不能称与原 IFNet 等训练成本。最终推理使用 IFNet 原路径，两个 adapter 均不参与。

### 初始化及配对

- IFNet 读取与 CE/V1/V2 对应的同一个未训练 I_s，包括分类头、BN 初值；不加载任何训练后的学生状态。
- A_T 使用 V2 的实际初态。当前 V2 保存了 `adapter_seed` 与初态 state hash，未承诺独立初态权重文件。若需重建，使用 **V2 experiment_id** 和原 `derive_adapter_seed`、原环境及 Linear 默认初始化，逐张量核对 `adapter_initial_sha256`/`initial_state_sha256`。验证后将实际初态保存到 V3 外部权重目录。不能用新 V3 experiment_id 派生的另一初态替代。
- A_S 使用 PyTorch Linear 默认初始化，在独立 CPU RNG/fork_rng 中生成。固定派生规则为 `int.from_bytes(SHA256(f"{experiment_id}/student-adapter-init/{seed}/{fold}").digest()[:8], "big") % 2**32`，fold 使用 0–8。两条新条件读取同一个实际 A_S 初态及 hash。
- 新增 adapter 初始化不消耗 IFNet 初始化、训练 UID 顺序或 Dropout RNG；两个新条件分别恢复同一学生随机流。训练使用原样保存的 100 epoch UID schedule，不改变最后一个 batch 的处理方式。
- 新条件的学生会因 loss 不同而沿不同参数轨迹训练；配对约束指相同的初态、数据访问和随机数序列。

## 5. 损失和更新范围

对同 UID 的师生 trial，双侧条件定义：

\[
z_T=LN(A_T(\operatorname{sg}(f_T))),\qquad z_S=LN(A_S(h_S)),
\]

\[
L_{align}=\frac{1}{512B}\sum_i\|z_{S,i}-z_{T,i}\|_2^2,
\qquad L=CE(logits_S,y)+0.5L_{align}.
\]

教师 f_T 停止梯度；**z_T、z_S、h_S 均不 detach**。MSE 同时更新 A_T、A_S 和 IFNet 骨干；CE 更新 IFNet 骨干与原分类头。若 detach z_T，A_T 就无法由该损失训练，不属于本版双侧联合训练。

固定教师目标对照仅替换为：

\[
z_T^{fixed}=\operatorname{sg}(LN(Pf_T)),\qquad z_S=LN(A_S(h_S)).
\]

P 必须是同 seed/fold 的 V1 实际固定矩阵，核对保存的实际矩阵、恢复记录及原身份，不重新抽取随机 P。

每 batch 只调用一次 IFNet 前向，同时取 h_S 和 logits。清零本条件所有优化器梯度后，一次 backward；分别裁剪 adapter 梯度，再按 IFNet、A_T（若存在）、A_S 的固定顺序 step。

首批使用上述 CE+MSE；不同时加入防塌缩正则、logits KD、MCC 或 DiffKD。这样可以检验双侧映射本身。若结果暴露退化，完整报告，并在后续独立版本设计约束，不在本版轨迹中途补 loss。

## 6. 固定优化配置

| 参数 | IFNet | A_T、A_S（各自独立优化器） |
|---|---|---|
| 终点 | 固定 epoch100 | 固定 epoch100 |
| optimizer | AdamW | AdamW |
| lr / weight decay | .001 / .01 | .001 / .01 |
| betas / eps | (.9,.999) / 1e-8 | 同左 |
| scheduler | epoch-end cosine，T_max=100，eta_min=0 | 同左 |
| batch / precision | 16 / fp32 | 同 batch / fp32 |
| 梯度裁剪 | 继承 V2：无裁剪 | 每个 adapter 独立 clip norm=1.0 |
| foreach / fused | False / False | False / False |
| 模式 | train，保留原 BN/Dropout | train；本身无 BN/Dropout |

从 epoch1 使用 feature_weight=.5；CE smoothing=0，无类别权重、无新增增强，采用固定 epoch100。所有条件的反馈与目标指标均不参与学习率、loss 权重、epoch 或模型选择。

## 7. 必须测量的退化及信息保留

双侧 MSE 存在常量解：两个 adapter 对所有 trial 输出同一个向量 c，经过相同 LN 后仍相等，MSE 即为零。学生 CE 位于原 h_S 分支，不阻止两个辅助分支同时变成常量。LN 约束单 trial 的尺度，也不能保证跨 trial 差异。这是该目标本身的数学可行解，不代表实际训练必然发生。

在 epoch0/1/10/50/100 的 D_train 全部 trial 与 D_feedback 分别记录：

1. CE、原分类头 BA/Accuracy、MSE 及其加权值；z_T、z_S、原生 h_S 的跨 trial 平均维度方差、中心化谱、有效秩、类间/类内距离。
2. `CKA(z_T,z_S)`，以及 `CKA(z_T,f_T)`、`CKA(h_S,f_T)`。前者测量匹配程度，后两者检查教师 adapter 与部署学生是否仍保留原教师相关的信息结构。CKA 允许输入维数不同；不把高 CKA 直接解释成预测能力。任一输入中心化后总方差为零时，CKA 记 `NA/undefined` 并记录退化，不填写0或1。
3. 对固定 UID 子集，记录两侧输出相对 epoch0 的 RMSE 漂移、不同 trial 的余弦相似度；零范数情形单独计数，不伪造余弦为 1。
4. 固定训练诊断 batch 上，feature 项对 A_T、A_S、h_S、IFNet stem 的梯度范数，以及 CE 与 feature 在 stem 上的梯度范数比和夹角。记录 adapter 是否主要自行吸收匹配误差。
5. 原生 h_S 与辅助 z_S 分开保存统计，避免只看到学生 adapter 输出改善。

有效秩在全部分区 trial 上计算，不能从 batch16 的秩上限直接判断512维表示塌缩。中心化协方差正特征值归一化为 p 后，`effective_rank=exp(-sum(p*log(p)))`；总方差为零时 rank 记0。跨 trial 方差定义为 `mean_j Var_i(z_ij)`，采用固定的样本方差约定。

预先标注 `mean_j Var_i(z_ij)<1e-4` 为“近常量输出”预警；effective_rank≤2 仅标“低秩”，二分类可存在有效低秩表示，不能单凭该值宣称塌缩。连续量仍完整报告。这些是诊断标记，不触发选 checkpoint、换参数、中途加正则或删除折。

正式训练中有限的零梯度可以是退化证据，应记录；不能复用“每一步 adapter 梯度必须大于零”的硬失败规则来掩盖此结果。训练张量/loss/梯度出现 NaN/Inf 或图断开时按实现/数值失败处理；塌缩导致诊断 CKA 无定义不属于该类硬失败。接线 smoke 的独立初始 batch 应验证 feature 梯度能到达两侧及 IFNet stem。

诊断保存并恢复模型状态、模式和 RNG，核对 hash；IFNet 的约束分类层可能在 eval 前向 renorm，必须沿用 V2 防护。缺失的历史 epoch0/中间特征不补造，也不为画完整曲线隐式重训；旧条件可用已保存诊断，最终特征可在保留的权重上只读提取。

## 8. 执行前验收与 smoke

正式实现时先生成覆盖清单：27 折 T_CE/完整 targets、V1 P/恢复记录、V2 教师 adapter 初态身份、I_s/BN、UID schedule、旧 CE/V1/V2 的逐 trial 预测及完成记录。逐折验证 hash、7/1/1、UID 无交叉、类别编码、预处理身份和唯一折。缺失则列清单；缺权重不触发隐式重训，已完成结果 reader 使用保留的预测/指标/provenance。

固定 seed666/fold0 的 train-only smoke 使用与正式训练相同的 loss dispatch，包含：

- f_T=2400、h_S=512、两个输出均512；仅 feature backward 时两 adapter、h_S、stem 有有限非零梯度；教师无梯度、状态 hash 不变。
- 一次学生前向产生分类与特征；辅助参数没有被意外接入原分类路径。
- 对固定目标对照和双侧条件分别验证 optimizer 参数集合、独立 clip、step 与 scheduler。
- feature_weight=0 时跳过辅助分支，不消费额外学生 RNG；恢复相同初态后与 CE 短轨迹及 BN/Dropout 状态一致。
- 保存/恢复 IFNet、A_T、A_S、各 optimizer/scheduler、随机流、UID 游标后，下一步可复现；不存在的 A_T 状态在固定P对照中显式省略。
- 只读诊断前后 state/RNG hash 一致；最终 IFNet 脱离 adapter 单独加载，重现原分类 logits。

smoke 不评目标；其权重不进入正式结果。通过后执行27个配对折任务，每任务训练两条新学生轨迹。全部新轨迹终点固定后统一进行目标预测；预警折也纳入，不按成绩剔除。

## 9. 最终评测与预定比较

主指标为目标被试 BA，另报 Accuracy、Kappa、CE、每类 recall、预测数量、collapse。目标阶段保存 UID、y、IFNet logits/probabilities 和对应 T_CE logits；同时在固定终点做只读特征诊断。

每条件须完整覆盖 seeds666/667/668 × 9 个唯一目标被试。成绩先在每 seed 内对9被试等权平均，再对三seed平均；如展示 mean±SD，明确这是三seed宏平均的样本 SD。被试间 SD 另列，不能混称 seed SD。

配对推断先在每目标被试内平均三seed，再在9个被试上计算差值、W/T/L 与10000次被试配对 bootstrap 95% CI；bootstrap seed=20261104，所有比较使用同一索引表，tie阈值1e-6 pp。LOSO训练集重叠、只有9个被试，这些区间作为探索性描述；不把27个seed-fold或trial当独立被试。本轮不报告p值。

预定比较：

| 层级 | 比较 | 回答的问题 |
|---|---|---|
| 主要 | DUAL_ADAPTER_FEATURE − TEACHER_ADAPTER_FEATURE | 当前teacher-only pipeline增加学生adapter后的整体变化 |
| 次要 | DUAL_ADAPTER_FEATURE − IFNet_CE | 最终部署学生是否改善 |
| 次要 | FIXED_P_STUDENT_ADAPTER_FEATURE − FIXED_P_FEATURE | 固定教师目标下学生adapter的作用 |
| 次要 | DUAL_ADAPTER_FEATURE − FIXED_P_STUDENT_ADAPTER_FEATURE | 两种教师映射方案在同一学生adapter下的整体差别 |

只有辅助MSE下降而IFNet不升，记录为“共同空间更接近，部署学生未改善”。若出现近常量输出，明确记录该配置的退化。即使双侧条件改善，也只能支持这套具体pipeline，不能单独归因于语义对齐、参数量或双adapter的普遍优势。若需要区分教师信息与额外分支带来的正则效应，后续另设教师目标打乱等机制对照。

## 10. 工作量、资源及后续边界

| 首批新增工作 | 数量 |
|---|---:|
| IFNet 100epoch轨迹 | 54：两条件×三seed×九折 |
| A_T 联合训练轨迹 | 27：仅双侧条件 |
| A_S 联合训练轨迹 | 54 |
| 教师微调/适配 | 0，复用合法T_CE |
| prior/去噪器 | 0 |

旧三类参考合计81条条件折记录只读复用；加上本批54条，学生报告共135条唯一条件折记录，教师单独列。若旧参考不满足身份匹配，标注不可配对，不默默补跑。

新架构尚未运行，不给出虚构GPU时间。执行时用同一共享3090上的smoke/单折实际训练步骤计时，分别记录训练、诊断与导出成本，据此估算54条轨迹的worker-hours；共享队列等待时间单列。

每GPU worker的OMP/MKL/OpenBLAS/NumExpr/BLIS及Torch intra/inter-op线程为1，DataLoader workers=0；进入已有共享队列并遵守原GPU/主机限制，不调整其他任务。

DiffKD不并入本批。若后续检验DiffKD，须先固定该阶段教师adapter输出空间，并设计同一冻结目标下的FEATURE与DiffKD配对；本批变化的z_T不能直接拿旧固定P prior去噪，联合训练学生也不能充当冻结目标的FEATURE对照。后续形成单独方案和预算。

## 11. 实现与产物接口（拟定）

本次仅创建本方案；拟新增独立实现：

- `configs/experiments/loso/cbramod_ifnet_dual_adapter_v3.yaml`
- `collab/cbramod_ifnet_dual_adapter.py`
- `experiments/distill/run_cbramod_ifnet_dual_adapter_loso.py`

保留V1/V2源码、条件名和产物身份；新代码复用只读工具时记录依赖源码hash。原始2400维teacher targets继续只读复用；训练期间不缓存可变adapter输出当作固定targets。

通过 `experiments.storage` 解析 `/data1/llx/BigSmallcollab/{cache,weights,results}/` 下新experiment/run目录。物理特征、初始化、adapter、IFNet和临时resume均存外部；旧官方权重继续使用canonical副本。用户报告经 `REPORTS_ROOT` / `require_report_output` 输出到checkout真实目录：`results/distill/cbramod_ifnet_dual_adapter_v3/`。

保存resolved config、代码/环境/commit/status、split/UID/输入hash、T_CE/P/初态引用与hash、两个adapter初态和终态、I_s、阶段history、梯度/退化诊断、逐trial预测与小完成记录。resume保存所有优化器、scheduler和独立RNG，验证完成后清理临时resume；已训练adapter和最终学生在后续分析/复用仍依赖时保留，不凭“推理不使用”删除。

报告含CSV/JSON、Excel与短Markdown：教师行在先，student/method开头，IFNet none基线后依次列四种FEATURE条件；Accuracy/BA分表，标明3seed/9被试/7+1+1，使用项目约定的匹配teacher/CE着色。缺seed/折保留缺失标记，不计算不完整宏平均。该方案不自动更新`docs/exp_log.csv`。

## 12. 核对来源和设计依据

- 当前V2源码：`collab/cbramod_ifnet_trainable_adapter.py`、`experiments/distill/run_cbramod_ifnet_trainable_adapter_loso.py`、对应YAML；V1固定P：`collab/cbramod_t2s_projection.py`。
- 当前V2结果：`results/distill/cbramod_ifnet_trainable_adapter_v2/{metric_summary,paired_differences,fold_metrics}.csv`；仅将实际完成数值列为参考。
- [SimSiam，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)研究了双分支表示学习中的塌缩与停止梯度；[VICReg](https://arxiv.org/abs/2105.04906)显式使用方差约束处理塌缩。这些是诊断背景，不是本版损失或EEG有效性的依据。本版没有直接套用其stop-gradient、variance/covariance损失，也不声称复现这些方法。

本版结论等待真实训练与完整目标预测。本次文档本身不代表已实现、已排队或已取得新结果。

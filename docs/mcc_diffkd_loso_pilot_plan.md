# MCC 适配 MIRepNet → Diffusion 特征蒸馏 IFNet：LOSO pilot 方案

日期：2026-10-30。版本：`MCC_TEACHER_ADAPT_DIFFKD_LOSO_V1`。

状态：设计文档。已只读核对当前模型接口、历史配置、split/cache manifest，以及 MCC、DiffKD 原文与官方实现；尚未实现本版 runner、运行 smoke、排队或训练。本文件中的新增超参数是预先指定的 pilot 起点，不是已验证的最优值。

## 1. 本轮要检验什么

流程固定为：

**同折原 MIRepNet T0 → CE 或 CE＋MCC 适配 → 冻结教师 → 教师特征训练去噪器 → 新 IFNet 做配对特征蒸馏 → 仅 IFNet 预测。**

分别检验两个因素：

1. 同样额外训练十个 epoch，MCC 是否使教师更有利于最终 IFNet？
2. 同一个教师、相同特征坐标和辅助投影，Diffusion 去噪是否优于直接特征匹配？

最终评价对象是 IFNet 原骨干与原分类头；教师准确率、MCC 数值和去噪误差是解释性指标。本版第一阶段不使用参考 IFNet S0，因此应称“教师自身的 MCC 正则适配”，不再声称这一阶段是“小模型指导大模型”。

这轮是查看过同一数据集前轮结果之后提出的探索性后续实验。新版本内部锁定参数、checkpoint 和评测时间，仍不能把整个研究过程描述为从未看过该数据集的目标结果。

## 2. 数据与固定协议

| 项目 | 固定设置 |
|---|---|
| 数据集 | BNCI2014004，session_3，左右手二分类 |
| 数据源 | `/data1/llx/BNCI2014004/wide_fre_band/` |
| 教师 / 学生 | MIRepNet / IFNet |
| 首轮 | seed 666，全部 9 个目标被试 |
| 每折 | 7 源训练被试 D_train＋1 源反馈被试 D_feedback＋1 目标被试 D_test |
| split | 原样继承同 seed 已存在的 UID/split manifest，不重新抽分 |
| 主指标 | 每目标被试 Balanced Accuracy，九被试等权平均 |
| 次指标 | Accuracy、Kappa、CE、每类 recall、预测数量、collapse |

复用依据是历史 `sample_utility_adaptive_weight_wideband_v1` 的 seed-666 resolved config，标识 hash：

`6dfe184f7f6a5ddd8b75e57e6d67c77ce4e68f1bdf4aa349d558c0240539342f`。

现存九折训练 trial 数：目标被试 1、2 的 D_train 为 1120，其余为 1080；目标 2 的 D_test 为 120，其余为 160。实施时以 manifest 的完整 UID 集、类别编码和 hash 再校验，不能只检查数量。现存 manifest 的 `train_subject_ids` 是零起始，`target_subject_id`/`feedback_subject_id` 是一起始，报告统一转换为被试 1–9。

所有梯度、MCC、PCA、特征统计和 Diffusion 拟合只使用 D_train。D_feedback 仅在预定时点做只读诊断，不选参数、epoch、教师或去噪器版本。所有条件的终点固定后统一评测 D_test。

原 MCC 的典型域适应设置将 MCC 用在无标签目标域；这里把它作为源训练集上的正则，是明确的 EEG/source-only 变体。即使不用目标标签，把目标 trial 输入 MCC 并更新模型也会改变本协议，不在本版范围内。[MCC 原文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)

## 3. 预处理与真实 feature tap

模型输入继承合法历史 recipe，不能用今天的 YAML 默认值覆盖历史 resolved 配置。

| 模型 | 本次已核对的输入/特征定义 |
|---|---|
| MIRepNet | 现有模型专用 8–30 Hz、per-subject EA、45 通道映射；输入缓存 `[N,45,1000]` |
| IFNet | 基础输入 `[N,3,1000]`；adapter 的 4–16 / 16–40 Hz 双频带后为 `[N,6,1000]` |
| 教师 h_T | Transformer token 轴 mean-pool 后、原 `clshead` 前的 256 维 trial 向量 |
| 学生 h_S | 原 IFNet 分类头实际读取的 512 维 trial 向量；保留原生分类路径 |

`wide_fre_band` 是源数据版本，不能由此把 MIRepNet 的实际输入写成 0–120 Hz。这里的频段是继承既有协议，不是新提的频段最优结论。

历史教师 EA 使用目标被试全部无标签 trial 的统计；各组原样继承并单列 provenance。因此准确表述是“目标数据不参与优化和先验拟合”，不是“预处理从未使用目标输入”。不得借 EA 的既有使用范围，额外把目标特征用于 MCC、PCA 或 Diffusion。

同一 trial 可有模型专用输入张量，但教师特征目标与学生输入必须按同一个原始 UID 对应。不得依靠数组长度或顺序猜测配对。

## 4. 正式四组与可复用参考

两种新教师从完全相同的 T0 独立复制：

- `T_CE`：CE 继续训练十个 epoch。
- `T_MCC`：CE＋MCC 继续训练十个 epoch。

| 条件名 | 第一阶段教师 | 最终 IFNet 损失 |
|---|---|---|
| `CE_FEATURE` | T_CE | CE＋普通配对特征 MSE |
| `MCC_FEATURE` | T_MCC | CE＋普通配对特征 MSE |
| `CE_DIFFKD` | T_CE | CE＋去噪后的配对特征 MSE |
| `MCC_DIFFKD` | T_MCC | CE＋去噪后的配对特征 MSE |

四组不加教师 logits KD、不加 AID、不加原教师保持项。真实标签 CE 始终存在。最终分类也始终由 IFNet 原分类头输出，辅助模块不代替其分类头。

额外参考为合法的 IFNet CE 基线和 T0 普通 logits KD。旧结果只能在 split、输入、I_s、BN 初值、训练 UID 顺序、dropout 随机流、优化器、100 epoch 终点等全部匹配时复用；不满足时标为不可严格配对，不能混入正式差值。需要可比参考时单独重跑，不能默默换用不匹配的旧成绩。

特别注意：上一轮 V1 的 `C` 是 **CE＋Keep**，不属于这里的纯 CE 继续训练教师，不能复用来冒充 T_CE。两种新教师均需从 T0 按本版目标适配。旧 F/G 原始编码曾发生先验反接，本版使用上述描述性条件名；旧原始编码不参与新条件选择。

## 5. 第一阶段：MCC 教师适配的精确定义

T0 是本折七源训练被试微调后的合法 MIRepNet。两个教师的原骨干和原分类头全部更新，且共享训练 UID batch 顺序。

对教师输出 logits z，类别数 K=2：

\[
P=\operatorname{softmax}(z/T_m),\quad
H_i=-\sum_k P_{ik}\log(P_{ik}+10^{-5}),
\]

\[
w_i=\frac{B[1+\exp(-\operatorname{sg}(H_i))]}{
\sum_j[1+\exp(-\operatorname{sg}(H_j))]},\quad
C=P^\top\operatorname{diag}(w)P.
\]

\[
\widetilde C_{jk}=\frac{C_{jk}}{\max(\sum_l C_{jl},10^{-8})},\qquad
L_{MCC}=\frac{1}{K}\sum_{j\ne k}\widetilde C_{jk}.
\]

只停止 entropy 权重的梯度；P 和教师参数保留梯度。按论文定义显式行归一化，在所有类别行分母≥1e-8的有限输入上验证标量与作者实现一致；低于阈值时单独检查数值边界并记录 clamp 触发次数，不把防下溢导致的低损失解释为改善。这里 MCC 是 2×2 类别混淆矩阵，不是样本关系矩阵。[作者 MCC 实现](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/tllib/self_training/mcc.py)、[entropy 实现](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/tllib/modules/entropy.py)

\[
L_{T\_CE}=CE(z,y),\qquad
L_{T\_MCC}=CE(z,y)+1.0L_{MCC}(z;T_m=2).
\]

| 教师适配参数 | 固定值 |
|---|---|
| epoch / checkpoint | 10 / 最终 epoch10 |
| optimizer | 新 Adam，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=1e-6 |
| batch / scheduler | 8 / 常数学习率 |
| model mode | eval 且允许 autograd：Dropout 关闭，BN running stats 固定 |
| 精度 / 梯度裁剪 | fp32 / 全部教师可训练参数范数上限 1.0 |
| sampler | 原始 trial 均匀打乱，无类别重采样；CE/MCC 完全相同 |
| MCC 温度 / 权重 | 2 / 1.0；无额外温度平方缩放 |

这些新增参数是固定 pilot 选择。低 MCC 不等于预测正确，尤其二分类中高置信错误也可能有低混淆；不按 MCC 最小值挑教师。

## 6. 两类教师共用的固定特征坐标

仅用冻结 T0 在本折 D_train 上的 256 维特征拟合 float64 PCA 白化：

\[
r_{T_j}(x)=W_0\big(h_{T_j}(x)-\bar h_{T0}\big),\quad j\in\{CE,MCC\}.
\]

数值秩阈值为特征值大于最大特征值的 1e-5；维度 `d=min(32,256,数值秩)`，白化分母为 `sqrt(eigenvalue+1e-4*eigenvalue_max)`。有效秩为零则审计失败，不虚构维度。

T_CE、T_MCC 都使用这个 T0 定义的固定坐标，记录各自投影后的均值、方差和保留能量。这样教师因素比较不会同时更换 PCA 坐标。仅建模这 d 维保留子空间，不声称传递了完整 256 维知识。

缓存两类教师 D_train 的 `UID、y、h_T、r_T、logits`。D_feedback 可为诊断单独导出，但不能合并进拟合数据。教师和 PCA 在最终学生训练期间不变。

## 7. 第二阶段的 Diffusion 拟合

分别用 T_CE 和 T_MCC 的 D_train 特征训练一个去噪器；每折两个。**数据来自对应适配教师，不来自旧 IFNet。**

本版借鉴 DiffKD 的“教师特征训练去噪器、去噪学生特征后做配对匹配”思路。采用预训练后冻结的向量去噪器、固定 PCA 和固定学生投影，是本 EEG pilot 的具体变体，不声称完整复现原 DiffKD 的联合训练、autoencoder 或 adaptive noise matching。[DiffKD 原文](https://papers.neurips.cc/paper_files/paper/2023/hash/cdddf13f06182063c4dbde8cbd5a5c21-Abstract-Conference.html)、[作者实现](https://raw.githubusercontent.com/hunto/image_classification_sota/main/lib/models/losses/diffkd/diffkd.py)

令 beta_t 从 1e-4 线性增加到 0.02，t=1…1000；`a_t=sqrt(prod_{s<=t}(1-beta_s))`，`s_t=sqrt(1-a_t^2)`，并约定 `a_0=1,s_0=0`。

\[
r_t=a_t r_T+s_t\epsilon,\qquad
L_{noise}=\mathbb E\frac1d\|\epsilon_\phi(r_t,t)-\epsilon\|^2.
\]

| 去噪器参数 | 固定值 |
|---|---|
| 输入 | r_t 与 32 维固定 sinusoidal t 编码；不输入标签、Teacher logits 或 Student 特征 |
| 网络 | Linear(d+32,128)→SiLU→Linear(128,128)→SiLU→Linear(128,d) |
| BN / Dropout / EMA / CFG | 无 |
| 特征抽样 | D_train trial 均匀有放回，batch=min(128,N_train) |
| t / epsilon | 每样本独立均匀 t∈1…1000 / 标准高斯 |
| optimizer | AdamW，lr=3e-4，betas=(0.9,0.999)，eps=1e-8，wd=1e-4 |
| 更新 / 终点 | 5000 次 / step5000，无反馈早停 |
| 精度 / 梯度裁剪 | fp32 / 1.0 |

不输入真实类别是为了让去噪器依靠输入特征完成恢复；真实类别只用于学生 CE、分组诊断。两类教师的去噪器共享网络初值、训练 UID 抽样序列、t/epsilon 随机流；输入教师特征不同。

每 500 step 在预生成的固定噪声表上记录 train/feedback 分噪声段 MSE（1–250、251–500、501–750、751–1000）及零噪声预测器基线，不据此回选 step。拟合成本单独计入预算。

## 8. 固定学生投影与可微的五步去噪

四组 IFNet 从同一个未训练初始状态 I_s 开始，包含相同原分类头和 BN 初值。每个 batch 只做一次学生前向，同时取得 h_S 与 logits；其原分类路径始终是 `h_S(512) → 原 fc → CE`。不得为 CE 和特征监督各做一次前向，额外消耗 Dropout RNG 或更新两次 BN。

辅助分支为：

\[
u(x)=Q\,\operatorname{LayerNorm}_{\text{no-affine}}(h_S(x)),\qquad Q\in\mathbb R^{d\times512}.
\]

LayerNorm 的 eps=1e-5；本折独立 CPU float64 RNG 生成 512×d 高斯矩阵，做 thin QR 并把 R 对角符号规范为正，取 Q=Q_QR^T，转 fp32 保存，保证 `QQ^T≈I_d`，无 bias，之后固定。四组读取同一实际 Q，不在不同 worker 中各自重建。记录 seed、float64/fp32 矩阵、hash、Q 以及 `Q(I−11^T/512)` 的数值行秩。

固定投影是本版选择，目的是阻止一个额外可训练映射独自拟合教师目标；它也限制了可用特征映射，后续不能把本版结论直接外推到可训练 projector 的 DiffKD。LayerNorm 只在辅助分支，原分类头仍读取原始 h_S。

这一定义也使 `||u||₂≤sqrt(512)≈22.63`。源训练审计记录两个教师 r_T 范数的分位数、最大值和超出此界的比例，报告目标幅度不可达的限制；不得按反馈或目标成绩临时增加尺度参数、更换 Q 或另拟坐标。

普通特征蒸馏：

\[
L_{feature}=\frac1{Bd}\sum_i\|u_i-\operatorname{sg}(r_{T,i})\|^2.
\]

Diffusion 特征蒸馏：先取 `z_250=a_250*u+s_250*epsilon`，随后固定使用 `250→200→150→100→50→0` 的五次噪声预测。每次从 t 更新到较小的 s：

\[
\widehat r_0=(z_t-s_t\epsilon_\phi(z_t,t))/a_t,\qquad
z_s=a_s\widehat r_0+s_s\epsilon_\phi(z_t,t).
\]

这是 eta=0 的 DDIM 更新。每步只调用一次 epsilon 预测器，并复用该预测计算两个更新式；总计在 t=250/200/150/100/50 调用五次，最后到 0 不再调用去噪器。无额外逐步随机噪声，无输出裁剪、动态阈值或未记录归一化。初始 epsilon 每次学生更新独立采样；两种教师的 DiffKD 条件使用相同噪声表。这里固定从 250 开始，是预先提出的 EEG pilot 设置，不由目标成绩选择。[DDIM](https://arxiv.org/abs/2010.02502)、[DDPM](https://arxiv.org/abs/2006.11239)

\[
\widehat u=D_{\phi,T}(u,\epsilon),\qquad
L_{diffkd}=\frac1{Bd}\sum_i\|\widehat u_i-\operatorname{sg}(r_{T,i})\|^2.
\]

去噪后的学生特征匹配的是**同一 UID 的教师特征**，不是随机生成的类原型或只有总体分布距离。初始学生特征未必就是教师特征的加性高斯噪声，这是假设；记录直接误差与去噪误差，不能假定去噪必定更好。

去噪器拟合结束时清空残留 `.grad`，全部参数设 `requires_grad=False`，但学生分支的五步前向保留 autograd。不得把整个去噪放入 `no_grad()`，不得 detach u。梯度必须穿过固定去噪器、Q 和 LayerNorm，回到 IFNet 骨干。此处使用普通可微去噪匹配，不是上一轮停止梯度 score 系数的 SDS 代理损失。

## 9. 最终 IFNet 的固定训练配置

\[
L_{CE\_FEATURE}=L_{MCC\_FEATURE}=CE(z_S,y)+0.5L_{feature},
\]

\[
L_{CE\_DIFFKD}=L_{MCC\_DIFFKD}=CE(z_S,y)+0.5L_{diffkd}.
\]

等式表示公式形式相同，教师特征目标/去噪器按条件选择。特征 MSE 都先除以 B×d，权重 0.5 固定，不据目标或反馈表现修改。

| IFNet 参数 | 固定值 |
|---|---|
| epoch / checkpoint | 100 / 最终 epoch100 |
| optimizer | AdamW，lr=.001，betas=(.9,.999)，eps=1e-8，wd=.01，foreach/fused=False |
| batch / sampler | 16 / 继承审计后的 100 epoch UID 顺序 |
| scheduler | epoch-end CosineAnnealingLR，T_max=100，eta_min=0 |
| 模型模式 | 学生 train，保留其原 Dropout/BN 行为；评测 eval |
| 精度 / 数据增强 | fp32 / 无新增增强 |
| CE smoothing / class weight | 0 / 无 |
| 梯度裁剪 | 沿用历史学生无裁剪；记录范数，非有限即失败，不临时改规则 |
| 蒸馏启用时点 | epoch1 起；无纯 CE warmup |
| 学生辅助可训练模块 | 无；只更新 IFNet 原骨干和原分类头 |

相同 I_s、训练 UID、dropout 随机流分别配对；Diffusion 拟合、学生去噪噪声、固定投影和诊断使用独立随机流，不改变 Student 数据访问或 Dropout。不能把另一个已训练 IFNet 的状态当作新学生初始化。

## 10. 冻结/更新关系与最终推理

| 阶段 | 更新 | 固定 |
|---|---|---|
| 教师 CE / CE＋MCC | 对应 MIRepNet 原骨干和原分类头 | T0 参考与分割 |
| 特征坐标 | 闭式 PCA 统计拟合 | T0、其他模型 |
| 去噪器拟合 | 对应 epsilon MLP | 教师、PCA、所有 IFNet |
| 最终 IFNet | IFNet 原骨干和原分类头 | 教师、PCA、Q、LayerNorm 定义、去噪器参数 |
| 目标评测/部署 | 无更新 | 只调用 IFNet 原推理路径 |

教师、Diffusion、PCA、Q 只构成训练监督分支。最终导出只需 IFNet；拿掉所有辅助模块后的 logits 必须与其原推理路径一致。

## 11. 预定诊断，避免把辅助 loss 下降当成成功

教师 epoch0/1/5/10：train/feedback BA、Accuracy、CE、每类 recall、预测熵/数量、错误预测置信度；MCC 数值及 2×2 原始/归一化矩阵、行分母；单类别 batch 比例；固定训练诊断 batch 上 CE/MCC 加权梯度范数、比值和夹角。

Diffusion 每 500 step：分噪声段 train/feedback 噪声 MSE；固定 teacher 特征加噪后的恢复误差；记录均值、协方差谱和有效秩；对比初始零噪声预测器。两种教师的 prior 成本分开列出。

学生 epoch1/10/50/100：CE、未加权/加权 feature loss；原生 h_S 和 u 的类别分离、有效秩；u 到同 UID 教师的直接误差、去噪后误差及同/异类统计；固定 batch 上 `dL_feature/dh_S`、stem 参数梯度和 CE/feature 梯度夹角。输入敏感性明确记录冻结五步映射的 `||J_(u→û)v||`，v 是独立预生成的固定单位向量；普通 FEATURE 的对应映射是恒等，作为量级参照。

固定去噪器可能压缩输入差异。应检查逐 trial 输出差异和学生骨干梯度；去噪输出像教师而原生 IFNet 分类不改善，不能说知识已有效传入学生。

现有 IFNet `LinearWithConstraint.forward` 即使在 eval 也会 renorm 分类头权重。只读反馈和导出必须保存/恢复模型状态与 RNG，验证诊断前后 hash 相同，不能让额外评测改变后续训练。

## 12. 实施前必须完成的审计与 smoke

这些是未来实现的验收条件，本次方案制定没有运行它们：

- 核对每折 7/1/1、UID 无交叉、类别编码、数据/preprocessing/hash；所有辅助拟合 UID 属于 D_train。
- 核对两种教师从同一 T0 开始，新增适配 optimizer 均从零开始；T_CE 不含 MCC/Keep/AID。
- MCC 与显式公式及作者标量实现一致；entropy 权重不回传，教师骨干有有限非零梯度。
- MCC 权重为零时，跳过该项并与 T_CE 在相同初值、optimizer 和 UID 随机流下退化为同一教师轨迹。
- 固定 PCA/Q/hash、`QQ^T≈I`、有效 d；全程不更新 Q 或隐含可训练 LayerNorm。
- 先验证 teacher 特征加噪/恢复，再验证固定 DDIM 的五次调用和 timestep 端点；epsilon 预测器不接触标签。
- 仅反传学生 DiffKD 项时，h_S 和至少一个 stem 参数有有限非零梯度；教师、去噪器和统计没有梯度且状态不变。
- feature_weight=0 时跳过去噪、无额外 RNG 消耗，四组应退化为相同 CE 轨迹。
- 一个固定折的短 smoke：Teacher 两次更新、prior 少量更新、学生两次更新及中断恢复；不评目标、不选择超参数。正式 prior 必须从规定初值重新训练 5000 步。
- resume 恢复 model/optimizer/scheduler、所有独立 generator、sampler cursor、固定统计和 hash；同环境下一步可复现。
- 模块实际选择与报告标签一致。smoke 与正式训练调用同一个 loss dispatch，不再各写一套 F/G 式分支。
- 完整评测 registry 包含四组和所有合法参考；每条件必须有九个唯一目标被试，缺失即明确报错，不计算不完整宏平均。
- 脱离教师和所有辅助模块，单独加载 IFNet 重现预测。

如 smoke 暴露数学/接线错误，先修实现再锁定正式版本；如需修改温度、权重、投影、去噪日程或优化范围，应形成新版本，不能拼接旧轨迹。

## 13. 目标评测和统计

九折四条件、两个教师和所有 prior 固定后，才统一生成目标预测。统一评测 T0、T_CE、T_MCC 和四组最终 IFNet；每条保存 UID、y 与对应模型 logits。

教师单列表，给出逐被试及宏平均 BA/Accuracy/CE/每类 recall、预测数量和 collapse，并报告 `T_MCC−T_CE`、`T_CE−T0`、`T_MCC−T0` 的被试配对差。教师指标只作解释，不参与选择，也不能与学生合并当作最终成绩；这一评测不增加训练预算。

汇总包括：

- 逐被试 BA/Accuracy/Kappa/CE、预测数量、每类 recall、collapse。
- 九被试等权均值和被试间样本 SD；注明 `n_seed=1, seed=666, n_subject=9`，SD 不是跨 seed SD。
- 每项配对差 pp、九被试逐项差、Win/Tie/Loss；tie 阈值绝对差≤1e-6 pp。
- 10000 次被试配对 bootstrap，seed=20261030；同一 bootstrap 被试索引表用于所有比较，报告探索性 95% CI。LOSO 训练集合重叠，不把 trial 当独立样本。

预定主要比较：`MCC_DIFFKD − MCC_FEATURE`，检验同一 MCC 教师下去噪流程的额外价值。

预定关键次要比较：

1. `MCC_DIFFKD − CE_DIFFKD`：固定 Diffusion 路径时 MCC 的贡献。
2. `MCC_FEATURE − CE_FEATURE`：普通特征路径时 MCC 的贡献。
3. `CE_DIFFKD − CE_FEATURE`：无 MCC 时 Diffusion 的贡献。
4. `MCC_DIFFKD − CE_FEATURE`：完整组合相对同预算基本流程。
5. 若原 logits KD 严格可配对，报告完整组合对它的差值；否则只列为非配对历史参考。

描述性报告每被试交互项：`(MCC_DIFFKD−MCC_FEATURE)−(CE_DIFFKD−CE_FEATURE)`。初轮不报告 p 值、不按“CI 是否跨零”筛选结果。后续若预注册检验，须预定检验及比较族校正。

## 14. 怎样解释结果与后续机制验证

| 观察 | 本轮可支持的解释 |
|---|---|
| MCC 两条学生路径都改善 | 在两条学生路径中呈方向一致的探索性信号，尚未跨 seed 复现 |
| 教师 MCC 更低但学生不升 | 教师去混淆没有转化为学生收益 |
| DiffKD 优于同教师 FEATURE | 去噪流程对本固定特征空间有额外收益，尚不能归因为多噪声层级的独特机制 |
| 只有完整组合高，两个单因素不一致 | 检查交互和逐被试差异，不能分别宣称两个模块普遍有效 |
| 去噪 loss 降低、Student 骨干梯度近零或 BA 不升 | 辅助恢复有效不等于部署学生有效 |
| 单 seed 差值小或跨被试不一致 | 报告 pilot 不确定性，不宣称等效、无效或普遍有效 |

若完整组合相对 CE_FEATURE 和合法 logits KD 均有正向信号，再固定方案补 seed667/668；各 seed 用各自合法 T0、初始学生和 prior，不复用 seed666 prior 冒充独立模型。跨 seed 先对每个目标被试平均，再做九被试配对统计，不把 9×3 当 27 个独立被试。

后续要归因“Diffusion 有必要”，需另加同空间、同噪声/去噪步骤的简单高斯解析去噪器，以及只加噪声/恒等路径对照；可再比较单噪声与多噪声训练。首轮四组只检验整个去噪蒸馏流程，不声称已经完成这些机制验证。

## 15. 工作量、资源与复用

首轮每折：两条教师适配轨迹各 10 epoch；两个 prior 各 5000 次特征更新；四条 IFNet 各 100 epoch。

九折共 **18 条教师适配、18 个 prior、36 条最终 IFNet**。严格可复用的 CE/logits KD 不重复计算；若需要重新建立任一参考，各增加九条学生轨迹，并在排队前给出实际清单。原 T0、源数据、split 与合法模型输入缓存只读复用，不重复下载或复制官方预训练权重。

每个 GPU worker 的 OMP/MKL/OpenBLAS/NumExpr/BLIS/Torch intra/inter-op 线程均为 1，DataLoader workers=0。进入现有共享队列，遵守其 GPU/主机资源限制；不改变其他用户任务。分别报告教师适配、prior 拟合、学生训练的墙钟、GPU 峰值显存和 CPU 配额；不能把 prior 成本藏在学生训练时间之外。

## 16. 拟定独立实现、产物与报告

待实施入口：

- `configs/experiments/loso/mcc_diffkd_loso_v1.yaml`
- `experiments/distill/run_mcc_diffkd_loso.py`
- `collab/mcc_diffkd.py`

本次仅新增此设计文档，以上三个文件尚未创建。新实现独立于已有 V1 runner，避免修改其源码 hash 后破坏历史 provenance。

物理产物按 `experiments.storage` 放在：

- `/data1/llx/BigSmallcollab/cache/mcc_diffkd_loso_v1_seed666/`：已验证输入引用、UID 特征目标、固定统计。
- `/data1/llx/BigSmallcollab/weights/mcc_diffkd_loso_v1_seed666/`：训练所需权重及临时 resume。
- `/data1/llx/BigSmallcollab/results/mcc_diffkd_loso_v1_seed666/`：逐 trial 预测、history、provenance、小 completion 记录。
- checkout 的真实 `results/distill/mcc_diffkd_loso_v1_seed666/`：用户阅读的 CSV/JSON/Excel/Markdown。

保存 config、源码/环境/commit/dirty status、source/split/UID hash、T0/I_s/prior 初值和终点 hash、模块类型与条件名、独立 RNG、各阶段历史、梯度诊断、预测、配对统计和成本。报告 Excel 按项目约定：student/method 开头、Accuracy/BA 分开、明示 seed/subject 聚合，教师/CE 参考匹配后才着色。单 seed 结果不写入三 seed 的 `docs/exp_log.csv`。

resume 只保留当前必要状态，阶段完成并验证后删除临时 resume；模型清理必须在预测/指标/完成记录可独立读取、下游依赖完成且已检查活动进程后进行，缺权重不能触发静默重训。旧实验需保留的教师、学生、投影器不在此方案的清理范围内。

报告 reader 必须支持只读取已完成的预测和 provenance，并拒绝条件名/实际模块不一致，不能把“有汇总文件”当成全部条件完整。

## 17. 依据与本版研究边界

- MCC：借用类别混淆损失及停止梯度的 entropy 权重；使用源训练集是本版改变的协议，不宣称复现 UDA。
- DiffKD：借用教师特征去噪器辅助学生配对特征蒸馏的思路；固定 Q/PCA、无标签向量 MLP、预训练冻结及固定 DDIM 日程均是本版具体设计。
- DDPM/DDIM：提供加噪与确定性去噪的数学形式，不用其图像生成结果证明 EEG 蒸馏有效。
- 本版没有 AID、旧学生先验、教师保持项、logits KD 或目标域适配。若增加任一项，另定义条件和版本。

本方案的目标是让两个模块的实际作用可解释、最终 IFNet 可独立部署；所有有效性结论等待本版真实训练和目标预测。

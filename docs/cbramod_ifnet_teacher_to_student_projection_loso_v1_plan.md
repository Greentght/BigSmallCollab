# CBraMod 完整教师表示投影到 IFNet：LOSO V1

日期：2026-11-01。版本：`CBRAMOD_T2S_FULL_FEATURE_LOSO_V1`。

状态：方案和只读依赖审计完成；本版训练入口尚未实现，未排队、未启动训练。首轮仅做 CBraMod→IFNet，暂不扩展 MIRepNet。

固定正交P、辅助LayerNorm及512维prior是本轮预先指定的pilot设计，不是已经验证的最优方案。学生/教师预算和feature权重沿用配对实验；若后续改变这些设计，应另建版本。

## 1. 要回答的问题

保留 CBraMod 分类头实际读取的完整表示，不对输入通道或时间片求平均；在教师侧将其投影到 IFNet 原生的 512 维空间，再训练 IFNet。检验：

1. 完整教师表示的 T2S 特征监督是否改善 IFNet，相对匹配的 IFNet CE 基线有多少收益？
2. 在相同 T2S 路径中，MCC 适配教师是否优于只用 CE 继续训练的教师？
3. 同一教师、同一 512 维目标下，DiffKD 是否优于直接特征匹配？

本轮相对旧结果同时改变 feature tap、投影位置、目标维度及相应 Diffusion 空间。只能评价这条新 pipeline，不能把全部差值归为“投影方向”的独立作用，也不能预先认定它一定解决 MCC 退化。

## 2. 已核对的实际模型接口

CBraMod 本轮原生分类路径：

```text
三通道 EEG 的模型专用输入 [B,3,4,200]
  → CBraMod backbone F_T [B,3,4,200]
  → reshape [B,2400]
  → 原 Dropout + Linear(2400,2)
  → 教师 logits
```

旧“三通道”特征缓存已经执行 `F_T.mean(dim=2)`，因此是 `[B,3,200]`；通道平均版又对该缓存的通道轴取均值，得到 `[B,200]`。两者都不能逆恢复完整 `[B,3,4,200]`。

IFNet 原分类头读取 `[B,512]` 的原生 trial 特征 `h_S`。新方案不改变其分类路径；一个学生 forward 同时得到 `h_S` 和 logits。reshape 教师张量仅重排全部元素，不执行平均或池化。

## 3. 数据和配对协议

| 项目 | 固定设置 |
|---|---|
| 数据集 / session | BNCI2014004 / session_3，左右手二分类 |
| 源版本 | wide_fre_band；继承历史模型专用预处理，不以今天 YAML 默认值替换 |
| 种子 | 666、667、668 |
| 外层折 | 每 seed 九折：7 源训练被试、1 源反馈被试、1 目标被试 |
| 划分 | 原 UID/split manifest，不重新随机抽分 |
| 拟合范围 | Teacher 训练、适配、Diffusion 及全部统计仅用 D_train |
| D_feedback | 固定时点只读诊断；不选超参数、checkpoint 或条件 |
| D_test | 全部条件终点锁定后统一评测 |
| 学生配对 | 同 canonical IFNet 未训练初态、BN 初值、100 epoch UID 顺序及 dropout 随机流 |

参考协议为 `sample_utility_adaptive_weight_wideband_v1`，历史 resolved hash：

- 666：`6dfe184f7f6a5ddd8b75e57e6d67c77ce4e68f1bdf4aa349d558c0240539342f`
- 667：`775d37f4ee707b0593270a5c25892135171790c4365b3717b900745bf0b5ffdf`
- 668：`1f11b668046af82a4d3bbea0012f05e5fb8258862f383e668e37c8d25e79cde2`

## 4. 教师表示与固定教师侧投影

对每个冻结教师取分类头前的完整张量：

\[
f_T(x)=\operatorname{reshape}(F_T(x))\in\mathbb R^{2400}.
\]

首轮采用固定线性正交投影 `P_T∈R^(512×2400)`。独立 CPU float64 RNG 生成 `2400×512` 高斯矩阵，thin QR，并规范 R 对角线符号，取 Q 的转置。保存 float64/fp32 实际矩阵及 hash，核验 `P_T P_T^T≈I_512`。同一 seed、同一折的 CE/MCC 和 FEATURE/DIFFKD 条件读取同一个 P。

随机流由 `experiment_id/stage/seed/fold` 的 SHA-256 派生，记录实际整数 seed；P 不消耗学生、教师适配或 prior 随机流。

辅助分支：

\[
z_T=\operatorname{LN}_{\text{no-affine}}(P_T f_T),\quad
z_S=\operatorname{LN}_{\text{no-affine}}(h_S),
\]

LayerNorm eps=1e-5；二者均为512维。学生不再接受 512→32 或三组学生投影；原分类头始终读取原始 `h_S`。P、Teacher 和目标缓存固定，只有 IFNet 骨干与原分类头更新。

本版固定 P 的目的，是给出没有可训练 adapter 吸收损失的明确对照；它不是已经学会师生语义对应的映射。不引入 CE 参考学生来拟合 P，不把 epoch10 warmup 特征当作 epoch100 CE 特征。可学习投影器及其训练目标属于后续独立版本。

2400→512 仍是有损投影，不能称为完整保留全部教师信息。每个教师必须报告投影前后的决策信息保留诊断：原分类器 `W,b` 的 `||W(I-P_T^T P_T)||_F/||W||_F`，以及 train/feedback 上原 logits 与 `W P_T^T P_T f_T+b` 的差异、预测一致率、BA。该诊断针对 LN 前的线性投影；LN 后另报告特征范数、方差和类别分离，不把前一个性质冒充归一化后仍严格成立。

数学边界：若固定 P 满足 `PP^T=I`，未加归一化时 `||Pf_T-h_S||²` 与 `||f_T-P^T h_S||²` 对 h_S 的梯度相同，差值仅含与 h_S 无关的项。投影方向本身没有天然优势；新特征内容、维度与归一化也共同影响结果。

## 5. 正式四组

从同一折同一个 T0 独立得到：`T_CE` 用 CE 适配10epoch，`T_MCC` 用 CE+MCC 适配10epoch。

| 条件 | 冻结教师 | IFNet 损失 |
|---|---|---|
| CE_T2S_FEATURE | T_CE | CE + 0.5×直接512维配对MSE |
| MCC_T2S_FEATURE | T_MCC | CE + 0.5×直接512维配对MSE |
| CE_T2S_DIFFKD | T_CE | CE + 0.5×五步去噪后512维配对MSE |
| MCC_T2S_DIFFKD | T_MCC | CE + 0.5×五步去噪后512维配对MSE |

直接特征损失：`mean((z_S-stopgrad(z_T))²)`，按 batch×512 求均值。保留全部训练样本 CE；从 epoch1 启用特征项；没有额外纯 CE warmup、logits KD、教师保持项或 AID。

教师继续使用真实原生分类路径计算 CE/MCC，不在教师分类前插入 P。MCC 为现有最小类别混淆实现：temperature=2、weight=1；entropy 权重停止梯度，概率本身及教师骨干保留梯度。

## 6. DiffKD 在同一512维空间实现

每折为 T_CE、T_MCC 分别拟合一个无标签噪声预测器，训练数据是同 UID 的 `z_T`。两教师 prior 共用网络初态、UID抽样序列、t与epsilon序列。

| 参数 | 固定值 |
|---|---|
| 网络 | concat(z_t,32维固定sinusoidal时间编码)→Linear128→SiLU→Linear128→SiLU→Linear512 |
| 类别/通道条件 | 无；三个通道及四个时间片已共同进入固定P，未分别求平均 |
| DDPM | 1000步；linear beta 1e-4→0.02；训练t均匀抽样1–1000 |
| 特征抽样 | D_train trial均匀有放回，batch=min(128,N_train) |
| 优化器 | AdamW，lr3e-4、betas(.9,.999)、eps1e-8、wd1e-4 |
| 更新/终点 | 5000次/step5000，fp32，梯度裁剪1.0，无反馈早停、EMA、BN或Dropout |
| 诊断 | 每500步，固定train/feedback噪声表，按四个噪声区间报告MSE与零预测器参考 |
| 学生去噪 | z_S在t250加独立高斯噪声；DDIM eta0：250→200→150→100→50→0 |

冻结 prior 参数，但对学生输入保留五步 autograd；去噪输出与**同 UID**的 z_T 做 MSE，不 detach z_S，不对整个去噪调用 no_grad。无输出裁剪、额外归一化或逐步随机噪声。

512维先验比历史32维先验更难拟合；本版固定相同5000更新，只能据分噪声诊断说明是否学好，不能据目标结果临时改宽度或训练步数。

## 7. Teacher 与 IFNet 预算

T0 的真实历史 recipe 保存于逐折 `resolved_run.json`，正式 T0 为50epoch；重建时读取该记录，不读取当前模型 YAML 默认值。T_CE/T_MCC 都从同一 T0 开始，适配10epoch：新Adam、lr1e-4、wd1e-6、batch8、常数LR、fp32、全参数梯度裁剪1.0、eval但允许autograd。CE/MCC适配共享UID顺序。

IFNet 四组统一：100epoch，batch16，AdamW lr.001、wd.01、betas(.9,.999)、eps1e-8，foreach/fused=False；epoch-end cosine T_max100、eta_min0；fp32，无新增增强、label smoothing或class weights，沿用历史无学生梯度裁剪。每个batch只做一次学生前向，原BN/Dropout行为保持。

固定epoch100；所有目标评测在三seed全部正式条件训练终点固定后统一进行。IFNet的约束线性层即使eval也可能renorm，诊断要保存/恢复状态及RNG并验证前后hash相同。

## 8. 真实产物缺口与重建边界

2026-11-01只读审查：

- 27折正式 T0/T_CE/T_MCC 权重，共81份均已被旧runner的完成清理删除；未发现同hash正式副本。
- 27份 teacher_channel_targets.npz 保留，但native只有 `[N,3,200]`；不存在完整 `[N,3,4,200]` 缓存。
- seed666/fold1留有2epoch smoke教师，不能代替正式50epoch/10epoch教师。
- 官方预训练、canonical模型输入、27折UID/split/schedule、三seed IFNet未训练初态、27份epoch100 BASE_CE checkpoint与审计均保留。
- `reference_student_train.npz` 是epoch10 warmup特征，不能当作epoch100参考空间。

因此本版不能宣称“只做前向重导出即可”。需要先恢复正式同hash权重，或按历史resolved**显式重建**27个T0和54条适配轨迹，保存完整特征。重建读取历史初态seed、UID与随机顺序；比较原教师记录的model-state hash，而不是只比较torch.save文件hash。

若重建模型state hash完全匹配，旧三通道/meanpool结果可按教师身份、seed、UID和学生初态继续做配对比较。若不能匹配，记录为新教师；新四组内部仍严格配对，但旧方法结果只能作历史参考。首轮不静默增加重跑旧方法的预算，也不把新旧教师拼为同一运行。

历史记录：

```text
/data1/llx/BigSmallcollab/results/distill/cbramod_mcc_diffkd_loso_v1_seed{seed}/seed_{seed}/folds/{split}/resolved_run.json
/data1/llx/BigSmallcollab/cache/cbramod_mcc_diffkd_loso_v1_seed{seed}/seed_{seed}/folds/{split}/teacher_channel_targets.npz
/data1/llx/BigSmallcollab/cache/artifact-loso/model_inputs/BNCI2014004/{cbramod,ifnet}/
```

## 9. 验收和机制诊断

启动前重新核验 canonical 输入、source与manifest内容hash，并与历史split/resolved的身份记录匹配。本次只读审计确认了文件存在、shape及身份字段，没有重算全部大文件内容hash。

正式训练前在固定单折运行train-only smoke：验证完整feature shape与UID、teacher分类logits与原forward一致、固定P的正交性/hash、四组loss dispatch、仅特征项反传能到达IFNet stem。在学生训练/反传阶段Teacher/P/prior不更新；prior参数无梯度但学生梯度能穿过五步去噪。教师重建/适配和prior拟合是各自独立的可更新阶段。

还需核验：feature_weight=0时跳过所有辅助前向与额外RNG并退化为同一CE轨迹；同一IFNet初态与schedule；checkpoint/resume恢复优化器、scheduler、RNG及sampler cursor，下一步一致；最终只加载IFNet可重现预测。

记录 teacher class-output指标与特征指标，不把MCC下降、MSE下降当成Student成功。学生epoch1/10/50/100报告CE/feature loss、有效秩/类间类内距离、CE与feature梯度范数/夹角。DiffKD报告直接与去噪后的配对误差、输入敏感性和stem梯度，检查去噪器是否仅压缩样本差异。

## 10. 评测、比较与报告

这是查看过同数据集既有目标结果后的探索性后续实验；冻结本版参数不代表整个研究过程从未看过目标成绩。主指标为Balanced Accuracy；Accuracy、Kappa、CE、每类recall、预测数量、collapse独立报告。教师 T0/T_CE/T_MCC 单列；IFNet_CE为同架构同协议reference，不把教师成绩当学生成绩。

预定主要比较：

1. MCC_T2S_FEATURE − CE_T2S_FEATURE。
2. MCC_T2S_DIFFKD − CE_T2S_DIFFKD。

预定次要比较为同教师DIFFKD−FEATURE，以及四组各自相对匹配的IFNet_CE。旧三通道/meanpool仅在教师state身份匹配时作为严格配对；否则明确历史参考。

每seed先等权平均九个目标被试，再等权平均666/667/668；SD报告三seed均值的sample SD。逐被试差值先跨seed平均，再做九被试的配对bootstrap10000次，固定seed20261101；报告探索性95%CI、Win/Tie/Loss，tie为绝对差≤1e-6 pp；不把27个seed×subject或trial当独立被试。每格明示seed数与完整折数；缺折不写完整宏平均。

输出Excel/CSV/JSON和不超过10行结论。Excel按项目模板的student/method列、教师与CE参考、匹配比较色及明确指标制作。用户未要求登记log，本版不自动改docs/exp_log.csv。

## 11. 工作量、资源与保存

正式四组共108条IFNet轨迹（3seed×9fold×4条件），每条100epoch；54个512维prior，每个5000更新；匹配IFNet CE reference可复用。因正式教师缺失，另需27条50epoch T0和54条10epoch教师适配，不能把这项计算隐藏在“缓存复用”中。先完成单折train-only smoke，再按冻结配置完成全27折；未启动任何本版任务。

所有GPU worker数值库及Torch intra/inter-op线程均为1，DataLoader workers0。进入共享队列并遵守GPU/主机资源限制，不改其他用户进程。审计时队列无活动项目任务，但GPU有外部高利用率任务，实际开跑时间由队列资源条件决定。

拟定独立入口：

- configs/experiments/loso/cbramod_t2s_full_feature_loso_v1.yaml
- experiments/distill/run_cbramod_t2s_full_feature_loso.py
- collab/cbramod_t2s_projection.py

这些入口本次尚未实现。通过experiments.storage解析的新产物目录为：

- `/data1/llx/BigSmallcollab/cache/cbramod_t2s_full_feature_loso_v1_seed{seed}/`：完整教师targets、固定P与输入引用。
- `/data1/llx/BigSmallcollab/weights/cbramod_t2s_full_feature_loso_v1_seed{seed}/`：教师、prior及训练权重。
- `/data1/llx/BigSmallcollab/results/distill/cbramod_t2s_full_feature_loso_v1_seed{seed}/`：阶段history与provenance。
- checkout真实 `results/distill/cbramod_t2s_full_feature_loso_v1/`：用户阅读的Excel/CSV/JSON。

保存resolved config、source/environment/commit状态、split/UID/preprocess/model-state hash、完整feature tap与targets、P、每阶段history、梯度与投影信息保留诊断、逐trial预测、完成记录与资源成本。新教师权重和完整targets需用于后续投影诊断，不沿用旧runner整目录rmtree；临时resume在完成且可独立读取后清理，保留范围依据后续复用和既有存储约定决定。

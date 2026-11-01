# MIRepNet → 三种学生、五数据集：T2S FEATURE / DiffKD

版本：`MIREPNET_T2S_ALLSETS_LOSO_V1`，2026-11-01。
用户确认：标准 LOSO、复用合法同折教师、四组方法加同初态 CE 基线、seed 666/667/668。
执行状态和队列 ID 以 `results/distill/mirepnet_t2s_allsets_v1/dispatch.json` 为准；此文不包含实验准确率。

## 数据与比较边界

| 数据集 | 任务/session | 被试数 | 三 seed 折数 |
|---|---|---:|---:|
| 004 / BNCI2014004 | 左右手，session_3 | 9 | 27 |
| 001 / BNCI2014001 | 左右手，0train | 9 | 27 |
| 001-4 / BNCI2014001-4 | 四分类，0train | 9 | 27 |
| 15001 / BNCI2015001 | 二分类，session_A | 12 | 36 |
| AlexMI | 保留既有 RH/feet 二分类 recipe | 8 | 24 |

每折其余 N−1 个被试全部作为源训练集，1 个目标被试只在正式终点锁定后评测。
本版本无反馈被试、验证早停或目标数据优化；与旧 7+1+1 pilot 独立汇总。
所有模型按同一个物理 trial UID 配对；核对标签、被试、类别映射、session、源文件及输入文件 hash。
宽域源使用已有 `wide_fre_band`，AlexMI 保持既有源与窗口；各模型专用预处理从合法历史 recipe 继承。
MIRepNet 既有按被试 EA 可能使用目标被试的无标签输入统计；它不授权目标 trial 进入适配、prior 拟合或学生梯度。

## 已核验的教师来源

141 个同 seed、同折 MIRepNet checkpoint 均存在且通过 SHA 校验，缺失时直接报错，不由读取器隐式重训。
004/15001 读取 source-refreshed baseline 的 manifest；001/001-4 读取 `wideband_npy_v3` 的 result.json；AlexMI 读取保留的历史 manifest/target provenance。
源 T0 预算保留实际历史值：001-4 为20epoch，其余为10epoch；适配另加10epoch。
当前模型 YAML 即使改过训练超参数，也不覆盖这些已训练教师的历史 resolved recipe。

## 保留原生教师表示并投影到学生维度

真实分类头前向接口已经核验：MIRepNet256、IFNet512、EEGNet496、ADFCNN552维。
MIRepNet 原生分类路径中的 token pooling 保留；蒸馏不额外平均教师输出。
ADFCNN 原生分类卷积读取的 `[8,1,69]` 特征图仅重排为552维，所有元素保留。

三种学生均采用固定高斯 thin-QR **扩维**矩阵 `P[d_student,256]`，满足 `PᵀP=I256`。
不能对这些高矩阵声称行正交。保存实际 float64/fp32 P、派生 seed、文件 SHA 与正交误差。
P 的随机流独立于教师、prior 和学生；同一 student/seed/fold 的所有方法共用 P。

`z_T = LayerNorm(P h_T)`，`z_S = LayerNorm(h_S)`，均无 affine，eps=1e-5。
学生原分类头始终读取原始 h_S；辅助分支没有可训练 projector。
扩维在 LayerNorm **之前**保留教师向量和线性读出信息；LayerNorm 后不能沿用严格恢复结论。
记录原分类器权重的残差、投影恢复 logits 误差、一致率和训练集 BA。
固定随机投影并不等于已经学会师生的语义对应，本轮检验整条 pipeline 的实际效果。

## 五个学生条件

| 方法 | 教师 | 学生目标 |
|---|---|---|
| none | 无 | CE |
| CE_T2S_FEATURE | T_CE | CE + 0.5 × 配对特征 MSE |
| MCC_T2S_FEATURE | T_MCC | CE + 0.5 × 配对特征 MSE |
| CE_T2S_DIFFKD | T_CE | CE + 0.5 × 五步去噪后配对特征 MSE |
| MCC_T2S_DIFFKD | T_MCC | CE + 0.5 × 五步去噪后配对特征 MSE |

T_CE/T_MCC 从同一个保留 T0 独立复制，只在本折源训练 trial 上适配。
教师全参数更新，eval但保留autograd；新Adam lr1e-4、wd1e-6、batch8、10epoch、固定末epoch、梯度范数上限1。
T_CE 仅CE；T_MCC 为CE+MCC，温度2、权重1，entropy样本权重停止梯度，概率保留梯度。
MCC 为类别混淆矩阵损失；此处是源域正则，不是目标域适配或小模型指导教师。

每个 student/seed/fold 的五组从完全相同的新初态、BN初值开始，使用相同UID顺序和Dropout随机流。
CE baseline重新成对训练；旧CE仅因seed相同不能冒充严格配对。
学生均100epoch，optimizer/lr/batch/wd由本次冻结的各模型各数据集LOSO YAML读取并写入resolved.json。
从epoch1使用特征项，无纯CE warmup、logits KD、AID、Keep或目标选择。
每个batch只前向学生一次；学生原骨干与分类头更新，教师/P/prior固定。

## DiffKD

每个教师、每个学生空间各拟合一个无标签MLP：`[z_t,time32] →128→SiLU→128→SiLU→d_student`。
教师CE/MCC两个prior共用初始状态、训练UID抽样、t与epsilon随机流。
DDPM1000步，linear beta1e-4至.02；源训练trial均匀有放回，batch最多128。
AdamW lr3e-4、wd1e-4、梯度裁剪1、固定5000更新，无早停/BN/Dropout/EMA。
每500步报告四个噪声区间的训练MSE与零预测器参考；本版本没有反馈集。
学生在t250加独立噪声，DDIM eta0固定250→200→150→100→50→0。
五次冻结MLP前向保留输入梯度；输出与同UID的z_T匹配，不能detach学生或用no_grad包住整个去噪。

## 规模、检查与执行

共141个dataset/seed/fold组，复用141个源教师，新增282条教师适配轨迹。
三个学生共423个配对单元，每单元none+四方法：合计2115条100epoch学生轨迹。
846个prior各5000更新，423个固定P；教师适配在三种学生间共享。
先依次运行五数据集首折、seed666的CPU短smoke：仅取16条源训练trial，教师/学生两batch、prior四更新，不评目标。
正式任务进入现有共享GPU队列；每进程数值库与Torch intra/inter-op线程1，DataLoader workers0，不改变已有任务。
optimizer/scheduler、全局与独立RNG、epoch/order游标写临时resume；恢复检查下一步轨迹一致。
完成reader接受小完成记录与预测，缺权重不允许触发静默重训。

## 评测、报告与保存

141个训练组全部完成后，监控器才提交五个dataset级评测队列任务。
目标逐trial保存UID、标签、logits与hash；教师T0/T_CE/T_MCC独立于最终学生列表。
Accuracy、BA、Kappa、CE分别报告；每seed等权平均目标被试，再等权平均三个seed，SD为三个seed均值的样本SD。
方法相对none、MCC−CE及DIFFKD−FEATURE给出配对差；先跨seed平均每个被试差，再按被试bootstrap10000次，seed20261101。
完整注册表需要2538条唯一模型/方法/seed/被试记录；缺失即拒绝完整汇总，不写零、不自动登记exp_log。
Excel按项目规范分Accuracy/BA表、teacher在前、student none在前，颜色比较同学生CE与对应T_CE/T_MCC。
模型输入、targets、P、教师/prior/学生权重与resume位于 `/data1/llx/BigSmallcollab/{cache,weights,results}/`。
用户阅读的CSV/JSON/Excel位于checkout真实 `results/distill/mirepnet_t2s_allsets_v1/`。
源教师保留；新教师、学生及prior权重本次也保留以供后续复用，不执行批量清理。

入口：`python -m experiments.distill.run_mirepnet_t2s_loso --mode enqueue`。
固定参数文件：`configs/experiments/loso/mirepnet_t2s_allsets_v1.yaml`。

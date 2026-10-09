# EEG 样本价值学习：BNCI2014004 实施方案

日期：2026-10-09 更新。状态：P0–P7 已完成。九折 Teacher / shared warm-up / 六条件 Student 矩阵、epoch100 评测和最终报告均已生成；训练隔离、checkpoint 身份、目标 UID、shuffle UID 对齐与 batch 多重集合审计通过。正式结果是 seed666 的 pilot，不是跨 seed 结论。

## 1. 当前环境与固定范围

本次实际检查到 `/home/lixinli/BigSmallCollab`、`/data1/llx/BNCI2014004/{X.npy,labels.npy,meta004.csv}` 和 `/data1/llx/pre_weight/mirepnet.pth` 存在。原设计第 9 节的环境缺失描述已不适用于这里。实现基准之前的项目 commit 为 `aa5b0cc11f62146be854c4fda5aa16a06512f3fa`；开始时有未跟踪的 `docs/prompt.txt`，实施提交不得包含该文件。

已确认 `mirepnet` 环境为 Python 3.10.18、PyTorch 2.1.0+cu118，支持 `torch.func.functional_call`，CUDA 可用。九折 preflight 已重算数据与预训练文件 hash、验证 1400 条 trial 的形状/标签及 split 计数，并将 manifest 写入外置 artifact store。fold 0 smoke 使用 CPU 四线程完成 Teacher 10 epoch、Student warm-up 10 epoch 和六条件各 3 epoch，没有运行目标评测。最近检查时 10 张 RTX 3090 均有 99–100% 利用率；正式训练尚未启动，需等到有足够 GPU 余量再调度。

第一轮沿用 [canonical LOSO 数据规格](../configs/reproductions/loso_five_datasets_v1.yaml)：**仅 session_3，3 通道、250 Hz、每 trial 前 1000 点、左右手两类**。所谓完整目标被试，指该选定 session 的全部 trial。改为全部 session 必须另建数据协议并重跑全部条件。

当前放行状态：P0 九折 preflight 通过；P1 8 项针对性测试、`mirepnet` 编译与配置解析通过；P2 fold 0 CPU smoke 六条件、replay、控制器和无目标评测检查通过；P3–P7 正式 Teacher、Student 矩阵、epoch100 目标评测和最终报告均完成。完整输出位于 `/data1/llx/BigSmallcollab/results/distill/sample_utility_adaptive_rl_loso_pilot_v1/`。

实施时发现：该 canonical YAML 当前 SHA256 与伴随 source snapshot 中记录的 `spec_sha256` 不一致，原 canonical runner 因而拒绝加载。pilot 不修改或刷新历史 snapshot；新 runner 会将两个实际值都写入 resolved config，严格锁定所需 session/形状/试次数/类别字段，并逐文件验证 snapshot 中的数据源 hash 和 trial UID/标签。九折 preflight 已通过。报告保留 `canonical_manifest_spec_hash_matches=false` 的事实。

| 项目 | 已解析的既有设定 | 来源 |
|---|---|---|
| Student | IFNet，100 epoch，随机初始化，前 10 epoch CE | `configs/models/ifnet.yaml`、既有 LOSO runner |
| Student optimizer | AdamW，lr=0.001，weight_decay=0.01 | `run_loso_small_baselines.py::_optimizer` |
| batch | 16，drop_last=false，num_workers=0 | 既有 LOSO KD DataLoader |
| scheduler | CosineAnnealingLR，T_max=100，eta_min=0；每 epoch 末推进一次 | `run_loso_distillation.py::_fit` |
| KD | CE 系数 1，lambda_KD=0.5，tau=2，feature loss=0 | `configs/reproductions/loso_distillation_v1.yaml` |
| Student 输入 | 4–16 / 16–40 Hz filter bank，通道由 3 变 6；不做目标分布适应 | IFNet YAML / adapter |
| Teacher | MIRepNet，10 epoch，Adam，lr=0.001，batch=8，weight_decay=1e-6，cosine | MIRepNet YAML / canonical Teacher runner |
| Teacher 输入 | 8–30 Hz 带通 → 各训练被试独立 EA → IDW 补至 45 通道 | `_prepare_mirepnet` |
| seed / 评测 | 666；固定 Student epoch100；主指标 BA，附 Accuracy、Kappa | 本 pilot 固定规格 |

以上是从源码和 YAML 读取的值。实施 preflight 还需导出实际 optimizer 的 betas、eps、amsgrad、foreach/fused 等默认值以及实际 LR；不得用历史结果替代本轮解析。

## 2. 拆分、UID 与缓存身份

外层内部 index 为 `t=0..8`，反馈 index 为 `(t+1)%9`。`train = (subject != t) & (subject != feedback)`，反馈和测试分别是相应被试的全部已选 trial。被试显示 ID 为 S01..S09。

| 目标 | 反馈 | 训练 trial | 反馈 trial | 测试 trial | 每 epoch 训练 batch |
|---|---|---:|---:|---:|---:|
| S01 | S02 | 1120 | 120 | 160 | 70 |
| S02 | S03 | 1120 | 160 | 120 | 70 |
| S03 | S04 | 1080 | 160 | 160 | 68 |
| S04 | S05 | 1080 | 160 | 160 | 68 |
| S05 | S06 | 1080 | 160 | 160 | 68 |
| S06 | S07 | 1080 | 160 | 160 | 68 |
| S07 | S08 | 1080 | 160 | 160 | 68 |
| S08 | S09 | 1080 | 160 | 160 | 68 |
| S09 | S01 | 1080 | 160 | 160 | 68 |

该表来自当前冻结 trial manifest 的计数，正式 preflight 必须逐折重验。1080 条的尾 batch 为 8 条，CE 和 KD 都覆盖它。

复用 `_load_trials` 返回的 canonical UID `(zero_based_subject, raw_cache_row)`，同时保存 dataset namespace 和 `(subject, selected_trial_index)` 对照。UID 是对齐键，不能作为控制器输入。按 UID 重排 Teacher / S10 缓存后逐项验证标签、集合和顺序；仅标签相同不足以证明对齐。

每个 fold 保存 `split_manifest.json`：目标和反馈的实际 ID 映射、七个训练被试、各 split 全部 UID/标签/类别计数、源文件和 trial-manifest hash、预处理版本、split hash。断言三集合互斥且并集等于全部选定 trial，并验证 Teacher 训练 UID 恰等于 D_train。

旧八源 Teacher 权重与缓存不能通过删掉反馈行复用。新缓存 key 至少包含 `protocol_id + target + feedback + seed + split_hash + Teacher_checkpoint_hash + preprocessing_hash`。Teacher 和 warm-up 都从头生成，使用独立命名空间。

MIRepNet 预训练权重的历史 snapshot 记录 SHA256 为 `432288958007e344a5a84a9ffe9d0e5e5c0cb616aef86c85522375a3f4da9aaf`，需重算确认。现有本地资料不足以证明其预训练从未见过本数据集；报告只能据本轮记录声明反馈/目标未参与微调和 Student 普通训练，预训练暴露情况另记为未知，不能扩大留出声明。

## 3. 代码接入与存储

已接入文件职责如下；正式训练和报告已经通过后续放行门槛。

| 文件 | 职责 |
|---|---|
| `configs/experiments/sample_utility_adaptive_rl_loso_pilot.yaml` | 六条件、七源 split、继承配置、控制器与 RNG/报告规格 |
| `collab/sample_utility.py` | MLP、detach 状态输入、per-trial KD、meta/REINFORCE 更新 |
| `collab/lookahead.py` | 参数/buffer/RNG 隔离、IFNet 约束模型视图、可微 AdamW 单步 |
| `experiments/distill/sample_utility_splits.py` | 无模型依赖的固定九折划分和 batch 内 replay 置换函数 |
| `experiments/distill/sample_utility_protocol.py` | split、Teacher cache、共享 warm-up、schedule、replay、manifest/汇总 |
| `experiments/distill/run_sample_utility_adaptive_rl_loso.py` | 分阶段 CLI、依赖与完成门槛、resume/失败处理 |
| `tests/test_sample_utility_adaptive_rl_loso.py` | 划分、optimizer/meta/policy、状态隔离、shuffle/恢复验证 |

复用入口：

- `run_loso_five_datasets.py::_verify_source_files/_load_trials/_config/_prepare_mirepnet/_train_mirepnet`：数据与 Teacher 配方。只给 Teacher 传 D_train；旧主循环固定八源且会推理目标，不能照搬。
- `run_loso_small_baselines.py::_config_for`：Student 配置。不要把 Teacher 专用 `_config` 的非 MIRepNet 分支用于 IFNet。
- `run_loso_distillation.py::_optimizer/_rng_state/_restore_rng`：Student optimizer 与状态管理思路。旧 `_baseline` 依赖八源 UID，本轮重跑 CE，只继承训练配方。
- `run_delayed_kd_teacher_correct_paired.py::_schedule/_create_warmup/_train_branch`：显式 batch schedule 和 epoch10 分叉的设计；该 runner 是 few-shot，不能直接用于本轮 LOSO。
- `loso_teacher_cache.py`：UID、checkpoint provenance、原子保存与锁的检查方式；旧 cache loader 固定八源身份，需新增七源缓存实现。
- `eval/stats.py::paired_stats/holm`：以 `balanced_accuracy` 为 metric 复用 bootstrap/配对统计；补充 Tie/Loss、完整九折校验和报告。

新 runner 自行校验本 pilot schema 并读取上述既有配置；当前 `run_distill.py` 的 method schema 不承载这两种控制器。不得添加一个只转发旧命令却没有反馈算法的 runner。

所有训练产物通过 `experiments.storage.require_external_output` 保存：

```text
/data1/llx/BigSmallcollab/
  cache/sample_utility_adaptive_rl_loso_pilot_v1/
    BNCI2014004/target_XX_feedback_YY/seed_666/<split_hash>/
      split_manifest.json
      student_inputs.*
      teacher_train.npz            # logits/feats/y/UID
      reference_student_train.npz  # S10 feats/y/UID
      train_schedule.* / feedback_schedule.*
  weights/sample_utility_adaptive_rl_loso_pilot_v1/
    target_XX_feedback_YY/seed_666/<split_hash>/
      teacher_final.pt / teacher_provenance.json
      warmup_epoch10.pt
  results/distill/sample_utility_adaptive_rl_loso_pilot_v1/
    resolved.yaml / environment.json / execution_snapshot.json
    folds/target_XX_feedback_YY/seed_666/<condition>/
      checkpoint_final.pt / checkpoint_resume.pt
      controller_final.pt          # 两个学习组
      history.csv / step_metrics.* / replay_epoch_*.npz
      predictions.npz / metrics.json / manifest.json
    metrics_by_subject.csv / paired_comparisons.csv / report.md
    smoke/<smoke_id>/                 # 环境、代码和 resolved config
    folds/<target...>/smoke/<smoke_id>/ # 每折隔离的 smoke 产物
```

共享原始数据仍在 `/data1/llx/BNCI2014004/`。不创建 checkout 内的 `results/`、训练缓存或 symlink。这里的 `BigSmallcollab` 大小写按 storage 模块使用。

## 4. 第一版需固定的新增配置

以下是 pilot 的工程默认值，并非实验验证过的最优值。写入新 YAML 和 resolved config，整个正式矩阵使用同一版本。

```yaml
protocol_id: sample_utility_adaptive_rl_loso_pilot_v1
dataset: BNCI2014004
teacher: mirepnet
student: ifnet
seed: 666
split: target_t_feedback_next_subject_train_remaining_7
epochs: 100
warmup_epochs: 10
conditions: [BASE_CE, DELAYED_KD_ALL, ADAPTIVE_WEIGHT_KD, RL_GATE_KD,
             ADAPTIVE_SHUFFLE, RL_SHUFFLE]
model_selection: final_epoch
loss:
  ce_weight: 1.0
  lam_kd: 0.5
  temperature: 2.0
  lam_feature: 0.0
  kd_denominator: actual_batch_size
controller:
  hidden_dims: [128, 64]
  activation: silu
  output_min: 0.05
  output_max: 0.95
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0
  scheduler: none
  update_every_student_batches: 1
  feedback_batch_size: 16
  entropy_coefficient: 0.0
  gradient_clip: none
rl:
  baseline_initial: 0.0
  baseline_ema_decay: 0.9
  advantage_normalization: none
  log_prob_reduction: sum
execution:
  precision: fp32
  amp: false
  student_optimizer_foreach: false
  student_optimizer_fused: false
  gradient_accumulation: 1
  checkpoint_every_epochs: 1
  num_workers: 0
report:
  primary_metric: balanced_accuracy
  bootstrap_draws: 10000
  bootstrap_seed: 666
  primary_pvalue_family: [adaptive_vs_kd_all, rl_vs_kd_all]
  primary_pvalue_correction: holm
```

MLP 为 Linear(d_input,128) → SiLU → Linear(128,64) → SiLU → Linear(64,1)，输出统一为 `0.05 + 0.90*sigmoid(score)`，只有一次sigmoid。Student AdamW 的数值参数继承既有配方；关闭 foreach/fused 便于验证可微公式，须记录这一实现差异。控制器 Adam 的 betas/eps 也写为解析值；两路线共用相同初始化张量和 hash。前两层采用 seeded Linear 默认初始化，最后一层 weight 采用 normal(mean=0,std=1e-3)、bias=0，使输出接近 0.5；保存初始化规则和实际输出分布。

KD 用 `reduction='none'` 后对类别求和得到每 trial 的 KL，再计算 `tau² * sum(v_i*KL_i) / B`；**B 是当前 batch 的实际长度**，包括尾 batch 的 8，不是权重和。全 1 应数值等于旧 runner 的 `batchmean`。若把分母固定为配置常数 16，则尾 batch KD 会减半，必须另记协议差异。

训练 schedule 预生成 100 epoch，各 epoch 对全部 D_train 一次置换；六条件共用，warm-up 消费前 10 epoch，续训从第 11 epoch 指针开始。feedback 独立 shuffle/cycle，每次控制器更新取一个 batch，耗尽后重排；A/B 使用相同预生成 feedback schedule。所有尾 batch 保留，不用 feedback 标签选择 batch。

RNG seed 固定为：Student 初始化和训练流 666；训练 schedule 666；其余流 `666 + offset + 100*t`，offset 分别为 controller_init=10000、action=20000、feedback=30000、adaptive_shuffle=40000、rl_shuffle=50000。创建模型/控制器、导出缓存时保护真实 Student RNG。保存 Python、NumPy、Torch CPU/CUDA、动作 generator、shuffle generator 和所有 schedule cursor。

progress 按后 90 epoch 的真实 step 归一化：`postwarmup_step/(total_postwarmup_steps-1)`，首步 0、末步 1；不因 resume/smoke 改变正式定义。

## 5. Teacher、共享 warm-up 与特征导出

1. 对九折各自构建七源 split；MIRepNet 从同一预训练权重重新初始化，seed666 固定分类头和训练 RNG，只用 D_train 按既有配方微调 10 epoch。固定最后 epoch，不使用反馈或目标选择 Teacher。保存最终 checkpoint、训练历史、初始权重 hash、训练 UID 与预处理 provenance。
2. 冻结 Teacher，以 eval 导出 D_train logits/features。本方案不需要 Teacher 对 D_feedback 或 D_test 推理；只预处理和导出 D_train。
3. IFNet 从 seed666 初始化，在 D_train CE 训练 10 epoch，Cosine 的 T_max 始终为 100。epoch10 在 scheduler 已推进、epoch11 尚未开始的边界保存模型、optimizer、scheduler、训练 RNG、schedule/hash/cursor 和历史。
4. 创建冻结参考 S10 和两个控制器的共享初始化快照。用状态隔离的 S10 eval 视图导出 D_train 特征；缓存导出不得改 warm-up checkpoint 或真实续训 RNG。
5. 六条件恢复同一 warm-up 全部状态；不在 epoch11 重建 optimizer 或重置 LR。续训各有自己的可训练 Student，S10 始终冻结。

feature tap 使用模型 adapter 的实际返回值。当前源码预计 MIRepNet pooled dim=256、IFNet flatten dim=512，二分类概率各 2 维，输入预计 `256+512+2+2+1=773`；以真实导出 shape 为准建 MLP并记录 tap/维度，不写死 773。

输入为 `concat(L2norm(h_T), L2norm(h_S10), softmax(z_T), softmax(z_St_eval), progress)`，全部 detach。L2norm 按每trial的feature维归一化，eps=1e-8；控制器输入概率使用未温度缩放的 softmax，KD 单独使用 tau=2。当前 Student 的概率来自只读模型视图，不能用 train-mode dropout 输出冒充 eval 概率。

## 6. 临时状态与可微 optimizer 的实现门槛

**IFNet 的前向会改权重。** `models/ifnet/ifnet.py::LinearWithConstraint.forward` 在 train/eval 下均通过 `.data` 做 max-norm=0.5 的 renorm。仅 `eval()` 或 `no_grad()` 不能保证只读。PyTorch 官方也明确指出 functional call 内部的原地操作会作用于传入的参数/buffer 字典：[PyTorch 2.1 functional_call](https://docs.pytorch.org/docs/2.1/generated/torch.func.functional_call.html)。

实现新实验局部使用的纯函数 IFNet 视图：stem/分类器形状与 state key 不变，分类器计算用 `F.linear`，约束显式执行于独立参数副本。不要让原生 `.data` 写入真实参数或带图的 theta_prime。

临时训练先复制当前参数、全部 buffers 和每参数 optimizer 状态。显式投影训练起点分类头，再对这个起点 `detach().clone().requires_grad_(True)`；梯度在投影后的自由参数上求取，AdamW 也从投影后的起点更新，以复现原生 forward→optimizer 的数值语义。训练视图用普通Linear，不重复投影。反馈 eval 对更新后分类头执行可微 `torch.renorm(p=2,dim=0,maxnorm=0.5)`；其数值要与原生 eval 的约束一致。元梯度需穿过 theta_prime 以及反馈投影，不能在这个阶段 detach/no_grad。不能把更新后投影偷偷换成 `.data` 或 straight-through 并声称是精确元梯度；若最终使用这种近似，必须在配置和报告明确标注。

可微 AdamW 单步读取当前 param-group LR、betas、eps、weight_decay、amsgrad 与各参数 step/m/v，实现偏置校正和 decoupled weight decay。过去状态 detach；A 的当前训练梯度用 `torch.autograd.grad(..., create_graph=True)`，保留权重到 theta_prime 的图。原生 `optimizer.step()` 不可作为 A 的可微步骤。RL 采用同一单步定义的无外层梯度版本。不把虚拟AdamW默默替换为SGD；若必须使用代理optimizer，另建明确标记的协议，A/B共用代理，重跑本轮条件并报告差异。

虚拟训练 forward 使用 train mode/dropout 和自己的 BN buffers；feedback 使用该分支更新后的 BN buffers，以 eval mode 计算 CE。BN running statistics 按原生行为更新且不参与元梯度，记录这一状态语义。真实模型的所有 buffers、optimizer/scheduler 和 `.grad` 均不接受虚拟分支写回。

每个真实 batch 起点保存 Student 训练 RNG。A 临时训练、B 的 CE/action 临时分支及真实训练都使用这个起点；附加评估/策略更新后恢复它，再执行真实更新。真实训练完成后保留它实际推进的 RNG。动作采样使用独立 generator，例如 `torch.bernoulli(q.detach(), generator=action_generator)`，log-prob 另从保留计算图的 q 计算。

RL 两分支必须在更新前从相同参数/buffer/optimizer/RNG 各自复制；不能先走 CE 分支再从其状态构造 action 分支。虚拟 step 不调用 scheduler；真实 scheduler 仅在真实 epoch 末推进。

## 7. 两条路线的每 batch 操作顺序

路线 A：

```text
u = detached_state_from_cache_and_read_only_current_student()
w_old = controller(u)
virtual = isolated_adamw_step(CE + 0.5 * sum(w_old * KD) / B,
                             create_graph=True, student_rng=batch_start_rng)
V = feedback_CE(virtual, scheduled_feedback_batch)
controller_optimizer.step(grad_phi(V))
w_used = controller(u).detach()  # 同一更新前 Student 状态输入
restore_student_rng(batch_start_rng)
real_student_step(CE + 0.5 * sum(w_used * KD) / B)
save_replay(actual_train_UID_order, w_used)
```

路线 B：

```text
u = detached_state_from_cache_and_read_only_current_student()
q = controller(u)
b_previous = reward_baseline.detach()  # 动作采样前的过去奖励基线
a = sample_with_independent_generator(q.detach())
ce_branch = isolated_adamw_step(CE, student_rng=batch_start_rng)
action_branch = isolated_adamw_step(CE + 0.5 * sum(a * KD) / B,
                                  student_rng=batch_start_rng)
r = (feedback_CE(ce_branch) - feedback_CE(action_branch)).detach()
policy_loss = -(r - b_previous) * Bernoulli(q).log_prob(a).sum()
controller_optimizer.step(grad_phi(policy_loss))
reward_baseline = 0.9 * b_previous + 0.1 * r
restore_student_rng(batch_start_rng)
real_student_step(CE + 0.5 * sum(a * KD) / B)  # 同一个 a，不重新采样
save_replay(actual_train_UID_order, q, a, r, b_previous)
```

CE 始终是全 batch 的均值。a 全零合法，仍完成一次 CE 更新；奖励应在数值容差内为零。reward/EMA detach，q/log-prob 保留控制器梯度。每步奖励属于整批动作，不记成逐 trial 的因果收益。

BASE_CE 和 DELAYED_KD_ALL 也从 warm-up 分叉，后者使用 v=1。A/B 训练完成后各自运行一个 shuffle：读取每步真正使用的权重/动作，以 UID 校验并恢复原 batch 顺序，再用独立 RNG 在 batch 内置换；自己的 Student 按同一 schedule 更新。shuffle 无学习控制器、无反馈查询，并不重算来自当前 shuffle Student 的策略。

每步保存 UID 排序版 replay 和顺序恢复信息；断言置换前后连续权重多重集合完全相同、sum 相同、二值动作 keep count 相同，各 UID 的训练出现次数相同。一个 batch 内跨源被试/类别置换是本轮定义；班内单条尾 batch 也合法。

## 8. 分阶段执行与放行条件

| 阶段 | 具体工作与交付 | 进入下一阶段的门槛 |
|---|---|---|
| P0 preflight | 解析所有配置、源文件/预训练 hash、九折 manifest、环境与 Git 状态；验证输出路径 | 九折 7/1/1 无交叠，计数/类别/UID 完整；配置可冻结 |
| P1 算法验证 | 完成控制器、纯函数视图和 AdamW；新增针对性测试 | 下面列出的梯度/隔离/数值检查通过 |
| P2 单折 smoke | target index0，仅 D_train/D_feedback；Teacher 全10 epoch、Student 全10 warm-up，再六条件续训3 epoch | warm-up/切换/feedback/replay/shuffle/resume 全链路通过；不查看目标成绩 |
| P3 七源 Teacher | 九折 Teacher 从头训练并导出 D_train | 九个 checkpoint/cache 的 provenance 和 UID 验证通过 |
| P4 warm-up + controls | 九个共享 warm-up；BASE_CE / DELAYED_KD_ALL 各九折完成到100 | 各条件同一 warm-up/schedule hash，最终 checkpoint 完整 |
| P5 learned | ADAPTIVE_WEIGHT_KD / RL_GATE_KD 各九折完成到100 | 控制器/日志/replay 完整，有限梯度与状态核验通过 |
| P6 shuffle | 完整来源 replay 冻结后运行两个 shuffle 各九折到100 | 逐步匹配源 replay、多重集合与训练出现次数 |
| P7 evaluate/report | 配置冻结且六组全部训练完整后，仅对最终 Student 推理目标；汇总54条结果 | 54/54，每组9fold，各目标 UID 与 manifest 一致，配对和报告齐全 |

P2 使用独立 `smoke/<id>` 输出，scheduler 仍为 T_max100；smoke 的 epoch13 不记正式指标，也不进行 D_test 推理。小 toy 测试可先用缩短 warm-up 检查边界，但不能代替真实10→11切换。通过门槛后正式任务从正式命名空间的新初始化运行，不导入 toy/smoke 状态。

正式调度以完整 fold 为任务，一块空闲 GPU 同时一个工作进程，条件在该 fold 内串行；同一输出 cell 加锁，禁止不同进程同时写同一 cache/checkpoint。并行数由 smoke 的显存峰值和实时 GPU 占用决定，不抢占或停止已有进程。

实现、smoke、正式启动分别提交相应代码/文档；按 AGENTS.md 只 stage 本次相关源码/YAML/文档，提交后及时 push `origin/master` 并核验远端 commit。训练产物保持外置。

## 9. 必须实现的检查

| 检查 | 验收方式 |
|---|---|
| 训练隔离 | Teacher/Student普通训练loss与可学习预处理仅使用D_train；D_feedback仅进入临时分支的反馈CE指导控制器；D_test不进入任何训练/反馈loss或拟合过程；训练UID恰等于七源 |
| 缓存一致 | UID 唯一、重排后 UID/y 完全一致、tap/shape/finite/hash 正确；错误 split cache 拒绝载入 |
| 状态隔离 | 附加状态计算/虚拟反馈前后真实参数、buffers、grad、optimizer、scheduler、模式、RNG 逐项完全相同 |
| 可微 AdamW | 与相同状态的原生 AdamW 对照参数、m/v、step；覆盖零状态、warm-up后的非零状态、当前LR、max-norm约束内外 |
| 元梯度 | FP64 deterministic toy 检查外层梯度/有限差分；实际 IFNet 检查 MLP 梯度 finite且非零、能更新参数；不要求单batch反馈改善 |
| 策略梯度 | 固定 mask/非零 advantage，对照解析 score-function 梯度；梯度到达MLP，reward/EMA没有梯度 |
| CE / KD | v=1 等于既有batchmean KD；v=0等于CE；改变mask不改变CE本身；尾batch分母正确 |
| RL 分支 | 相同训练dropout/状态；全零a使CE/action结果一致，reward约等于0；同一a进入真实更新 |
| shuffle | 对齐来源step/UID、逐步多重集合完全一致、keep count与出现次数相同 |
| 可恢复性 | 连续运行和在step边界checkpoint后恢复，后续schedule/mask/权重/EMA和参数一致；日志不会重复或漏step |
| 目标使用 | train/feedback阶段不调用目标评测入口；epoch100齐全后才执行evaluate |

隔离/UID/多重集合检查要求 exact，不以浮点容差掩盖写回。数值对照预设 FP32 CPU atol=1e-7/rtol=1e-6、CUDA atol=1e-6/rtol=1e-5；FP64元梯度有限差分 atol=1e-6/rtol=1e-3，避开投影不可导拐点；optimizer step计数 exact。实际误差写入检查报告，失败先定位原因，不临时放宽阈值继续正式运行。

checkpoint 至少保存 model/optimizer/scheduler、controller及其optimizer、EMA、训练/动作/shuffle RNG、train/feedback/replay cursor、completed_epoch/next_batch_index、warm-up/cache/config/code hash。逐step replay分epoch原子保存；checkpoint记录已提交日志偏移或块hash，恢复时检查/截断未提交尾部。正式skip需验证完整身份及100epoch，存在一个文件不算完成。

## 10. 拟定 CLI 与运行顺序

runner 已支持 `--stage`、`--folds`（0..8）、`--conditions`、`--gpu`、`--resume`、`--smoke-only`、`--smoke-id`、`--assert-complete`；配置冲突时停止，不覆盖旧实验。

```bash
SAMPLE_UTILITY_CFG=configs/experiments/sample_utility_adaptive_rl_loso_pilot.yaml
SAMPLE_UTILITY_RUNNER=experiments/distill/run_sample_utility_adaptive_rl_loso.py

# 完成源码接入后，先执行配置/划分检查和针对性测试
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage preflight
conda run --no-capture-output -n mirepnet python -m pytest tests/test_sample_utility_adaptive_rl_loso.py -q

# 挑选空闲 GPU；启动前按占用修改。smoke-id 自动由配置 hash 和 commit 生成
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --smoke-only --folds 0 --gpu 0

# smoke 放行后，正式九折；每个 stage 成功才进入下一个
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage teacher --gpu 0 --resume
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage warmup --gpu 0 --resume
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage controls --gpu 0 --resume
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage learned --gpu 0 --resume
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage shuffle --gpu 0 --resume
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage evaluate --gpu 0 --assert-complete
conda run --no-capture-output -n mirepnet python "$SAMPLE_UTILITY_RUNNER" --config "$SAMPLE_UTILITY_CFG" --stage report --assert-complete
```

多GPU时对同一个stage分配不同 `--folds`，每折仍按依赖顺序运行，全部九折通过完整性检查后汇总。`evaluate` 要验证六条件54个最终 checkpoint；`report` 要验证54条目标结果。外部权限不足或资源不足时保持明确失败状态，不能降级为假训练或用历史数字填表。

## 11. 预算、日志与最终报告

54 条 Student 条件各具有100epoch历史，但 warm-up计算共享：实际新训练为 `9×10 + 54×90 = 4950` 个 Student epoch，加 `9×10=90` 个 Teacher epoch。按当前计数、batch16且每batch控制器更新：

| 项目 | 九折合计 |
|---|---:|
| 一种条件后90epoch真实Student step | 55,440 |
| 六条件续训真实step | 332,640 |
| 九折共享warm-up真实step | 6,160 |
| A 临时optimizer step / feedback batch查询 | 55,440 / 55,440 |
| B 临时optimizer step / feedback batch查询 | 110,880 / 110,880 |

feedback query 按一次反馈 batch loss计，另记录查询trial数；A包含高阶梯度，B包含两个临时分支，不能由epoch数推断计算公平。成本表为静态推算，正式manifest须重验。先在smoke分别测Teacher、普通/元梯度/RL step时间与显存峰值，再按step数估算墙钟；当前不承诺GPU小时数。

逐step至少保存训练/反馈UID、真实使用的w或q/a、reward、b_previous、feedback loss（B两分支）、controller梯度范数、Student CE、原始KD、加权KD、实际LR、有效KD系数 `0.5*mean(v)`、临时step/反馈query数与耗时。A应分别记录元更新前w与真实使用的新w。逐epoch汇总权重均值/std/分位数、类别和源被试分布、Teacher correct/wrong分布（仅诊断）、RL keep rate/概率/熵、reward均值方差、损失、墙钟和峰值显存。

epoch100固定评测保存目标UID/y/logits/probabilities/predictions、Accuracy/BA/Kappa、混淆矩阵和预测类别数量。collapse预定义为只预测一个类别，并另报告最大预测类别占比。`metrics_by_subject.csv`必须是6×9完整表；各条件总体指标按被试等权平均，不能因S02样本少而采用全trial pooled Accuracy替代主表。

`paired_comparisons.csv` 按九个目标被试配对，保存delta明细、mean delta、Win/Tie/Loss、subject bootstrap 95% CI（10,000次，seed666；BA/Acc单位为百分点，Kappa原单位）。Tie用未四舍五入指标的abs(delta)<=1e-12。五项比较沿用原设计；前两项为主要比较，若报告p值对这两项BA的Wilcoxon使用Holm，后三项及Accuracy/Kappa的p值归为探索性并明确其检验族，不把多个未校正p值用于宣布胜出。

`report.md`包含配置/来源/训练完整性、六组成绩、五项配对、学习组与shuffle关系、反馈与目标差异、权重塌缩/策略噪声诊断以及时间/查询/有效KD强度。CI跨0只报告方向；9fold单seed仅为pilot。继续扩展seed或数据集时另建预先固定的实验规格，不能根据本轮目标成绩回调本轮配置。

## 12. 方法来源

本实验借鉴验证集元梯度样本重加权、元数据训练MLP权重函数，以及Student反馈驱动教学策略的思想；不声称是下面论文的直接复现。

- [Ren et al., ICML 2018](https://proceedings.mlr.press/v80/ren18a.html)：用验证反馈指导样本重加权。
- [Shu et al., NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html)：用元数据训练MLP权重函数。
- [Fan et al., ICLR 2018](https://www.microsoft.com/en-us/research/publication/learning-to-teach/)：用Student反馈优化教学策略。

本轮针对性测试为 8 项；静态 preflight 九折通过；fold 0 smoke 的每组 3 epoch、controller/reward、shuffle UID 多重集合和“无目标评测”检查通过。正式九折训练、最终 epoch100 评测与报告已完成，实际结果和解释见下一节。

## 13. 正式运行记录与结果（2026-10-09）

### 完成范围与产物

- 按本方案使用 BNCI2014004 `session_3`、MIRepNet / IFNet、seed=666、9 个 LOSO target folds。每折 D_train 为 7 个源被试，D_feedback 为固定下一个源被试，D_test 为目标被试；Teacher 与 Student 普通训练不使用 feedback/test，反馈数据仅用于控制器临时分支，目标数据仅在全部训练 checkpoint 完整后评测。
- 9 个七源 Teacher 各微调 10 epoch；每折共享一个 Student 10-epoch warm-up；54 个 Student 条件各完成至 epoch100。所有最终评测均为固定最终 epoch checkpoint，没有基于 D_test 的选模。
- 配置 hash：`95acac7486d2df3bc9025d2d1c41c272c0f9f37f0c477237676c1a0097b94e24`。解析值包括 batch_size=16、lambda_KD=0.5、tau=2、Teacher 10 epoch。每折 split、Teacher、warm-up、checkpoint、预测和 controller provenance 均写入外部结果目录。
- 最终交付：`resolved.yaml`、环境与执行快照、split / UID manifest、Teacher 和 warm-up provenance、54 份最终 Student checkpoint/history/predictions/metrics、学习控制器 checkpoint、逐 epoch replay 与压缩 step metrics、配对表和 `report.md`。训练产物没有写入 Git checkout。

### Shuffle UID 对齐更正

第一次 shuffle 矩阵运行后，完整日志审计发现 replay 数组按 batch 顺序保存，但旧回放路径把数组位置当作 D_train UID 位置读取。因此旧 ADAPTIVE_SHUFFLE / RL_SHUFFLE 会把策略值分配给错误 UID，旧 batch 检查只能验证错位后向量的守恒，不能代表方案要求的 UID 对齐 shuffle。这 18 个旧 shuffle run 及其旧汇总已保存在 `/data1/llx/BigSmallcollab/migrations/sample_utility_adaptive_rl_loso_pilot_v1/shuffle_pre_uid_alignment_fix_20261009/`，明确不用于正式比较。

回放代码现先按 UID 将权重、概率、动作及其他数组重排到 D_train 顺序，再读取当前 batch。新增 `align_replay_by_uid` 回归测试后，8 项测试通过；修正后的 18 个 shuffle 条件从共享 warm-up 重新训练至 epoch100。最终报告再次逐批核对来源日志与 shuffle 日志的 UID 顺序、策略值对齐、batch 内权重 / 动作多重集合及 RL keep count。两种 shuffle 各验证 55,440 个 minibatch；每个 fold-epoch 的 D_train UID 均恰好出现一次。

### 目标被试成绩与配对结果

| 条件 | Mean BA | Mean Accuracy | Mean Kappa |
|---|---:|---:|---:|
| BASE_CE | 0.7375 | 0.7375 | 0.4750 |
| DELAYED_KD_ALL | 0.7384 | 0.7384 | 0.4769 |
| ADAPTIVE_WEIGHT_KD | 0.7363 | 0.7363 | 0.4727 |
| RL_GATE_KD | 0.7331 | 0.7331 | 0.4662 |
| ADAPTIVE_SHUFFLE | 0.7387 | 0.7387 | 0.4773 |
| RL_SHUFFLE | 0.7338 | 0.7338 | 0.4676 |

54/54 目标评测行、每个条件 9 个目标被试；collapse=0/54。主要 BA 配对结果为：

- ADAPTIVE_WEIGHT_KD − DELAYED_KD_ALL：mean delta −0.0021，subject bootstrap 95% CI [−0.0118, +0.0090]，Win/Tie/Loss=2/3/4，Holm p=1.0。
- RL_GATE_KD − DELAYED_KD_ALL：mean delta −0.0053，95% CI [−0.0211, +0.0076]，Win/Tie/Loss=4/1/4，Holm p=1.0。
- ADAPTIVE_WEIGHT_KD − ADAPTIVE_SHUFFLE：−0.0023，95% CI [−0.0139, +0.0081]，Win/Tie/Loss=3/3/3。
- RL_GATE_KD − RL_SHUFFLE：−0.0007，95% CI [−0.0139, +0.0097]，Win/Tie/Loss=4/3/2。

四个区间均覆盖零。这个单 seed pilot 没有显示学习的样本分配优于固定 KD 或对应 shuffle；点估计方向也没有显示学习组更好，因此不宣布 ADAPTIVE_WEIGHT_KD 或 RL_GATE_KD 胜出。前两项 Wilcoxon 的小样本 / ties 触发 SciPy 正态近似警告，Holm 校正后均为 1.0；这不足以说明两种方法等效或原理无效。

### 控制器与计算诊断

- ADAPTIVE_WEIGHT_KD：平均权重 0.5432、epoch100 均值 0.5489；平均有效 `lambda_KD * mean(w)` 为 0.2716，epoch100 为 0.2744。记录 55,440 个临时 Student step 和 55,440 个 feedback batch 查询（880,744 个查询 trial）。feedback CE 的 fold-epoch 均值从 epoch11 的 0.5506 到 epoch100 的 0.5140。
- RL_GATE_KD：平均 keep rate 0.5844、epoch100 为 0.6023；平均有效 KD 系数为 0.2922，epoch100 为 0.3012。记录 110,880 个临时 Student step、110,880 个 feedback batch 查询（1,761,488 个查询 trial）。epoch 平均 reward mean 为 +0.000222，批奖励 epoch 内 std 的 epoch 平均值为 0.000987；feedback CE 均值从 0.5616 到 0.5199。
- 上述反馈 CE 下降没有转化为目标被试上的优势。RL 平均奖励接近零，且报告保留 reward 方差；该现象应视为本轮策略反馈信号有限，不能外推为 RL 方法普遍无效。
- 记录 manifest 时长总和：BASE_CE 1.60h、DELAYED_KD_ALL 1.60h、ADAPTIVE_WEIGHT_KD 6.83h、RL_GATE_KD 5.86h；shuffle 缺少历史 manifest 时长字段，报告用 epoch11 replay 到最终 manifest 的文件时间跨度近似为 ADAPTIVE_SHUFFLE 1.65h、RL_SHUFFLE 1.68h。计时不包含恢复运行间等待，A/B 有不同数量的临时分支，不能仅按 epoch 数声称计算公平。
- 数据预训练暴露情况仍未知；不能据本轮证明 MIRepNet 预训练从未接触 BNCI2014004。反馈源被试参与控制器训练，所以这是“反馈源被试监督的控制器能否泛化到未见目标被试”实验，不是 feedback-independent 的评价器验证。

最终机器可读结果与逐被试指标见 `/data1/llx/BigSmallcollab/results/distill/sample_utility_adaptive_rl_loso_pilot_v1/report.md`、`paired_comparisons.csv` 和 `metrics_by_subject.csv`。主报告的显著性、样本分配和 pilot 限制按第 11 节的预设规则解释。

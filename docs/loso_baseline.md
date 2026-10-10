# 当前 LOSO baseline

新数据源上已经完成的配置是当前 baseline。训练与预处理参数直接读取
`configs/models/<model>.yaml` 的 `finetune.<dataset>.loso`；来源、场次、
类别映射和 baseline 位置读取 `configs/datasets/<dataset>.yaml` 的 `loso`。
004/5001 协议入口为 `configs/protocols/loso.yaml`，001/001-4 为
`configs/protocols/loso_001.yaml`。旧参数迁移和数据对照配方已退役。

| 任务 | 共享 NPY 来源 | 所选场次 | seeds |
|---|---|---|---|
| 001 / 001-4 | `/data1/llx/BNCI2014001/broadband_0p1_75hz` | `0train`（原 `session_T`） | 0、1、2 |
| 004 | `/data1/llx/BNCI2014004/broadband_0_120hz` | `session_3`（MOABB `3test`） | 666、667、668 |
| 5001 | `/data1/llx/BNCI2015001/broadband_0p1_75hz` | `session_A`（MOABB `0A`） | 666、667、668 |

每折留出整位被试，其余被试的全部所选 trials 用于训练；固定最后一轮
评估一次。001 是左右手二分类，001-4 为四分类。模型窗口为 4 秒；
MIRepNet 按被试 EA，测试被试使用全部无标签 trial 协方差，记录为
transductive LOSO。

001/001-4 的五个模型均使用 seeds **0、1、2**。教师沿有序被试折
保持每个 seed 的随机数流，学生在每折重新设置同名 seed；种子编号相同，
随机数使用方式不同，不能据此宣称模型之间的初始化或批次随机过程相同。

004/5001 当前训练参数如下；正式数值以模型 YAML 为准。

| 模型 | Optimizer / LR | Batch / WD | Epochs | 模型输入处理 |
|---|---|---|---:|---|
| MIRepNet | Adam / 1e-3 | 8 / 1e-6 | 10 | 8–30 Hz → 逐被试 EA → 45 通道 |
| CBraMod | AdamW / 1e-4 | 64 / 0.05 | 50 | 200 Hz、0.3–75 Hz、60 Hz notch、无 EA/CAR |
| IFNet | AdamW / 1e-3 | 16 / 0.01 | 100 | 内部 4–16 / 16–40 Hz filterbank |
| EEGNet | AdamW / 1e-3 | 32 / 1e-4 | 100 | 8–32 Hz |
| ADFCNN | AdamW / 1e-3 | 32 / 1e-4 | 100 | 8–32 Hz |

CBraMod 004/5001 的 head dropout=0.1、label smoothing=0.1，逐 epoch
cosine，warmup=0。001/001-4 使用已完成的宽带输入配方；其当前参数也
直接写入模型 YAML。

004/5001 baseline 结果位于
`/data1/llx/BigSmallcollab/results/reproductions/loso_source_refresh_004_5001_v1/`。
当前四组蒸馏读取对应教师与学生基线，规格为
`configs/experiments/loso_distillation.yaml`，三个学生和四组方法共用
126 份教师目标。配置路径整理保持已有训练与缓存身份兼容。

供查看的 Excel、CSV 和 JSON 汇总报告保存在项目真实的 `results/` 目录，
通过 `experiments.storage.require_report_output` 导出。本轮报告为
`results/loso_source_refresh_004_5001/loso_distillation_accuracy_3seed.xlsx`。

输入缓存按来源 manifest、所选场次、类别、窗口和预处理方式核对，修改
LR 或训练轮数可以继续复用相同输入。当前入口会明确拒绝未支持的频率、
EA、CAR 或调度改动，避免 YAML 与实际处理不一致。已完成结果和中断
状态按实际参数比较，新增默认值说明不会使已有产物失效。

旧临时配置已经从活动配置目录删除。不可变的原始字节只存于
`/data1/llx/BigSmallcollab/migrations/retired_loso_configs/`，供历史
manifest 和断点追溯；当前 baseline 选择由正式模型、数据集和协议配置
决定。历史外部产物目录名继续作为已有结果的身份标识。

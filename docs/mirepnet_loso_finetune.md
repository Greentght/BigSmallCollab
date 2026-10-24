# MIRepNet 独立 LOSO 微调

001（二分类）或 001-4（四分类）使用独立入口。每次执行默认生成新的实验目录，
从官方预训练权重重新微调所有被试折；三个 seed 在一张已分配的 GPU 上顺序运行。

```bash
conda run --no-capture-output -n mirepnet \
  python experiments/finetune/run_mirepnet_loso.py \
  --dataset 001 --seeds 666 667 668 --gpu 9
```

GPU 号替换为已经分配给本实验的非 0 号卡。此入口直接训练，不负责等待空闲 GPU。
每个进程的数值计算线程和 PyTorch intra/inter-op 线程数为 1，DataLoader workers 为 0。

训练参数默认读取 `configs/models/mirepnet.yaml` 的 `finetune.<dataset>.loso`。
可用 `--epochs`、`--lr`、`--batch-size`、`--weight-decay` 和 `--optimizer` 覆盖：

```bash
conda run --no-capture-output -n mirepnet \
  python experiments/finetune/run_mirepnet_loso.py \
  --dataset 001 --seeds 666 667 668 --gpu 9 \
  --epochs 10 --lr 0.001 --batch-size 8 --weight-decay 0.000001
```

训练产物在 `/data1/llx/BigSmallcollab/results/finetune/mirepnet/<dataset>/<run-name>/`。
默认 run-name 为时间加随机后缀；也可以用 `--run-name mi001_trial01` 指定新名称。
指定的目录已存在时，新实验会报错。中断后显式恢复同一实验：

```bash
conda run --no-capture-output -n mirepnet \
  python experiments/finetune/run_mirepnet_loso.py \
  --dataset 001 --run-name mi001_trial01 --resume --gpu 9
```

每次新实验保存 YAML、协议和实际参数快照。续跑读取保存的快照，不受后来编辑 YAML
影响。恢复时不能修改训练参数或 seed 集合；新参数使用新实验目录。

入口核验当前 canonical EEG 输入的来源、处理配方和文件哈希，再读取共享输入缓存。
当前输入处理为 YAML 中的 8–30 Hz、逐被试 EA、45 通道映射，属于 transductive LOSO；
改变滤波频段等预处理需要另行准备匹配的输入缓存，本入口未提供频段覆盖选项。
它保持现有 001 协议的单 seed 有序 LOSO 随机数流和最终 epoch 评估。

全部完成后，Accuracy (%) 按每个 seed 的 9 个被试等权平均，再对 seed 等权平均，
导出到项目 `results/mirepnet_<dataset>_<run-name>.json`。不会自动修改已有 baseline log。

# REPRO.md — 全谱系重跑手册

目标:在新树上按固定种子完整重跑 pre-D0 全部实验(10 个家族),与历史
`results/metrics/*.csv` 对比,验证可复现性。本手册是权威记录;`scripts/legacy/run_*.sh`
是各家族的**规范命令来源**(超参与当时收口实验一致),REPRO 阶段以它们为准。

---

## 0. 数据流总览

```text
/data1/llx/<ds>/{X,labels}.npy  (DATA_ROOT 可覆盖)
  └─ data/eeg_dataset.py   EEGDataset:按数据集规则选 session/截断/重采样/过滤类别
      └─ data/split.py     确定性切分:subject_split(seeded 666/667/668)/ loso_split
          │
          ├─ 大模型线(各自 conda env: cbramod / labram / mirepnet)
          │   models/<family>/adapter[_native].py: preprocess → build(weights/*.pth) → forward
          │   experiments/bigmodel/*_adapt.py / tune_*.py: 复现、调参、协议消融
          │   └─ scripts/export/*.py(export_preds / finetune_export / export_teacher_loso / _mc)
          │        └─ 工件 results/artifacts/<ds>/<model>[|_loso]/<key>_<seed>_<split>.npz
          │           {logits(N,C) f32, feats(N,D) f32, y(N,) i64}   ← collab/artifacts.py 契约
          │
          └─ 学生/蒸馏线(env: mirepnet,不加载大模型,只读工件)
              experiments/run.py + configs/exp/*.yaml
              experiments/distill/run_distill.py (KD/MMD/Combo/mask/dkd/eakd/adaptive/pearson)
              experiments/bidir/ + collab/{bidirectional,mutual,bdeeg}.py (双向线)
              └─ collab/distill.py distill_student: CE + KD + feature-align (+ 各变体)
                   └─ eval/metrics.py evaluate → acc% / kappa
                        └─ results/metrics[|_repro]/<name>.csv (long-form)

统计:  python -m eval '<glob>'        → eval/stats.py: 种子先平均 → subject/fold 配对
                                        Wilcoxon + Holm + bootstrap CI
对比:  tools/compare_repro.py         → 历史 vs 重跑(deterministic / ci 两档)
汇总:  tools/make_summary_xlsx.py     → results/summary_*.xlsx
```

要点:
- **跨 env 解耦全靠工件 hub**:大模型只在自己的 conda env 里训练并导出
  logits/feats/y;学生与协同实验在 mirepnet env 消费工件,**从不加载大模型**。
  `load_aligned` 会断言 y 逐行对齐(切分一致性护栏)。
- **没有 checkpoint 落盘**:模型每次从 `weights/*.pth` 重建 + finetune,只落工件和 CSV。
- 工件命名:within = `<model>/<subject>_<seed>_<split>.npz`;
  loso = `<model>_loso/<fold>_<seed>_<split>.npz`;特殊键 `mirepnet_loso`、
  `mirepnet_loso_subjoof`、别名 `cbramod_native`——改名会破坏所有消费者。

---

## 1. 环境与约定

### conda env(每模型独立,见 configs/models/*.yaml 的 `env:`)

| env | 用途 |
|---|---|
| `mirepnet` | MIRepNet + 全部小模型(IFNet/EEGNet/ADFCNN)+ 所有蒸馏/协同实验 |
| `cbramod` | CBraMod native 适配与调参、CBraMod 教师导出 |
| `labram` | LaBraM 复现/调参 |

### 运行约定

- **种子**:`666/667/668`(configs/datasets/*.yaml 默认);✅ 家族(蒸馏各线)期望
  与历史 **bit-identical**;❌ 家族(大模型调参、bidir/CR-AMD/BD-EEG、LOSO 大模型)
  期望 **CI 内**(见 §14)
- **线程**:每进程 OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4(入口已 setdefault);总量 ≲40 核
- **GPU**:单卡任务固定 `CUDA_VISIBLE_DEVICES=2`(或 `--gpu 2`);并行任务 2,3,4,5;
  多卡进程**不要**在播种时 `manual_seed_all` 触碰别的卡(07-28 OOM 教训)
- **脱离**:长任务一律 `setsid nice -n 19 conda run -n <env> python ... > logs/repro_<family>/<task>.log 2>&1 < /dev/null &`
- **数据根**:`DATA_ROOT`(默认 `/data1/llx`);**工件根**:`ARTIFACT_ROOT`(默认 `results/artifacts`)
- **重跑落盘**:`REPRO_OUT=results/metrics_repro`(历史 `results/metrics/` 只读不改);
  旧工件先 `mv results/artifacts results/artifacts_v0`(LOSO 教师可复用则 mv 回,见 §14 风险表)

## 2. 基线校验(放全量前的闸门)

先跑 1 dataset 的 base 条件,与历史 CSV 该行比对 bit-identical,再放全量:

```bash
REPRO_OUT=results/metrics_repro \
setsid nice -n 19 conda run -n mirepnet python -m experiments.run \
  configs/exp/distill_kd_within.yaml --gpu 2 \
  > logs/repro_f1/smoke.log 2>&1 < /dev/null &
python tools/compare_repro.py --hist 'results/metrics/distill_kd_within.csv' \
  --repro 'results/metrics_repro/distill_kd_within.csv' --mode deterministic
```

---

## 3-12. 家族重跑矩阵

| # | 家族 | 规范入口(以 run_*.sh 为准) | env | 历史 CSV 锚点(results/metrics/) |
|---|---|---|---|---|
| 1 | 基础蒸馏/MMD/COMBO | `python -m experiments.run configs/exp/distill_kd_within.yaml --gpu 2`(datasets 字段扩到 5 数据集) | mirepnet | `distill_kd_within.csv`;`<ds>_distill_mirepnet_to_ifnet.csv` |
| 2 | few-shot/K-shot | `experiments/distill/run_distill.py --fewshot --shots ...` — 历史 `_fs{K}` 参数从 PROGRESS 06-27 小节定位后固化到 REPRO | mirepnet | `*_fewshot*`(含 shots 列) |
| 3 | EEGNet/ADFCNN 学生 | `experiments/distill/run_distill.py --dataset <ds> --teacher mirepnet --student {eegnet,adfcnn} --gpu 2` | mirepnet | `<ds>_distill_mirepnet_to_{eegnet,adfcnn}.csv` |
| 4 | CBraMod 复现/调参/消融 | `scripts/legacy/run_cbramod_native_paper5.sh native70`、`tune_cbramod_native.py --phase all --gpus 2 3 5`、`tune_cbramod_004_caronly.py`;**★ LOSO 收尾**:`tune_cbramod_loso.py --phase all --gpus 1 8 --threads 4`(07-29 未完成,断点续跑) | cbramod | `results/cbramod_native/*`、`results/mirepnet_loso/tuned/summary_tuned.csv`(已有基线) |
| 5 | CBraMod 教师蒸馏 | `scripts/legacy/run_adaptive_entropy_kd.sh` 前半(export_teacher_mc cbramod_native)→ `run_distill.py --teacher cbramod_native --student ifnet` | cbramod→mirepnet | `<ds>_distill_cbramodnative_to_ifnet.csv` |
| 6 | LaBraM 复现/调参 | `scripts/legacy/run_labram_native_paper5.sh 2`、`tune_labram_native.py --phase all --gpus 2 3 5` | labram | `results/labram_native/*` |
| 7 | Mask/Conf/Adaptive/EA-KD/DKD | `run_mask_ablation*.sh`、`run_dkd_ablation.sh`、`run_adaptive_entropy_kd.sh`、`run_eakd_within.sh`、`run_eakd_combo_matrix.sh`、`run_relational_ablation.sh` | mirepnet | `*_{maskablation,dkd,adaptivekd,eakd,eacombo}_*.csv` |
| 8 | LOSO 跨被试 | `run_loso_full.sh`(export_teacher_loso → run_loso_distill)+ `run_loso_ext_0014/004.sh`;subject-OOF: `export_teacher_loso_subjoof.py` → `run_loso_subject_oof_kd.py` | mirepnet | `*_loso*.csv` |
| 9 | 双向/CR-AMD/BD-EEG | `run_bidir_full.sh`、`run_cramd_full.sh`、`run_bdeeg_parallel.sh`(07-25 收口两实验 = CR-AMD + BD-EEG) | mirepnet | `*_loso_{bidir,cramd,bdeeg}_*.csv` |
| 10 | Pearson | `run_distill.py --fewshot_pearson --shots 5 10 20 --teacher {mirepnet,cbramod_native}`(仅 4 类 001-4)+ `experiments/distill/analyze_fewshot_pearson.py` | mirepnet | `BNCI2014001-4_fewshot_pearson_*_to_ifnet.csv` |

执行顺序与依赖:族 1 base → 族 3 学生 base → 导出教师工件(MIRepNet within / CBraMod
within / mirepnet_loso)→ 族 7(依赖 within 工件)→ 族 2/10 few-shot → 族 5/6 教师
→ 族 8 LOSO → 族 9 最后。族 4 的 LOSO 收尾最先跑(耗时最长)。

## 13. 结果对比与验收

```bash
# ✅ 确定性家族:逐行 bit-identical
python tools/compare_repro.py --hist 'results/metrics/<fam>_*.csv' \
  --repro 'results/metrics_repro/<fam>_*.csv' --mode deterministic

# ❌ 审计家族:逐 subject 配对差在 bootstrap CI 内(|mean|<=0.5pp 也放行)
python tools/compare_repro.py --hist 'results/metrics/<fam>_*.csv' \
  --repro 'results/metrics_repro/<fam>_*.csv' --mode ci --tol 0.5
```

汇总: `python tools/make_summary_xlsx.py --hist 'results/metrics/*.csv' --repro 'results/metrics_repro/*.csv'`
统计: `python -m eval 'results/metrics_repro/<fam>_*.csv'`(基线自动 = `*_base`)

偏差清单 → `docs/repro_report.md`(结构:环境指纹 / bit-identical 表 / CI 内表 /
偏差原因+处置 / 结论)。

## 14. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| LOSO 教师工件重建开销大 | `results/artifacts_v0/` 中 `*_loso/` 完整则 mv 回复用,只补缺 |
| CBraMod LOSO 未收尾 | 断点续跑 `tune_cbramod_loso.py`(runner 按 (ds,fold,seed) 续跑) |
| BD-EEG 9 fold 并行超线程 | `nice -n 19` + 每进程 4 线程 + 总核 ≲40 |
| ❌ 家族非 bit-identical 误判 | compare_repro 分 deterministic/ci 两档 |
| 重跑覆盖历史结果 | REPRO_OUT 隔离;artifacts 先 mv artifacts_v0 |

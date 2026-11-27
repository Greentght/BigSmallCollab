# HFD position and DML-DS ablations

The study uses 30% calibration and 70% test trials, seeds 666, 667 and 668, on BNCI2014004 and BNCI2014001. It covers CBraMod–EEGNet, MIRepNet–ADFCNN and NeuroGPT–IFNet. C1 is independent CE plus the two existing deep-supervision heads. C0 is DML-DS. C2–C5 use one bidirectional HFD layer pair, and C6 averages the four HFD losses in each direction. The total HFD coefficient is 0.1 in every HFD condition, with five warm-up and five ramp epochs.

| Condition | Large-model HFD start | Small-model HFD start |
| --- | --- | --- |
| C2 | shallow | shallow |
| C3 | shallow | first DS location |
| C4 | first DS location | shallow |
| C5 | first DS location | first DS location |
| C6 | both | both |

The large-model shallow and first-DS locations are CBraMod patch embedding and block 4, MIRepNet embedding and block 2, and NeuroGPT embedding and block 2. The small-model locations are EEGNet batchnorm1 and pooling2, ADFCNN spectral branches and spatial_1, and IFNet sconv and sconv. IFNet's existing shallow point and first DS location coincide. The four conditions are recorded individually even when their mathematical paths coincide.

The HFD target follows the native subsequent operator. For ADFCNN at spatial_1, the native attention also consumes spatial_2, so its captured output is passed as frozen context. That preserves the parallel branch structure and makes the spatial_1 path directly comparable. EEGNet after pooling2 uses the native dropout, separable convolution, batch normalization, ELU and pooling3. The first DS outputs have already mixed the electrode dimension, and the projection handles their dimensions accordingly.

Existing C0 and C2 are reused for CBraMod and MIRepNet after matching trial identities. The old NeuroGPT C0 used per-trial time normalization. The user requires all normalization parameters to come from calibration, so NeuroGPT C0 and C2–C6 are retrained with channel means and standard deviations fitted across calibration trials and time. The same calibration normalization is applied to its independent CE and C1 models. MIRepNet EA is fitted on calibration trials and then applied to test trials. CBraMod CAR is a fixed instantaneous channel reference, with no fitted test-set parameters.

CBraMod and MIRepNet pair conditions use 20 epochs and NeuroGPT uses 100, matching the existing DML-DS budget. Independent CE students additionally run at the other budget. CE20 for CBraMod and MIRepNet is reused from each pair's matched supervised baseline. C1 minus the same-budget CE result is the DS increment, and C0 minus C1 is the increment after adding DML to DS.

Results under `results/hfd_position_fewshot_v1/` include every completed subject, seed, role and condition, subject paired differences averaged over seeds, group mean paired differences, C2–C5 rankings, per-dataset best-position consistency, and independent-student budget comparisons. Each seed standard deviation is computed across the three seed-level subject means. Confidence intervals bootstrap the nine subject-level differences after averaging each subject's three seeds.

Training state is stored under `/data1/llx/BigSmallcollab/results/bidir/hfd_position_fewshot_v1/` and removed on successful completion. GPU workers use one CPU thread, choose a card with fewer than three visible compute processes at dispatch, and stay on that card for the complete training unit. The position queue runs one four-path calibration preflight per dataset and pair before formal training.

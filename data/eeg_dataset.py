import os
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
import torch
from mne.filter import resample
from scipy.spatial.distance import cdist
import pandas as pd
from collections import Counter
from data.channels import *

class EEGDataset(Dataset):
    def __init__(self, args=None):
        self.dataset_name = args.dataset_name
        self.args = args

        # 根据 dataset_name 决定加载哪个 npy
        if self.dataset_name == 'BNCI2014001-4':
            X = np.load('/data1/llx/BNCI2014001/X.npy')
            y = np.load('/data1/llx/BNCI2014001/labels.npy', allow_pickle=True)
        else:
            X = np.load('/data1/llx/' + self.dataset_name + '/X.npy')
            y = np.load('/data1/llx/' + self.dataset_name + '/labels.npy', allow_pickle=True)
        print("original data shape:", X.shape, "labels shape:", y.shape)

        if self.dataset_name == 'BNCI2014001-4':
            # -------- BNCI2014001 四分类 (feet / left_hand / right_hand / tongue) --------
            # 9 subjects × 576 trials, 22 channels, 1001 time points
            # 0-287: Session T, 288-575: Session E
            self.paradigm = 'MI'
            self.num_subjects = 9
            self.sample_rate = 250
            self.ch_num = 22

            data_mode = getattr(self.args, 'data_mode', 'finetune')
            indices = []
            target_subjects = self.args.sub if hasattr(self.args, 'sub') else range(self.num_subjects)

            for i in target_subjects:
                base_idx = i * 576
                if data_mode == 'phase1':
                    print(f"Loading Session T (train) for Subject {i}")
                    indices.append(np.arange(288) + base_idx)
                elif data_mode == 'session3':
                    print(f"Loading Session E (eval) for Subject {i}")
                    indices.append(np.arange(288) + base_idx + 288)

            if len(indices) > 0:
                indices = np.concatenate(indices, axis=0)
                X = X[indices]
                y = y[indices]
                X = X[:, :, :1000]  # 截断到4秒，与004一致，且能被IFNet patch_size=125整除
                # 不做类别筛选，保留全部 4 类
            else:
                print(f"Warning: No indices loaded for BNCI2014001-4 in mode {data_mode}")

        elif self.dataset_name == 'BNCI2014004':
            self.paradigm = 'MI'
            self.num_subjects = 9
            self.sample_rate = 250
            
            # (Start, End_Sess2, End_Sess3, End_Total)
            # S0-S2: 用于 Phase 1 (Transfer/Warmup)
            # S3:    用于 Phase 2 (Finetune) 和 Test
            split_points = {
                0: (0, 400, 560, 720),
                1: (720, 1120, 1240, 1400),
                2: (1400, 1800, 1960, 2120),
                3: (2120, 2540, 2700, 2860),
                4: (2860, 3280, 3440, 3600),
                5: (3600, 4000, 4160, 4320),
                6: (4320, 4720, 4880, 5040),
                7: (5040, 5480, 5640, 5800),
                8: (5800, 6200, 6360, 6520)
            }

            indices = []
            data_mode = getattr(self.args, 'data_mode', 'finetune')
            
            target_subjects = self.args.sub if hasattr(self.args, 'sub') else range(self.num_subjects)

            for subject_id in target_subjects:
                p0, p1, p2, p3 = split_points[subject_id]
                
                if data_mode == 'phase1':
                    # 加载 Session 0, 1, 2 (400条)
                    print(f"Loading Phase 1 Data (Sess 0-2) for Subject {subject_id}")
                    indices.append(np.arange(p0, p1))
                    
                elif data_mode == 'session3':
                    # 加载完整的 Session 3 (160条或120条)
                    print(f"Loading Session 3 (Full) for Subject {subject_id}")
                    indices.append(np.arange(p1, p2))

            if len(indices) > 0:
                indices = np.concatenate(indices, axis=0)
                X = X[indices]
                y = y[indices]
                X = X[:, :, :1000]
            else:
                print(f"Warning: No indices loaded for Subject {target_subjects} in mode {data_mode}!")
        elif self.dataset_name == 'BNCI2014001':
            self.paradigm = 'MI'
            self.num_subjects = 9
            self.sample_rate = 250
            self.ch_num = 22

            # 获取数据加载模式 (由 utils.py 传入)
            # phase1:   对应 Session T (训练 session)
            # session3: 对应 Session E (评估 session) -- 为了兼容 utils.py 的变量名，这里沿用 session3 这个 key
            data_mode = getattr(self.args, 'data_mode', 'finetune')
            
            indices = []
            target_subjects = self.args.sub if hasattr(self.args, 'sub') else range(self.num_subjects)

            for i in target_subjects:
                # BNCI2014001 规律：每个被试 576 条
                # 0-287: Session T
                # 288-575: Session E
                base_idx = i * 576 
                
                if data_mode == 'phase1':
                    # 加载 Session T (作为预热历史数据)
                    print(f"Loading Phase 1 Data (Session T) for Subject {i}")
                    indices.append(np.arange(288) + base_idx)
                    
                elif data_mode == 'session3': 
                    # 注意：utils.py 里写的 mode 是 'session3'，这里我们要把它映射到 Session E
                    # 加载完整的 Session E (将在 utils.py 中被切分为 80%微调 / 20%测试)
                    print(f"Loading Target Data (Session E) for Subject {i}")
                    indices.append(np.arange(288) + base_idx + 288)
                
                # 如果有 'test' 模式遗留，也可以映射到 Session E 的后半段，但根据新策略暂时用不到
                elif data_mode == 'test':
                     pass 

            if len(indices) > 0:
                indices = np.concatenate(indices, axis=0)
                X = X[indices]
                y = y[indices]

                # --- 关键：类别筛选 ---
                # 001 数据集有 4 类 (Left, Right, Foot, Tongue)
                # 我们只取 Left(1) 和 Right(2) 做二分类
                # 假设 label 编码对应: 0:Left, 1:Right, 2:Foot, 3:Tongue (具体需视 labels.npy 内容而定)
                # 这里沿用你之前的字符串筛选逻辑，或者基于 LabelEncoder 后的数字筛选
                
                # 假设原始 labels 是字符串数组
                keep_indices = []
                for k in range(len(y)):
                    if y[k] in ['left_hand', 'right_hand']:
                        keep_indices.append(k)
                
                X = X[keep_indices]
                y = y[keep_indices]
                X = X[:, :, :1000]  # 截断到4秒，与004一致

            else:
                print(f"Warning: No indices loaded for 001 in mode {data_mode}")
        elif self.dataset_name == 'AlexMI':
            # AlexMI as a 2-class task (right_hand vs feet; 'rest' is DROPPED).
            # 512 Hz -> 250 Hz via mne polyphase resample (up=125/down=256: 1537->750),
            # then tile the first 250 samples onto the 750 -> 1000 samples so the
            # window matches the pretrained MIRepNet length (4 s @ 250 Hz).
            # Subjects are contiguous 60-trial blocks. Class filter is applied after.
            self.paradigm = 'MI'
            self.num_subjects = 8
            self.sample_rate = 250
            self.ch_num = 16

            target_subjects = (self.args.sub if hasattr(self.args, 'sub')
                               else range(self.num_subjects))
            indices = [np.arange(60) + 60 * int(i) for i in target_subjects]
            indices = np.concatenate(indices, axis=0)
            X = X[indices]
            y = y[indices]

            X = resample(X, up=125, down=256, axis=2)   # mne.filter.resample: 512->250 Hz
            X = X[:, :, :750]
            if X.shape[2] == 750:
                X = np.concatenate([X, X[:, :, :250]], axis=2)   # 750 -> 1000 (tile head)
            else:
                X = X[:, :, :1000]
            X = X.astype(np.float32)

            valid = np.isin(y, ['right_hand', 'feet'])   # drop 'rest' -> 2-class
            X = X[valid]
            y = y[valid]
            print(f"Loaded AlexMI (2-class RH/feet) subjects {list(target_subjects)}: "
                  f"{len(y)} trials, shape {X.shape}")

        elif self.dataset_name == 'BNCI2015001':
            # 512 Hz, not natively 250 Hz. Select subject(s) via meta.csv, use ONLY
            # session_A (default; override via env MI2015001_SESSION) so the 70/30
            # split is a single within-session split. Resample 512->250, truncate to
            # 1000 (divisible by IFNet patch_size=125).
            from scipy.signal import resample as _sresample
            self.paradigm = 'MI'
            self.sample_rate = 250
            orig_fs = 512
            self.num_subjects = 12
            self.ch_num = 13

            meta = pd.read_csv('/data1/llx/' + self.dataset_name + '/meta.csv')
            subj_col = np.asarray(meta['subject'].values)  # 1-indexed
            target_subjects = (self.args.sub if hasattr(self.args, 'sub')
                               else range(self.num_subjects))
            wanted = [int(s) + 1 for s in target_subjects]  # 0-idx -> 1-idx meta
            subj_mask = np.isin(subj_col, wanted)
            sess = os.environ.get('MI2015001_SESSION', 'session_A')
            subj_mask &= (np.asarray(meta['session'].values) == sess)
            sel = np.where(subj_mask)[0]
            X = X[sel]
            y = y[sel]
            print(f"Loaded {self.dataset_name} subjects {wanted} "
                  f"(0-idx {list(target_subjects)}): {len(sel)} trials")

            n250 = int(round(X.shape[2] * self.sample_rate / orig_fs))
            X = _sresample(X, n250, axis=2)
            L = min(1000, (n250 // 125) * 125)
            X = X[:, :, :L].astype(np.float32)

        else:
            self.paradigm = None
            self.num_subjects = None
            self.sample_rate = None
            self.ch_num = None

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        print("preprocessed data shape:", X.shape, "preprocessed labels shape:", y.shape)

        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # Ensure label is of type long

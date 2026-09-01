"""Adapter for LaBraM (foundation model, runs in the `labram` conda env).

Input contract: ``[B, n_electrodes, n_patches, 200]`` at 200 Hz (1-s patches). The
canonical epoch ``(N, C_native, 1000)`` @ 250 Hz is resampled to 800 @ 200 Hz,
µV-scaled, and reshaped into 4 patches of 200.

LaBraM selects per-channel positional embeddings via ``input_chans`` — indices
into a fixed ``standard_1020`` montage, computed from the dataset's channel names
(``utils.get_input_chans``). Channel names come from MIRepNet's ``channel_list``
(uppercase, already matching the montage and the raw data's channel order).

The pretrained checkpoint stores the model under ``ckpt['model']`` with a
``student.`` prefix; we strip it and load non-strictly (logit_scale / mask_token /
head don't apply to a fresh fine-tune head).

``cfg``: ``dataset_name`` (selects channel names), ``pretrain`` (labram-base.pth),
``scale`` (µV divisor, default 100), ``patch_num`` (default 4).
"""
import numpy as np
import torch
from scipy.signal import resample as scipy_resample

import config
from data.preproc import SRC_FS, DST_FS
from models.base import ModelAdapter

PATCH = 200        # samples per 1-s patch at DST_FS (model input format)


class LaBraMAdapter(ModelAdapter):
    name = 'labram'

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        self.input_chans = None  # set in build() once ch_names are known

    def _ch_names(self):
        from data import channels as cl
        table = {
            'BNCI2014001': cl.BNCI2014001_chn_names,
            'BNCI2014001-4': cl.BNCI2014001_chn_names,
            'BNCI2014004': cl.BNCI2014004_chn_names,
        }
        return table[self.cfg['dataset_name']]

    def preprocess(self, X_raw):
        x = np.asarray(X_raw, dtype=np.float32) / self.cfg.get('scale', 100.0)
        n_dst = int(round(x.shape[2] * DST_FS / SRC_FS))    # 1000 -> 800
        x = scipy_resample(x, n_dst, axis=2).astype(np.float32)
        patch_num = self.cfg.get('patch_num', n_dst // PATCH)
        assert patch_num * PATCH == x.shape[2], (
            f'resampled length {x.shape[2]} not divisible into {PATCH}-pt patches')
        N, C, _ = x.shape
        return torch.as_tensor(x.reshape(N, C, patch_num, PATCH),
                               dtype=torch.float32)

    def build(self, num_classes):
        from .modeling_finetune import labram_base_patch200_200
        from .montage import get_input_chans

        self.input_chans = get_input_chans(self._ch_names())
        # kwargs mirror LaBraM's run_class_finetuning create_model defaults; abs
        # pos emb is ON so the pretrained per-channel pos_embed is loaded + used
        # (it is LaBraM's channel-position mechanism, indexed by input_chans).
        model = labram_base_patch200_200(
            num_classes=num_classes, drop_path_rate=0.1, use_mean_pooling=True,
            init_values=0.1, qkv_bias=True, use_abs_pos_emb=True,
            use_rel_pos_bias=False)

        pretrain = self.cfg.get('pretrain') or config.weight_path('labram')
        ckpt = torch.load(pretrain, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        sd = {k[len('student.'):]: v for k, v in sd.items()
              if k.startswith('student.')}
        # drop shape-mismatched keys (e.g. head) before non-strict load
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[labram] loaded {len(sd)} tensors; '
              f'{len(missing)} missing, {len(unexpected)} unexpected')
        return model.to(self.device)

    def forward(self, model, x):
        feat = model.forward_features(x, input_chans=self.input_chans)
        logits = model.head(feat)
        return feat, logits


ADAPTER = LaBraMAdapter

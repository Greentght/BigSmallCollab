"""
Parameter-efficient fine-tuning (PEFT) for the MIRepNet transformer teacher.

Hand-written LoRA (no `peft` dependency, because mlm_mask is a plain nn.Module,
not a HuggingFace model). Also provides BitFit and linear-probe configurations
so the "freeze-the-backbone" experiment has more than one PEFT data point.

Design notes (see CLAUDE.md / the PEFT experiment plan):
  * LoRA is injected on the attention projections of every TransformerEncoderBlock.
    The projections are named `queries` / `keys` / `values` / `projection`
    inside model.mlm.MultiHeadAttention.
  * Two things are NEVER frozen, or PEFT silently fails:
      1. the classification head (`clshead`) — it is randomly re-init'd per run;
      2. LayerNorm scale/bias — downstream distribution differs from pretrain,
         and norm stats must re-adapt. (This also covers half of BitFit.)
  * `configure_peft` returns the (n_trainable, n_total) param counts so the
    trainable fraction can be reported — that number goes in the paper.
"""

import math
import torch
import torch.nn as nn

from .mlm import MultiHeadAttention, FeedForwardBlock


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update B @ A.

    forward(x) = base(x) + scaling * (dropout(x) @ A^T @ B^T)
    A is kaiming-init'd, B is zero-init'd so the initial delta is exactly 0
    (the model starts identical to the pretrained backbone).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRALinear wraps nn.Linear only"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> delta = 0 at init

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + self.scaling * delta


def _parse_targets(targets: str):
    """'qv' -> {'q','v'}; 'qkvo' -> all four; append 'ffn' to also hit the MLP.
    Accepts e.g. 'qv', 'qkvo', 'qv_ffn', 'qkvo_ffn'."""
    targets = targets.lower()
    sel = set()
    if 'q' in targets:
        sel.add('q')
    if 'k' in targets:
        sel.add('k')
    if 'v' in targets:
        sel.add('v')
    if 'o' in targets:
        sel.add('o')
    if 'ffn' in targets:
        sel.add('ffn')
    return sel


def apply_lora(model: nn.Module, rank: int, alpha: float,
               targets: str = 'qv', dropout: float = 0.0) -> int:
    """Inject LoRALinear into the targeted projections. Returns #layers wrapped.

    Attention projection name map (model.mlm.MultiHeadAttention):
        q -> queries, k -> keys, v -> values, o -> projection
    'ffn' wraps the two Linear layers inside FeedForwardBlock (indices 0 and 3).
    """
    sel = _parse_targets(targets)
    name_map = {'q': 'queries', 'k': 'keys', 'v': 'values', 'o': 'projection'}
    n_wrapped = 0

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            for key, attr in name_map.items():
                if key in sel:
                    base = getattr(module, attr)
                    setattr(module, attr, LoRALinear(base, rank, alpha, dropout))
                    n_wrapped += 1
        if 'ffn' in sel and isinstance(module, FeedForwardBlock):
            # FeedForwardBlock = Sequential(Linear, GELU, Dropout, Linear)
            module[0] = LoRALinear(module[0], rank, alpha, dropout)
            module[3] = LoRALinear(module[3], rank, alpha, dropout)
            n_wrapped += 2

    return n_wrapped


def _count_params(model: nn.Module):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


def configure_peft(model: nn.Module, peft: str = 'none',
                   lora_rank: int = 8, lora_alpha: float = 8.0,
                   lora_targets: str = 'qv', lora_dropout: float = 0.0):
    """Set requires_grad on `model` according to the PEFT scheme.

    peft:
      'none'         -> full fine-tune (everything trainable; no change).
      'linear_probe' -> freeze everything except the classification head.
      'bitfit'       -> train only biases + LayerNorm (+ head).
      'lora'         -> freeze backbone, inject LoRA, train LoRA + LayerNorm (+ head).

    The head (`clshead`) and LayerNorm are always trainable for 'bitfit'/'lora'
    (and the head for 'linear_probe'); see module docstring for why.

    Returns dict with n_trainable / n_total / trainable_frac / n_lora_layers.
    """
    peft = peft.lower()
    n_lora_layers = 0

    if peft == 'none':
        for p in model.parameters():
            p.requires_grad = True

    elif peft == 'linear_probe':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.clshead.parameters():
            p.requires_grad = True

    elif peft == 'bitfit':
        for name, p in model.named_parameters():
            p.requires_grad = name.endswith('.bias') or ('clshead' in name)
        # LayerNorm scale (weight) is not a bias -> unfreeze it explicitly.
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    elif peft == 'lora':
        for p in model.parameters():
            p.requires_grad = False
        n_lora_layers = apply_lora(model, lora_rank, lora_alpha,
                                   lora_targets, lora_dropout)
        # apply_lora creates fresh nn.Parameters -> already requires_grad=True.
        # Unfreeze LayerNorm + head.
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
        for p in model.clshead.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown peft scheme: {peft!r} "
                         f"(expected none/linear_probe/bitfit/lora)")

    n_trainable, n_total = _count_params(model)
    return {
        'peft': peft,
        'n_trainable': n_trainable,
        'n_total': n_total,
        'trainable_frac': n_trainable / n_total if n_total else 0.0,
        'n_lora_layers': n_lora_layers,
    }

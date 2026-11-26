import torch
import torch.nn.functional as F

from collab.dml_ds_heads import build_heads
from models.adfcnn.adfcnn import ADFCNN_Net
from models.mirepnet.mlm import mlm_mask


def test_new_auxiliary_heads_connect_to_model_features():
    torch.set_num_threads(1)
    examples = [
        ('mirepnet', mlm_mask(emb_size=256, depth=6, n_classes=4), torch.randn(2, 45, 1000)),
        ('adfcnn', ADFCNN_Net(in_channels=3, samples=1000, num_classes=4), torch.randn(2, 3, 1000)),
    ]
    for architecture, model, x in examples:
        heads = build_heads(model, architecture, 4)
        model(x)
        logits = heads(len(x))
        assert len(logits) == 2 and all(z.shape == (2, 4) for z in logits)
        sum(F.cross_entropy(z, torch.tensor([0, 1])) for z in logits).backward()
        assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                   for head in heads.heads)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        heads.close()

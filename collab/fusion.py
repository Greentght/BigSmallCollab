"""F+T — few-shot subject-adaptive fusion heads over two models' frozen features.

D0 + R1 established: cross-subject *static* collaboration is capped (the "who is
right" signal is subject-specific — penultimate-feature separability is 0.999
in-sample but 0.507 across subjects), so the synergy is only reachable by adapting
on the test subject. This module fits lightweight heads on ``K`` labeled trials of
the *test subject* over the cached frozen features (big D_b, small D_s) and fuses
them. No base model is retrained.

Heads (all trained on K trials, evaluated on the held-out rest of the subject):
  head_big / head_small : linear head on one model's features (the controls — is
                          fusion better than adapting either model alone at same K?)
  fusion_lr / fusion_mlp: linear / MLP head on the concatenated features
  gated                 : per-sample gate a(feats) mixing the two per-model heads
                          p = a * softmax(W_b f_b) + (1-a) * softmax(W_s f_s)
  mutual (feature-level B): co-train head_big & head_small with a symmetric-KL
                          mutual term on the K trials (deep mutual learning in the
                          adapted-head space), then average — the B verification.

All heads standardise features with train-split stats. Torch heads are tiny and
run fine on CPU (K is small, D<=768).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fit_linear(Xtr, ytr, nc, epochs=300, lr=5e-2, wd=1e-2, seed=0, device='cpu'):
    torch.manual_seed(seed)
    lin = nn.Linear(Xtr.shape[1], nc).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xt), yt)
        loss.backward(); opt.step()
    return lin


@torch.no_grad()
def _logits(lin, X, device='cpu'):
    return lin(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def cv_acc(feats, y, nc, folds=3, C=0.3):
    """Internal stratified-CV accuracy of a linear head on ``feats`` — the support-
    set estimate used to gate/select among candidates without touching the eval set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    counts = np.bincount(y, minlength=nc)
    if counts[counts > 0].min() < 2:
        return 0.0
    k = int(min(folds, counts[counts > 0].min()))
    accs = []
    for tr, va in StratifiedKFold(n_splits=k).split(feats, y):
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=500, C=C).fit((feats[tr] - mu) / sd, y[tr])
        accs.append((clf.predict((feats[va] - mu) / sd) == y[va]).mean())
    return float(np.mean(accs))


def _sk_logreg(Xtr, ytr, Xte, nc, C=0.3):
    """Fast, deterministic L2 logistic head (used for the linear methods)."""
    from sklearn.linear_model import LogisticRegression
    if len(set(ytr.tolist())) < 2:                      # degenerate support
        return np.full(len(Xte), int(ytr[0]))
    clf = LogisticRegression(max_iter=1000, C=C).fit(Xtr, ytr)
    return clf.predict(Xte)


def head_single(ftr, ytr, fte, nc, **kw):
    Xtr, Xte = _standardize(ftr, fte)
    return _sk_logreg(Xtr, ytr, Xte, nc)


def fusion_concat(bf_tr, sf_tr, ytr, bf_te, sf_te, nc, hidden=0, **kw):
    ftr = np.concatenate([bf_tr, sf_tr], 1)
    fte = np.concatenate([bf_te, sf_te], 1)
    Xtr, Xte = _standardize(ftr, fte)
    if hidden <= 0:
        return _sk_logreg(Xtr, ytr, Xte, nc)
    seed = kw.get('seed', 0); torch.manual_seed(seed)
    device = kw.get('device', 'cpu')
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(hidden, nc)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=kw.get('lr', 1e-2),
                           weight_decay=kw.get('wd', 1e-2))
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    for _ in range(kw.get('epochs', 300)):
        opt.zero_grad(); F.cross_entropy(net(Xt), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.as_tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


class _Gated(nn.Module):
    def __init__(self, db, ds, nc):
        super().__init__()
        self.hb = nn.Linear(db, nc); self.hs = nn.Linear(ds, nc)
        self.gate = nn.Sequential(nn.Linear(db + ds, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, fb, fs):
        a = torch.sigmoid(self.gate(torch.cat([fb, fs], 1)))       # (N,1)
        pb, ps = F.softmax(self.hb(fb), 1), F.softmax(self.hs(fs), 1)
        return a * pb + (1 - a) * ps, a


def fusion_gated(bf_tr, sf_tr, ytr, bf_te, sf_te, nc, epochs=400, lr=1e-2,
                 wd=1e-2, seed=0, device='cpu'):
    torch.manual_seed(seed)
    bt, be = _standardize(bf_tr, bf_te); st, se = _standardize(sf_tr, sf_te)
    net = _Gated(bt.shape[1], st.shape[1], nc).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Bt = torch.as_tensor(bt, dtype=torch.float32, device=device)
    St = torch.as_tensor(st, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        p, _ = net(Bt, St)
        F.nll_loss(torch.log(p + 1e-12), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p, _ = net(torch.as_tensor(be, dtype=torch.float32, device=device),
                   torch.as_tensor(se, dtype=torch.float32, device=device))
    return p.argmax(1).cpu().numpy()


def fusion_mutual(bf_tr, sf_tr, ytr, bf_te, sf_te, nc, epochs=400, lr=1e-2,
                  wd=1e-2, lam=1.0, seed=0, device='cpu'):
    """Feature-level B: co-train the two per-model heads with a symmetric-KL mutual
    term (deep mutual learning), then average their probs. Tests whether making the
    two adapted heads teach each other beats plain concat fusion."""
    torch.manual_seed(seed)
    bt, be = _standardize(bf_tr, bf_te); st, se = _standardize(sf_tr, sf_te)
    hb = nn.Linear(bt.shape[1], nc).to(device); hs = nn.Linear(st.shape[1], nc).to(device)
    opt = torch.optim.Adam(list(hb.parameters()) + list(hs.parameters()), lr=lr, weight_decay=wd)
    Bt = torch.as_tensor(bt, dtype=torch.float32, device=device)
    St = torch.as_tensor(st, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    kl = nn.KLDivLoss(reduction='batchmean')
    for _ in range(epochs):
        opt.zero_grad()
        lb, ls = hb(Bt), hs(St)
        pb, ps = F.softmax(lb, 1), F.softmax(ls, 1)
        loss = (F.cross_entropy(lb, yt) + F.cross_entropy(ls, yt)
                + lam * (kl(F.log_softmax(lb, 1), ps.detach())
                         + kl(F.log_softmax(ls, 1), pb.detach())))
        loss.backward(); opt.step()
    hb.eval(); hs.eval()
    with torch.no_grad():
        pb = F.softmax(hb(torch.as_tensor(be, dtype=torch.float32, device=device)), 1)
        ps = F.softmax(hs(torch.as_tensor(se, dtype=torch.float32, device=device)), 1)
    return (pb + ps).argmax(1).cpu().numpy()

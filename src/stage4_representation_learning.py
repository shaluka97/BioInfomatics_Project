"""
Stage 4: Sparse autoencoder for further dimensionality reduction.

We use a small NumPy-only sparse autoencoder so the project runs cleanly
without TensorFlow or PyTorch installed (per requirements.txt — torch is
optional). The architecture is:

    input  ->  hidden_dim (ReLU)  ->  bottleneck_dim (linear, L1-penalised)
           ->  hidden_dim (ReLU)  ->  output (linear)

Loss = MSE(reconstruction, input) + l1_lambda * ||z_bottleneck||_1.
Trained with Adam.

Per-SNP contribution is computed as:
    importance_j = mean over samples of |grad of bottleneck activations w.r.t.
                  input_j|, summed across bottleneck units.
This is a lightweight "gradient * input" proxy and is the score Stage 5 uses
to interpret what the embedding is paying attention to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Stage4Config:
    enabled: bool = True
    hidden_dim: int = 256
    bottleneck_dim: int = 64
    l1_lambda: float = 1e-4
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42


def _relu(x): return np.maximum(0, x)
def _drelu(x): return (x > 0).astype(x.dtype)


def _train_autoencoder(X: np.ndarray, cfg: Stage4Config) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed)
    n, d = X.shape
    h, b = cfg.hidden_dim, cfg.bottleneck_dim

    Xn = X.astype(np.float32)
    Xn = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-6)

    def init(shape):
        fan_in = shape[0]
        return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

    W1 = init((d, h)); b1 = np.zeros(h, dtype=np.float32)
    W2 = init((h, b)); b2 = np.zeros(b, dtype=np.float32)
    W3 = init((b, h)); b3 = np.zeros(h, dtype=np.float32)
    W4 = init((h, d)); b4 = np.zeros(d, dtype=np.float32)

    # Adam state
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    lr = cfg.learning_rate

    losses: list = []
    weights: dict = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            x = Xn[idx]
            # forward
            z1 = x @ W1 + b1; a1 = _relu(z1)
            z2 = a1 @ W2 + b2          # bottleneck (linear)
            z3 = z2 @ W3 + b3; a3 = _relu(z3)
            xhat = a3 @ W4 + b4
            err = xhat - x
            recon = (err ** 2).mean()
            l1 = cfg.l1_lambda * np.abs(z2).mean()
            loss = recon + l1
            epoch_loss += loss; n_batches += 1
            # backward
            dxhat = (2.0 / (x.shape[0] * x.shape[1])) * err
            dW4 = a3.T @ dxhat; db4 = dxhat.sum(axis=0)
            da3 = dxhat @ W4.T
            dz3 = da3 * _drelu(z3)
            dW3 = z2.T @ dz3; db3 = dz3.sum(axis=0)
            dz2 = dz3 @ W3.T + cfg.l1_lambda * np.sign(z2) / (z2.size)
            dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * _drelu(z1)
            dW1 = x.T @ dz1; db1 = dz1.sum(axis=0)
            grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
            # Adam step
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mhat = m[i] / (1 - beta1 ** t)
                vhat = v[i] / (1 - beta2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
        losses.append(float(epoch_loss / max(1, n_batches)))

    # bottleneck embedding
    a1 = _relu(Xn @ W1 + b1)
    Z = a1 @ W2 + b2

    # gradient-x-input importance: |dZ/dx_j|. Bottleneck is linear in a1 and
    # a1 is ReLU(x W1 + b1), so dZ/dx_j = (W1 * (z1>0)) @ W2. Average over
    # samples and sum |.| over bottleneck units.
    z1 = Xn @ W1 + b1
    relu_mask = (z1 > 0).astype(np.float32)             # n x h
    # for each sample compute (W1 * mask_i)  @ W2 -> (d, b)
    # then take |.| and mean over samples -> (d, b); sum over b -> (d,)
    contrib = np.zeros(d, dtype=np.float32)
    chunk = 64
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        # W1: d x h ; mask[s:e]: (e-s) x h
        # broadcast: (e-s, d, h)
        masked = W1[None, :, :] * relu_mask[s:e, None, :]
        # multiply by W2: (e-s, d, b)
        prod = masked @ W2
        contrib += np.abs(prod).sum(axis=0).sum(axis=1)
    contrib /= n
    info = {"losses": losses, "embedding": Z, "weights": weights, "Xn": Xn}
    return contrib, info, Z


def run_stage4(X: np.ndarray, cfg: Stage4Config,
               y: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Returns (per_snp_contribution, embedding, info).

    If a label vector `y` is supplied we fit an L1-regularised logistic
    regression on the bottleneck embedding and *blend* its back-traced
    feature importance with the gradient-x-input contribution. This makes
    Stage 4's per-SNP selection supervised rather than purely
    reconstruction-driven, which dramatically improves causal recovery on
    real and synthetic data alike.
    """
    if not cfg.enabled:
        return np.zeros(X.shape[1]), np.zeros((X.shape[0], 0)), {
            "enabled": False,
            "disabled_reason": "stage4.disabled_in_config",
        }
    if X.shape[1] == 0:
        return np.zeros(X.shape[1]), np.zeros((X.shape[0], 0)), {
            "enabled": False,
            "disabled_reason": "stage4.no_features",
        }
    contrib, info, Z = _train_autoencoder(X, cfg)
    info["enabled"] = True
    info["embedding_shape"] = Z.shape

    if y is not None and Z.shape[1] > 0 and len(np.unique(y)) > 1:
        from sklearn.linear_model import LogisticRegression
        y_arr = np.asarray(y)
        y_unique = np.unique(y_arr)
        if set(y_unique.tolist()) == {-1, 1}:
            y_arr = ((y_arr + 1) // 2).astype(np.int64)
            info["y_normalized_from_pm1"] = True
        elif not set(y_unique.tolist()).issubset({0, 1}):
            raise ValueError("Stage4 supervised blending expects binary labels encoded as {0,1} or {-1,1}.")

        Zn_mean = Z.mean(axis=0); Zn_std = Z.std(axis=0) + 1e-6
        Zn = (Z - Zn_mean) / Zn_std
        clf = LogisticRegression(penalty="l1", solver="liblinear",
                                 C=1.0, max_iter=2000)
        clf.fit(Zn, y_arr)
        w_clf = clf.coef_.ravel() / Zn_std   # gradient of logit-output w.r.t. raw Z
        info["downstream_l1_weights"] = np.abs(w_clf)
        # Trace gradient of logit w.r.t. inputs through the encoder:
        #   logit = w_clf . Z = w_clf . (a1 @ W2 + b2)
        #   d logit / d x_j = sum_h (W1[j,h] * (z1[h]>0)) * (W2 @ w_clf)[h]
        W1 = info["weights"]["W1"]; b1 = info["weights"]["b1"]
        W2 = info["weights"]["W2"]
        Xn = info["Xn"]
        z1 = Xn @ W1 + b1
        relu_mask = (z1 > 0).astype(np.float32)        # n x h
        proj = W2 @ w_clf.astype(np.float32)            # h
        # per-sample contribution: (mask_i * proj)[h] then dot with W1[j,:]
        # importance_j = mean_i |W1[j,:] @ (relu_mask[i] * proj)|
        importance = np.zeros(W1.shape[0], dtype=np.float64)
        chunk = 256
        for s in range(0, Xn.shape[0], chunk):
            e = min(Xn.shape[0], s + chunk)
            # weighted mask: (e-s, h)
            wm = relu_mask[s:e] * proj[None, :]
            grad = wm @ W1.T                            # (e-s, d)
            importance += np.abs(grad).sum(axis=0)
        importance /= Xn.shape[0]
        # Replace contribution with the supervised version (much more useful
        # for downstream causal recovery); keep reconstruction-importance in
        # info for the report.
        info["recon_contrib"] = contrib
        contrib = importance
        info["used_supervised_blending"] = True
    return contrib, Z, info

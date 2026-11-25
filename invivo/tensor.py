"""Signal model of the training data (axially symmetric tensors, lambda2 = lambda3) and a DTI reference fit.
Units as in the training data: b in s/mm^2, MD and eigenvalues in 10^-3 mm^2/s."""

import numpy as np


def load_protocol(path):
    """Protocol file: one row per measurement, b, gx, gy, gz. Returns b (N,), g (N, 3) unit vectors (0 for b = 0)."""
    t = np.loadtxt(path, delimiter=",")
    b, g = t[:, 0], t[:, 1:4]
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return b, np.divide(g, n, out=np.zeros_like(g), where=n > 0)


def eigenvalues(md, fa):
    """lambda1 = MD + 2d, lambda_perp = MD - d with d = MD FA / sqrt(3 - 2 FA^2): the tensor of the data generator."""
    fa = np.clip(fa, 0.0, 1.0)
    d = md * fa / np.sqrt(3 - 2 * fa ** 2)
    return md + 2 * d, md - d


def simulate(md, fa, e, f, b, g, chunk=2000):
    """Voxel signals sum_q f_q exp(-b g^T D_q g). md, fa, f: (V, Q); e: (V, Q, 3); returns (V, N)."""
    out = np.empty((len(md), len(b)))
    for i in range(0, len(md), chunk):
        s = slice(i, i + chunk)
        l1, lp = eigenvalues(md[s], fa[s])
        cos2 = np.einsum("vqk,nk->vqn", e[s], g) ** 2
        adc = lp[..., None] + (l1 - lp)[..., None] * cos2                       # (v, Q, N), 10^-3 mm^2/s
        out[s] = np.einsum("vq,vqn->vn", f[s], np.exp(-1e-3 * b * adc))
    return out


def tensor_metrics(D):
    """D: (V, 3, 3) in 10^-3 mm^2/s -> MD, FA, principal eigenvector (V, 3)."""
    lam, vec = np.linalg.eigh(D)
    md = lam.mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.sqrt(1.5 * ((lam - md[:, None]) ** 2).sum(1) / (lam ** 2).sum(1))
    return md, np.nan_to_num(np.clip(fa, 0, 1)), vec[:, :, -1]


def dti_fit(y, b, g, floor=1e-3, chunk=20000):
    """Weighted linear least squares DTI on b0-normalized signals y (V, N); weights y^2 (Salvador et al. 2005).
    Returns MD, FA, principal direction, fitted signals (V, N)."""
    A = np.column_stack([np.ones_like(b), -b * g[:, 0] ** 2, -b * g[:, 1] ** 2, -b * g[:, 2] ** 2,
                         -2 * b * g[:, 0] * g[:, 1], -2 * b * g[:, 0] * g[:, 2], -2 * b * g[:, 1] * g[:, 2]])
    ys = np.maximum(y, floor)
    x = np.empty((len(y), 7))
    for i in range(0, len(y), chunk):
        s = slice(i, i + chunk)
        W = ys[s] ** 2
        AtWA = np.einsum("vn,ni,nj->vij", W, A, A)
        AtWy = np.einsum("vn,ni,vn->vi", W, A, np.log(ys[s]))
        x[s] = np.linalg.solve(AtWA, AtWy[..., None])[..., 0]
    d = x[:, 1:] * 1e3                                                          # mm^2/s -> 10^-3 mm^2/s
    D = np.stack([np.stack([d[:, 0], d[:, 3], d[:, 4]], -1),
                  np.stack([d[:, 3], d[:, 1], d[:, 5]], -1),
                  np.stack([d[:, 4], d[:, 5], d[:, 2]], -1)], 1)
    md, fa, v1 = tensor_metrics(D)
    return md, fa, v1, np.exp(x @ A.T)

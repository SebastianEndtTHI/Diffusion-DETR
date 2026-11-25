"""DETR detections -> voxel maps, fit residual, and clustering of the pooled compartments.
P: (V, Q, 7) = [MD, FA, dir_x, dir_y, dir_z, weight, existence probability] as the model outputs them."""

import numpy as np

from invivo import config as C
from invivo.tensor import simulate

MD_MAX = 4.0


# ─────────────────────────────────────────────────────────────────────────
# Detections and voxel maps
# ─────────────────────────────────────────────────────────────────────────
def detected(P, thr, at_least_one=C.MIN_ONE_COMPARTMENT):
    """(V, Q) bool: queries with existence >= thr; with at_least_one, a voxel without any keeps its most confident
    query. Returns (keep, number of voxels where that applied)."""
    keep = P[..., 6] >= thr
    none = ~keep.any(1)
    if at_least_one:
        keep[np.where(none)[0], P[none, :, 6].argmax(1)] = True
    return keep, int(none.sum())


def fractions(P, keep):
    """Signal fractions of the kept compartments, normalized to sum 1 per voxel (V, Q)."""
    w = np.where(keep, P[..., 5], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return w / w.sum(1, keepdims=True)


def unit_dirs(P):
    e = P[..., 2:5].astype(float)
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)


def voxel_maps(P, keep):
    """Signal-fraction-weighted means: MD, FA, colour FA (sum_q f_q FA_q |e_q|, RGB); number of compartments; sum of
    the raw predicted weights of the kept compartments (training: 1)."""
    f = fractions(P, keep)
    fa = P[..., 1]
    return {"MD": (f * P[..., 0]).sum(1), "FA": (f * fa).sum(1),
            "cFA": np.einsum("vq,vqk->vk", f * fa, np.abs(unit_dirs(P))),
            "n_c": keep.sum(1).astype(float), "sum_w": np.where(keep, P[..., 5], 0).sum(1)}


def fit_rmse(P, keep, X, protocol):
    """RMS difference between the b0-normalized data and the signal re-simulated from the kept compartments with
    their signal fractions (sum 1, so S(b=0) = 1 as in the data). Returns rmse (V,)."""
    f = np.nan_to_num(fractions(P, keep))
    s = simulate(P[..., 0].astype(float), P[..., 1].astype(float), unit_dirs(P), f, *protocol)
    return np.sqrt(((X.astype(float) - s) ** 2).mean(1))


# ─────────────────────────────────────────────────────────────────────────
# Clustering features
# ─────────────────────────────────────────────────────────────────────────
def compartments(P, keep):
    """Pooled kept compartments: md, fa, e (K, 3), signal fraction, voxel index."""
    v, q = np.nonzero(keep)
    return P[v, q, 0].astype(float), P[v, q, 1].astype(float), unit_dirs(P)[v, q], fractions(P, keep)[v, q], v


def _axis(e):
    x, y, z = e.T
    s = np.sqrt(2)
    return np.stack([x * x, y * y, z * z, s * x * y, s * x * z, s * y * z], 1) / s


def _angles(e):
    e = np.where(e[:, 2:3] < 0, -e, e)                                   # z >= 0 hemisphere
    theta = np.degrees(np.arccos(np.clip(e[:, 2], -1, 1))) / 90.0
    phi = np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 360.0 / 180.0       # [0, 2): full circle on the hemisphere
    return np.stack([theta, phi], 1)


def features(md, fa, e, mode=C.CLUSTER["direction"], dir_weight=C.CLUSTER["dir_weight"]):
    """MD / 4, FA and, unless mode == 'none', the FA-weighted direction block (see config.CLUSTER)."""
    base = np.stack([md / MD_MAX, fa], 1)
    if mode == "none":
        return base
    d = {"axis": _axis, "angles": _angles}[mode](e)
    return np.hstack([base, dir_weight * fa[:, None] * d])


def center_params(mu, mode=C.CLUSTER["direction"], dir_weight=C.CLUSTER["dir_weight"]):
    """Cluster centres in features -> MD, FA, mean direction (K, 3) (NaN without a direction block)."""
    md, fa = mu[:, 0] * MD_MAX, mu[:, 1]
    e = np.full((len(mu), 3), np.nan)
    scale = dir_weight * np.maximum(fa, 1e-6)[:, None]
    if mode == "axis":
        a = mu[:, 2:] / scale * np.sqrt(2)
        s = np.sqrt(2)
        for k, (xx, yy, zz, xy, xz, yz) in enumerate(a):
            M = np.array([[xx, xy / s, xz / s], [xy / s, yy, yz / s], [xz / s, yz / s, zz]])
            e[k] = np.linalg.eigh(M)[1][:, -1]
    elif mode == "angles":
        th, ph = np.radians((mu[:, 2:] / scale * [90.0, 180.0]).T)
        e = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return md, fa, e


# ─────────────────────────────────────────────────────────────────────────
# Weighted Gaussian mixture (ported from the T1-T2 project, any dimension)
# ─────────────────────────────────────────────────────────────────────────
def _log_gauss(Z, mu, cov):
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (Z - mu).T)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * Z.shape[1] * np.log(2 * np.pi)


def _weighted_gmm(Z, w, k, seed, n_init, max_iter=500, tol=1e-7, reg=C.CLUSTER["min_std"] ** 2):
    """Full-covariance GMM by EM with per-point weights (fitted to the signal fractions, not to compartment counts)."""
    from sklearn.cluster import KMeans
    w = w / w.sum()
    D = Z.shape[1]
    best = None
    for i in range(n_init):
        r = np.eye(k)[KMeans(k, n_init=1, random_state=seed + i).fit(Z, sample_weight=w).labels_]
        ll_old = -np.inf
        for _ in range(max_iter):
            nk = (w[:, None] * r).sum(0) + 1e-12
            mu = (w[:, None] * r).T @ Z / nk[:, None]
            cov = np.stack([((w * r[:, j])[:, None] * (Z - mu[j])).T @ (Z - mu[j]) / nk[j] + reg * np.eye(D)
                            for j in range(k)])
            logp = np.stack([np.log(nk[j]) + _log_gauss(Z, mu[j], cov[j]) for j in range(k)], 1)
            lse = logp.max(1) + np.log(np.exp(logp - logp.max(1, keepdims=True)).sum(1))
            r = np.exp(logp - lse[:, None])
            ll = float((w * lse).sum())
            if ll - ll_old < tol:
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = {"pi": nk, "mu": mu, "cov": cov, "ll": ll}
    return best


def fit_clusters(Z, w, n_clusters, seed=C.SEED):
    """Weighted GMM on a subsample of the compartment features; clusters sorted by MD. Returns (model, BIC with the
    Kish effective sample size)."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z), min(C.CLUSTER["n_fit"], len(Z)), replace=False)
    Zf, wf = Z[sub], w[sub]
    n_eff = wf.sum() ** 2 / (wf ** 2).sum()
    g = _weighted_gmm(Zf, wf, n_clusters, seed, C.CLUSTER["n_init"])
    D = Z.shape[1]
    bic = float(-2 * n_eff * g["ll"] + (n_clusters * (D + D * (D + 1) / 2) + n_clusters - 1) * np.log(n_eff))
    o = np.argsort(g["mu"][:, 0])
    return {"pi": g["pi"][o], "mu": g["mu"][o], "cov": g["cov"][o]}, bic


def responsibilities(g, Z):
    logp = np.stack([np.log(g["pi"][j]) + _log_gauss(Z, g["mu"][j], g["cov"][j]) for j in range(len(g["pi"]))], 1)
    r = np.exp(logp - logp.max(1, keepdims=True))
    return r / r.sum(1, keepdims=True)


def cluster_fractions(g, Z, f, v, V):
    """Per voxel and cluster: sum_q f_q p(c | q) (V, K); the f_q sum to 1 per voxel."""
    r = responsibilities(g, Z) * f[:, None]
    return np.stack([np.bincount(v, weights=r[:, j], minlength=V) for j in range(r.shape[1])], 1)

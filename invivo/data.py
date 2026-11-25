"""In vivo data: loading, matching to the training protocol, b0 normalization, mask, noise level (MP-PCA)."""

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

from invivo import config as C


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────
def load_array(path, ndims):
    """NIfTI or .mat (exactly one array with ndim in ndims). Returns (array, affine)."""
    if path.endswith(".mat"):
        arrays = {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__") and np.ndim(v) in ndims}
        if len(arrays) != 1:
            raise ValueError(f"{path}: expected exactly one {'/'.join(map(str, ndims))}D array, found {list(arrays)}")
        return np.asarray(next(iter(arrays.values()))), np.eye(4)
    import nibabel as nib
    img = nib.load(path)
    a = np.asarray(img.dataobj)
    if a.ndim not in ndims:
        raise ValueError(f"{path}: {a.ndim}D, expected {ndims}")
    return a, img.affine


def load_gradients(bval_path, bvec_path):
    b = np.loadtxt(bval_path).ravel().astype(float)
    g = np.loadtxt(bvec_path).astype(float)
    if g.shape[0] == 3 and g.shape[1] != 3:
        g = g.T
    if g.shape != (len(b), 3):
        raise ValueError(f"bvec {g.shape} does not match {len(b)} b-values")
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return b, np.divide(g, n, out=np.zeros_like(g), where=n > 0)


def match_protocol(b, g, b_p, g_p, b_tol=C.B_TOL, b0_max=C.B0_MAX, angle_tol=C.ANGLE_TOL_DEG):
    """
    One data volume per protocol entry with b > b0_max: Hungarian assignment on |b - b_p| / b_tol + angle / angle_tol,
    the angle between the gradient directions up to sign. Protocol b = 0 entries get the mean b0 (index -1).
    Returns (index (N_p,), b0 volume indices, report dict). Raises if an entry has no volume within the tolerances.
    """
    b0 = np.where(b <= b0_max)[0]
    dw = np.where(b > b0_max)[0]
    pw = np.where(b_p > b0_max)[0]
    if not len(b0):
        raise ValueError("no b = 0 volume in the data")
    ang = np.degrees(np.arccos(np.clip(np.abs(g_p[pw] @ g[dw].T), 0, 1)))       # (n_p, n_d)
    db = np.abs(b_p[pw, None] - b[None, dw])
    cost = np.where((db <= b_tol) & (ang <= angle_tol), db / b_tol + ang / angle_tol, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    if ok.sum() < len(pw):
        best = np.where(db <= b_tol, ang, np.inf).min(1)                       # per entry, same shell
        raise ValueError(f"only {ok.sum()} of {len(pw)} diffusion-weighted protocol entries have a data volume within "
                         f"{b_tol:g} s/mm^2 and {angle_tol:g} deg (median smallest angle within the shell {np.median(best):.1f} deg). "
                         f"Data b-values: {np.unique(np.round(b, -1))}. If the angles are large, the bvecs may be in "
                         f"another frame (e.g. FSL image frame vs. scanner frame) than the protocol.")
    idx = np.full(len(b_p), -1)
    idx[pw[r]] = dw[c]
    rep = {"n_volumes": len(b), "n_b0": len(b0), "n_used": int(ok.sum()),
           "angle_deg_median_max": [float(np.median(ang[r, c])), float(ang[r, c].max())],
           "b_diff_max": float(db[r, c].max())}
    return idx, b0, rep


def auto_mask(s0):
    from dipy.segment.mask import median_otsu
    return median_otsu(s0, median_radius=4, numpass=4)[1]


def load_invivo(protocol, files):
    """
    files: (dwi, bval, bvec). Returns dict: X (V, N_p) b0-normalized signals in protocol order, S0 (V,),
    mask (nx, ny, nz), affine, report. Voxels: inside the mask (and config.SLICES) with finite data and S0 > 0.
    """
    dwi, affine = load_array(files[0], (4,))
    b, g = load_gradients(files[1], files[2])
    if dwi.shape[-1] != len(b):
        raise ValueError(f"{dwi.shape[-1]} volumes but {len(b)} b-values")
    idx, b0, rep = match_protocol(b, g, *protocol)
    s0 = dwi[..., b0].astype(float).mean(-1)
    if C.MASK:
        mask = load_array(C.MASK, (3,))[0] != 0
        if mask.shape != s0.shape:
            raise ValueError(f"mask {mask.shape} does not match the data {s0.shape}")
    else:
        mask = auto_mask(s0)
    if C.SLICES is not None:
        z0, z1 = C.SLICES
        sel = np.zeros_like(mask)
        sel[:, :, z0:z1 + 1] = True
        mask &= sel
    used = np.r_[b0, idx[idx >= 0]]
    mask &= (s0 > 0) & np.isfinite(dwi[..., used]).all(-1)
    S = dwi[mask].astype(float)
    S0 = s0[mask]
    X = np.where(idx >= 0, S[:, np.maximum(idx, 0)], S0[:, None]) / S0[:, None]
    sigma = noise_sigma(dwi[..., used], mask) / S0                   # relative to S0, as the training noise
    rep["voxels"] = int(mask.sum())
    rep["noise_estimated"] = int(np.isfinite(sigma).sum())
    print(f"In vivo {dwi.shape}: {rep['n_b0']} b0 volumes, {rep['n_used']} matched to the protocol "
          f"(angle median / max {rep['angle_deg_median_max'][0]:.2f} / {rep['angle_deg_median_max'][1]:.2f} deg), "
          f"{rep['voxels']} voxels")
    return {"X": X.astype(np.float32), "S0": S0.astype(np.float32), "sigma": sigma.astype(np.float32), "mask": mask,
            "affine": affine, "report": rep}


def to_volume(values, mask):
    """Per-voxel values (V, ...) -> (nx, ny, nz, ...) with NaN outside the mask."""
    out = np.full(mask.shape + np.shape(values)[1:], np.nan)
    out[mask] = values
    return out


# ─────────────────────────────────────────────────────────────────────────
# Noise level
# ─────────────────────────────────────────────────────────────────────────
def noise_sigma(dwi, mask, patch_radius=C.NOISE["patch_radius"]):
    """Local noise level (data units) per mask voxel by MP-PCA (Veraart et al. 2016, dipy implementation) on the raw
    volumes dwi (nx, ny, nz, P). Patches of (2r+1)^3 voxels, more than P. Voxels within r of the volume border get no
    estimate (NaN)."""
    from dipy.denoise.localpca import mppca
    _, sig = mppca(dwi.astype(np.float32), mask=mask, patch_radius=patch_radius, return_sigma=True,
                   suppress_warning=True)
    sig = sig[mask].astype(float)
    sig[~(sig > 0)] = np.nan
    return sig

"""Figures of the in vivo evaluation. Maps show one axial slice (config.DISPLAY_SLICE), the mask only (white outside),
cropped to it."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from invivo import config as C
from invivo.analysis import MD_MAX

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
METHODS = {"detr": "DETR", "dti": "DTI"}
CLUSTER_COLORS = plt.get_cmap("tab10").colors


def _save(fig, path, formats=(".png", ".pdf")):
    for ext in formats:
        fig.savefig(os.path.splitext(path)[0] + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.splitext(path)[0]}")


def _view(vol, mask, z, margin=2):
    """Axial slice z of vol (nx, ny, nz, ...), rotated by config.ROT90, cropped to the mask, NaN outside."""
    m = np.rot90(mask[:, :, z], C.ROT90)
    v = np.rot90(vol[:, :, z], C.ROT90).astype(float)
    v[~m] = np.nan
    r, c = np.where(m)
    return v[max(r.min() - margin, 0):r.max() + margin + 1, max(c.min() - margin, 0):c.max() + margin + 1]


def _rgb(v):
    """Colour FA (h, w, 3) with NaN outside -> RGBA image, white outside."""
    out = np.ones(v.shape[:2] + (4,))
    inside = np.isfinite(v).all(-1)
    out[inside, :3] = np.clip(v[inside], 0, 1)
    return out


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────
# Parameter maps
# ─────────────────────────────────────────────────────────────────────────
def fig_maps(maps, mask, z, path):
    """maps: {method: {"MD", "FA", "cFA" (nx, ny, nz, 3), "rmse", "rmse_sigma"}}; rows DETR, DTI."""
    res = C.FIT_RESIDUAL
    rng = C.DISPLAY[res] or (0.0, float(np.nanpercentile(np.concatenate([mp[res][mask] for mp in maps.values()]), 99)))
    cols = [("MD", "MD [10$^{-3}$ mm$^2$/s]", C.DISPLAY["md"], "gray"), ("FA", "FA", C.DISPLAY["fa"], "gray"),
            ("cFA", "colour FA (|x|, |y|, |z|)", None, None),
            (res, {"rmse": "fit RMSE", "rmse_sigma": "fit RMSE / σ"}[res], rng, "magma")]
    fig, axs = plt.subplots(len(maps), len(cols), figsize=(7.0, 1.9 * len(maps) + 0.6), layout="constrained",
                            squeeze=False)
    for c, (key, label, lim, cmap) in enumerate(cols):
        for r, (m, mp) in enumerate(maps.items()):
            ax = axs[r, c]
            if key == "cFA":
                ax.imshow(_rgb(_view(mp[key], mask, z)), interpolation="nearest")
                if r == len(maps) - 1:
                    ax.set_xlabel(label)
            else:
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white")
                im = ax.imshow(_view(mp[key], mask, z), cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            _off(ax)
            if c == 0:
                ax.set_ylabel(METHODS[m], fontsize=9, fontweight="bold")
        if key != "cFA":
            fig.colorbar(im, ax=axs[:, c].tolist(), location="bottom", shrink=0.9, aspect=18, label=label)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────
# Diagnostics: is the in vivo input inside the training distribution?
# ─────────────────────────────────────────────────────────────────────────
def fig_diagnostics(snr, sum_w, n_c, exist, n_c_map, mask, z, thr, path):
    """SNR_b0 of the voxels with an MP-PCA estimate vs. training; sum of the raw predicted weights of the kept
    compartments (each weight is a sigmoid in [0, 1], not normalized: training targets sum to 1); compartments per
    voxel after thresholding, as histogram and map; existence probabilities of all queries."""
    fig, axs = plt.subplots(1, 5, figsize=(10.0, 2.0), layout="constrained", width_ratios=[1, 1, 1, 1, 0.9])
    ax = axs[0]
    if np.isfinite(snr).any():
        ax.hist(snr[np.isfinite(snr)], bins=60, color="0.4")
    else:
        ax.text(0.5, 0.5, "no MP-PCA estimate", transform=ax.transAxes, ha="center", va="center")
    ax.axvline(1 / C.TRAIN_NOISE, color=C.COLORS["detr"], ls="--", label="training")
    ax.set_xlabel("SNR$_{b0}$ = S0 / σ (MP-PCA)"); ax.set_ylabel("voxels"); ax.legend(frameon=False)
    ax = axs[1]
    ax.hist(sum_w, bins=np.linspace(0, max(2.0, np.percentile(sum_w, 99.5)), 61), color="0.4")
    ax.axvline(1.0, color=C.COLORS["detr"], ls="--", label="training")
    ax.set_xlabel("Σ predicted weights (kept)"); ax.legend(frameon=False)
    ax = axs[2]
    kmax = int(np.nanmax(n_c))
    ticks = np.arange(1, kmax + 1, 1 if kmax <= 8 else 2)
    k, n = np.unique(n_c, return_counts=True)
    ax.bar(k, n, color="0.4")
    ax.set_xticks(ticks); ax.set_xlim(0.4, kmax + 0.6)
    ax.set_xlabel("compartments per voxel")
    ax = axs[3]
    ax.hist(exist.ravel(), bins=np.linspace(0, 1, 51), color="0.4", log=True)
    ax.axvline(thr, color=C.COLORS["detr"], ls="--", label="threshold")
    ax.set_xlabel("existence probability (all queries)"); ax.legend(frameon=False)
    ax = axs[4]
    cm = plt.get_cmap("viridis", kmax).copy(); cm.set_bad("white")
    im = ax.imshow(_view(n_c_map, mask, z), cmap=cm, vmin=0.5, vmax=kmax + 0.5, interpolation="nearest")
    _off(ax)
    fig.colorbar(im, ax=ax, shrink=0.9, ticks=ticks)
    ax.set_xlabel("compartments per voxel")
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────
# Pooled compartments with clusters
# ─────────────────────────────────────────────────────────────────────────
def _lambert(e):
    """Equal-area projection of the axis e (z >= 0 hemisphere): disc of radius sqrt(2), centre = z (S-I)."""
    e = np.where(e[:, 2:3] < 0, -e, e)
    k = np.sqrt(2 / (1 + e[:, 2]))
    return e[:, 0] * k, e[:, 1] * k


def _ellipse(mu, cov, n_std=1.0):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    return (vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * n_std * np.array([np.cos(t), np.sin(t)]))).T + mu


def fig_pooled(md, fa, e, f, g, centers, path, nbins=(80, 50), min_fa_dir=0.2, dir_gamma=1.0):
    """Left: compartments pooled over the volume in the MD-FA plane (signal-fraction-weighted histogram) with the
    clusters (1 sigma of their MD-FA marginal). Right: directions of the compartments with FA >= min_fa_dir
    (weight f * FA, equal-area projection) with the cluster mean directions."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained", width_ratios=[1.5, 1])
    ax = axs[0]
    H, xe, ye = np.histogram2d(md, fa, bins=nbins, range=[[0, MD_MAX], [0, 1]], weights=f)
    pcm = ax.pcolormesh(xe, ye, (H / H.max()).T, cmap="magma", norm=PowerNorm(0.5, 0, 1), rasterized=True)
    ax.set_facecolor("black")
    c_md, c_fa, c_e = centers
    scale = np.array([MD_MAX, 1.0])
    for k in range(len(c_md)):
        col = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        el = _ellipse(g["mu"][k, :2], g["cov"][k, :2, :2]) * scale
        ax.plot(el[:, 0], el[:, 1], color=col, lw=1.2)
        ax.plot(c_md[k], c_fa[k], "o", color=col, mec="w", ms=5)
        ax.annotate(str(k + 1), (c_md[k], c_fa[k]), xytext=(5, 4), textcoords="offset points", color="w",
                    fontsize=8, fontweight="bold")
    ax.set_xlim(0, MD_MAX); ax.set_ylim(0, 1)
    ax.set_xlabel("MD [10$^{-3}$ mm$^2$/s]"); ax.set_ylabel("FA")
    ax.set_title("DETR compartments, pooled")
    fig.colorbar(pcm, ax=ax, label="relative signal fraction (γ = 0.5)", shrink=0.9)
    ax = axs[1]
    sel = fa >= min_fa_dir
    X, Y = _lambert(e[sel])
    r = np.sqrt(2)
    H, xe, ye = np.histogram2d(X, Y, bins=80, range=[[-r, r], [-r, r]], weights=(f * fa)[sel])
    # each pixel in the colour-FA colour of its own direction, brightness = density
    xc = 0.5 * (xe[1:] + xe[:-1])
    PX, PY = np.meshgrid(xc, xc, indexing="ij")
    rho2 = PX ** 2 + PY ** 2
    pz = np.clip(1 - rho2 / 2, -1, 1)
    q = np.sqrt(np.clip((1 + pz) / 2, 0, 1))
    level = np.clip(H / np.percentile(H[H > 0], 99), 0, 1) ** dir_gamma      # linear; 99th percentile = full
    rgb = np.abs(np.stack([PX * q, PY * q, pz], -1)) * level[..., None]
    img = np.zeros(H.shape + (3,))
    inside = rho2 <= r * r
    img[inside] = np.clip(rgb[inside] / np.abs(np.stack([PX * q, PY * q, pz], -1))[inside].max(-1, keepdims=True), 0, 1)
    im = ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[-r, r, -r, r], interpolation="nearest")
    im.set_clip_path(plt.Circle((0, 0), r, transform=ax.transData))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), color="0.3", lw=0.8)
    for k in np.where(np.isfinite(c_e).all(1))[0]:
        if c_fa[k] < min_fa_dir:
            continue
        col = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        x, y = _lambert(c_e[k:k + 1])
        ax.plot(x, y, "o", color=col, mec="w", ms=6)
        ax.annotate(str(k + 1), (x[0], y[0]), xytext=(5, 4), textcoords="offset points", color="w", fontsize=8,
                    fontweight="bold")
    ax.set_aspect("equal"); _off(ax)
    ax.set_title(f"fibre axes, FA ≥ {min_fa_dir:g} (upper hemisphere)", pad=12)
    ax.text(r, 0, " ±x", va="center", fontsize=7); ax.text(-r, 0, "±x ", va="center", ha="right", fontsize=7)
    ax.text(0, r, "±y", ha="center", va="bottom", fontsize=7); ax.text(0, -r, "±y", ha="center", va="top", fontsize=7)
    ax.text(0, 0, "z", ha="center", va="center", fontsize=7, color="w")
    ax.text(0, -r - 0.35, "centre: axis along z; rim: axis in the xy-plane, opposite rim points\n"
                          "are the same axis; brightness: signal fraction x FA (linear)",
            ha="center", va="top", fontsize=6.5)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────
# Cluster fraction maps
# ─────────────────────────────────────────────────────────────────────────
def fig_fractions(frac, mask, z, centers, path, ncols=None):
    """frac: (nx, ny, nz, K) compartmental signal fraction maps; panel title: MD / FA of the cluster centre and a
    dot in the colour-FA colour of its mean direction."""
    K = frac.shape[-1]
    ncols = ncols or K
    nrows = int(np.ceil(K / ncols))
    h, w = _view(mask.astype(float), mask, z).shape
    pw = 1.65
    fig, axs = plt.subplots(nrows, ncols, figsize=(pw * ncols + 0.6, (pw * h / w + 0.35) * nrows),
                            layout="constrained", squeeze=False)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("white")
    c_md, c_fa, c_e = centers
    for k, ax in enumerate(axs.flat):
        _off(ax)
        if k >= K:
            ax.set_visible(False)
            continue
        im = ax.imshow(_view(frac[..., k], mask, z), cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"cluster {k + 1}: MD {c_md[k]:.2f}, FA {c_fa[k]:.2f}")
        if np.isfinite(c_e[k]).all():
            ax.plot(0.04, 0.96, "o", transform=ax.transAxes, ms=7, mec="0.3",
                    color=np.clip(np.abs(c_e[k]) * c_fa[k], 0, 1))
    fig.colorbar(im, ax=axs.ravel().tolist(), label="signal fraction", shrink=0.8)
    _save(fig, path)

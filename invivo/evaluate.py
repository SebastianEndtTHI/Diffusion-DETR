"""
Step 2: evaluation of the saved in vivo predictions (run invivo.predict first). Within the mask:
  1. noise level (MP-PCA, computed in predict.py) -> SNR_b0 vs. the training SNR
  2. detections (existence >= config.DETR_THRESHOLD), threshold sweep
  3. maps: signal-fraction-weighted MD, FA, colour FA; fit RMSE of the re-simulated signal; DTI reference on the
     same signals (WLLS, b <= 1000)
  4. clusters of the pooled compartments (weighted GMM, config.CLUSTER) and compartmental signal fraction maps
  5. figures (param_maps, diagnostics, av_spectrum_cluster[_k<n>], comp_sig_frac_maps[_k<n>]), NIfTI maps,
     results.json, results.md
"""

import json
import os

import numpy as np

from invivo import config as C
from invivo import analysis as A
from invivo import plots
from invivo.data import to_volume
from invivo.datasets import protocol_file
from invivo.tensor import dti_fit, load_protocol


def save_nifti(vol, affine, path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.nan_to_num(vol).astype(np.float32), affine), path)


def threshold_sweep(P, X, sigma, protocol):
    out = []
    for t in C.THRESHOLD_SWEEP:
        keep, n_none = A.detected(P, t)
        n_c = keep.sum(1)
        rmse = A.fit_rmse(P, keep, X, protocol)
        out.append({"threshold": t, "voxels_without_detection": n_none, "n_c_mean": float(n_c.mean()),
                    "rmse_median": float(np.median(rmse)), "rmse_sigma_median": float(np.nanmedian(rmse / sigma))})
    return out


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = lambda name: os.path.join(C.RESULTS_DIR, name)
    d = np.load(C.PREDICTIONS)
    P, X, S0, mask, affine = d["P"], d["X"], d["S0"], d["mask"], d["affine"]
    report = json.loads(str(d["report"]))
    protocol = load_protocol(protocol_file())
    z = C.DISPLAY_SLICE if C.DISPLAY_SLICE is not None else int(np.median(np.nonzero(mask)[2]))
    ks = C.CLUSTER["n_clusters"]
    ks = list(ks) if isinstance(ks, (tuple, list)) else [ks]

    # 1. noise relative to S0, as the training noise (sigma = 0.01 S0 -> SNR_b0 = 100)
    if "sigma" not in d:
        raise ValueError(f"{C.PREDICTIONS} has no noise estimate: run invivo.predict again")
    sigma = d["sigma"].astype(float)
    snr = 1 / sigma                                            # NaN where MP-PCA gave no estimate
    sigma = np.where(np.isfinite(sigma), sigma, np.nanmedian(sigma))

    # 2. detections
    keep, n_none = A.detected(P, C.DETR_THRESHOLD)
    print(f"DETR threshold {C.DETR_THRESHOLD}: {n_none} voxels without a compartment above it")
    sweep = threshold_sweep(P, X, sigma, protocol)

    # 3. maps, DETR and DTI
    vm = A.voxel_maps(P, keep)
    vm["rmse"] = A.fit_rmse(P, keep, X, protocol)
    md, fa, v1, s_dti = dti_fit(X.astype(float), *protocol)
    dti = {"MD": md, "FA": fa, "cFA": np.abs(v1) * fa[:, None],
           "rmse": np.sqrt(((X - s_dti) ** 2).mean(1))}
    vm["rmse_sigma"], dti["rmse_sigma"] = vm["rmse"] / sigma, dti["rmse"] / sigma
    maps = {"detr": {k: to_volume(v, mask) for k, v in vm.items()},
            "dti": {k: to_volume(v, mask) for k, v in dti.items()}}
    plots.fig_maps(maps, mask, z, out("param_maps"))
    plots.fig_diagnostics(snr, vm["sum_w"], vm["n_c"], P[..., 6], maps["detr"]["n_c"], mask, z, C.DETR_THRESHOLD,
                          out("diagnostics"))
    for m, mp in maps.items():
        for k, v in mp.items():
            save_nifti(v, affine, out(f"{m}_{k}.nii.gz"))

    # 4. clusters
    cmd, cfa, ce, cf, cv = A.compartments(P, keep)
    Z = A.features(cmd, cfa, ce)
    clusters = {}
    for k in ks:
        g, bic = A.fit_clusters(Z, cf, k)
        centers = A.center_params(g["mu"])
        frac = A.cluster_fractions(g, Z, cf, cv, len(P))
        suffix = f"_k{k}" if len(ks) > 1 else ""
        fvol = to_volume(frac, mask)
        plots.fig_pooled(cmd, cfa, ce, cf, g, centers, out("av_spectrum_cluster" + suffix))
        plots.fig_fractions(fvol, mask, z, centers, out("comp_sig_frac_maps" + suffix))
        save_nifti(fvol, affine, out(f"comp_sig_frac{suffix}.nii.gz"))
        clusters[k] = {"bic": bic, "centers": [
            {"MD": float(centers[0][j]), "FA": float(centers[1][j]),
             "direction": [float(x) for x in centers[2][j]], "mean_fraction": float(frac[:, j].mean())}
            for j in range(k)]}

    pct = lambda a: [float(x) for x in np.nanpercentile(a, [5, 50, 95])]
    res = {"seed": C.SEED, "voxels": int(mask.sum()), "display_slice": z, "data": report,
           "noise": {"patch_radius": C.NOISE["patch_radius"], "voxels_estimated": int(np.isfinite(snr).sum()),
                     "snr_b0_5_50_95": pct(snr), "training_snr_b0": 1 / C.TRAIN_NOISE},
           "detr_threshold": C.DETR_THRESHOLD, "voxels_without_detection": n_none, "threshold_sweep": sweep,
           "sum_w_5_50_95": pct(vm["sum_w"]),
           "n_c_histogram": {int(k): int(n) for k, n in zip(*np.unique(vm["n_c"], return_counts=True))},
           "maps": {m: {k: pct(v) for k, v in src.items() if k != "cFA"} for m, src in (("detr", vm), ("dti", dti))},
           "cluster_features": C.CLUSTER, "clusters": clusters}
    with open(out("results.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_markdown(res, out("results.md"))
    print(open(out("results.md")).read())


def write_markdown(r, path):
    f3 = lambda v: f"{v[1]:.3g} ({v[0]:.3g}-{v[2]:.3g})"
    nz, dt = r["noise"], r["data"]
    lines = [f"# In vivo results ({r['voxels']} voxels, display slice z = {r['display_slice']})", "",
             f"- Data: {dt['n_b0']} b0 volumes, {dt['n_used']} volumes matched to the protocol (angle median / max "
             f"{dt['angle_deg_median_max'][0]:.2f} / {dt['angle_deg_median_max'][1]:.2f} deg); DETR {dt['detr_s']:.1f} s "
             f"on {dt['device']}",
             f"- SNR_b0 = S0 / σ (MP-PCA, {nz['voxels_estimated']} voxels): median "
             f"{f3(nz['snr_b0_5_50_95'])}; training: {nz['training_snr_b0']:.0f}",
             f"- Existence threshold {r['detr_threshold']}; {r['voxels_without_detection']} voxels without a "
             f"compartment above it (kept their most confident one: {C.MIN_ONE_COMPARTMENT})",
             f"- Σ predicted weights of the kept compartments (training: 1): {f3(r['sum_w_5_50_95'])}",
             f"- Compartments per voxel: {r['n_c_histogram']}", "",
             "| threshold | voxels without compartment | mean compartments | median fit RMSE | median RMSE / σ |",
             "| --- | --- | --- | --- | --- |"]
    for s in r["threshold_sweep"]:
        lines.append(f"| {s['threshold']:.2f} | {s['voxels_without_detection']} | {s['n_c_mean']:.2f} | "
                     f"{s['rmse_median']:.4f} | {s['rmse_sigma_median']:.2f} |")
    lines += ["", "| map (median, 5-95 %) | DETR | DTI |", "| --- | --- | --- |"]
    for k, label in (("MD", "MD [1e-3 mm²/s]"), ("FA", "FA"), ("rmse", "fit RMSE"), ("rmse_sigma", "fit RMSE / σ")):
        lines.append(f"| {label} | {f3(r['maps']['detr'][k])} | {f3(r['maps']['dti'][k])} |")
    for k, cl in r["clusters"].items():
        lines += ["", f"| {k} clusters ({r['cluster_features']['direction']}) | MD | FA | direction (x, y, z) | "
                      f"mean fraction |", "| --- | --- | --- | --- | --- |"]
        for j, c in enumerate(cl["centers"]):
            lines.append(f"| {j + 1} | {c['MD']:.2f} | {c['FA']:.2f} | "
                         f"({', '.join(f'{x:+.2f}' for x in c['direction'])}) | {c['mean_fraction']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""
Evaluation of the synthetic test set (formerly evaluation.ipynb). Run after DETR_training.run_inference:

    python -m evaluation.evaluate_synthetic

Reads evaluation/inference_eval/{val,test}_predictions_<EXPERIMENT>*.csv, selects the existence threshold with the best
F1 on the validation set, and evaluates the test set: mAP (101-point, ND thresholds 0.05 / 0.07 / 0.1) with PR curves,
precision / recall / F1 and parameter errors, an example voxel, scatter plots after Hungarian matching, the mAP
heatmap over MD-FA and mAP by compartment size. Figures and metrics.json go to evaluation/evaluation_plots/<EXPERIMENT>/.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ── settings ──────────────────────────────────────────────────────────────
DATASET = "comp_setup_dynamic__max_n_comp_5__min_n_comp_2__noise_0.01__n_signal_values_166"
EXPERIMENT = "paper_model_" + DATASET
SAMPLE_IDX = 100                         # example voxel
ND_THRESHOLDS = (0.05, 0.07, 0.1)
MAP_EXTNC_THRESHOLD = 0.00001            # holistic PR curves: (almost) all predictions
F1_THRESHOLDS = np.round(np.arange(0.25, 0.76, 0.05), 2)
W_SMALL, W_LARGE = 0.35, 0.65            # compartment size classes for the mAP by weight

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
PROTOCOL_FILE = os.path.join(REPO_ROOT, "datagen", "diffusion_protocols", "cfin_multib_b1000.txt")
TEST_SIGNALS = os.path.join(REPO_ROOT, "datagen", "data", "test", DATASET + ".csv")
SAVE_DIR = os.path.join(EVAL_DIR, "evaluation_plots", EXPERIMENT)

MD_MAX, FA_MAX, DIRECTION_MAX = 4.0, 1.0, 180.0   # direction: degrees of deviation
MD_LABEL = r"Mean Diffusivity (MD) [$10^{-3} \, \mathrm{mm}^2/\mathrm{s}$]"


# ─────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────
def load_predictions(split):
    pattern = os.path.join(EVAL_DIR, "inference_eval", f"{split}_predictions_{EXPERIMENT}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No {split} prediction CSV found matching: {pattern}. Run run_inference.py first.")
    if len(files) > 1:
        print(f"Warning: multiple {split} files found. Using: {files[0]}")
    print(f"Loaded {split} predictions from: {files[0]}")
    return pd.read_csv(files[0])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def gt_components(row):
    """{i: md, fa, dir, weight} of the ground-truth compartments (1-based)."""
    return {i: {"md": row[f"MD_comp{i}"], "fa": row[f"FA_comp{i}"], "weight": row[f"weight_comp{i}"],
                "dir": _unit(np.array([row[f"dir_X_comp{i}"], row[f"dir_Y_comp{i}"], row[f"dir_Z_comp{i}"]]))}
            for i in range(1, int(row["n_comp"]) + 1)}


def pred_components(row, n_queries, extnc_threshold):
    """{q: md, fa, dir, weight, extnc} of the queries with existence >= extnc_threshold (1-based)."""
    out = {}
    for i in range(1, n_queries + 1):
        extnc = row[f"pred_q{i}_extnc"]
        if extnc >= extnc_threshold:
            out[i] = {"md": row[f"pred_q{i}_md"], "fa": row[f"pred_q{i}_fa"], "weight": row[f"pred_q{i}_w"],
                      "extnc": extnc,
                      "dir": _unit(np.array([row[f"pred_q{i}_dir_x"], row[f"pred_q{i}_dir_y"],
                                             row[f"pred_q{i}_dir_z"]]))}
    return out


def nd_match(p, gt_comps, nd_threshold):
    """Best ground-truth compartment of prediction p by normalized distance, if MD, FA and angle are all within
    nd_threshold of their range. Returns (gt index or None, [md_diff, fa_diff, dir_diff, w_diff, extnc])."""
    best, best_score, diffs = None, np.inf, None
    for gt_idx, g in gt_comps.items():
        md_diff = abs(p["md"] - g["md"])
        fa_diff = abs(p["fa"] - g["fa"])
        dir_diff = np.degrees(np.arccos(np.abs(np.clip(np.dot(p["dir"], g["dir"]), -1.0, 1.0))))   # undirected
        if md_diff <= nd_threshold * MD_MAX and fa_diff <= nd_threshold * FA_MAX \
                and dir_diff <= nd_threshold * DIRECTION_MAX:
            score = md_diff / MD_MAX + fa_diff / FA_MAX + dir_diff / DIRECTION_MAX
            if score < best_score:
                best, best_score = gt_idx, score
                diffs = [md_diff, fa_diff, dir_diff, abs(p["weight"] - g["weight"]), p["extnc"]]
    return best, diffs


def nd_assignment(row, n_queries, nd_threshold=0.07, extnc_threshold=0.0):
    """GT index -> list of [md_diff, fa_diff, dir_diff, w_diff, extnc] of the predictions assigned to it;
    'unassigned' -> existence scores of the predictions above the threshold without a match."""
    gt_comps = gt_components(row)
    assignment = {i: [] for i in gt_comps}
    assignment["unassigned"] = []
    for p in pred_components(row, n_queries, extnc_threshold).values():
        gt_idx, diffs = nd_match(p, gt_comps, nd_threshold)
        if gt_idx is None:
            assignment["unassigned"].append(p["extnc"])
        else:
            assignment[gt_idx].append(diffs)
    return assignment


# ─────────────────────────────────────────────────────────────────────────
# mAP (COCO-style 101-point interpolation)
# ─────────────────────────────────────────────────────────────────────────
def interpolated_precision(precisions, recalls):
    """P_interp(r) = max P(r') for r' >= r at r = 0, 0.01, ..., 1."""
    return np.array([precisions[recalls >= r].max() if (recalls >= r).any() else 0.0
                     for r in np.linspace(0.0, 1.0, 101)])


def precision_recall(preds, total_gt):
    """preds: list of (confidence, matched gt id or None). Highest-confidence match of a GT is a TP, every other
    prediction an FP. Returns (precisions, recalls) along decreasing confidence."""
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = np.zeros(len(preds))
    seen = set()
    for i, (_, gt_id) in enumerate(preds):
        if gt_id is not None and gt_id not in seen:
            tp[i] = 1
            seen.add(gt_id)
    acc_tp, acc_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recalls = acc_tp / total_gt if total_gt > 0 else np.zeros_like(acc_tp)
    return acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(float).eps), recalls


def ap101(preds, total_gt):
    if total_gt == 0:
        return np.nan
    return float(np.mean(interpolated_precision(*precision_recall(preds, total_gt))))


def calculate_map_101(df, n_queries, extnc_threshold=0.0, nd_threshold=0.07):
    preds, total_gt = [], 0
    for idx, row in df.iterrows():
        total_gt += int(row["n_comp"])
        for gt_idx, assigned in nd_assignment(row, n_queries, nd_threshold, extnc_threshold).items():
            if gt_idx == "unassigned":
                preds += [(extnc, None) for extnc in assigned]
            else:
                preds += [(p[4], f"{idx}_{gt_idx}") for p in assigned]
    precisions, recalls = precision_recall(preds, total_gt)
    return float(np.mean(interpolated_precision(precisions, recalls))), precisions, recalls


def binned_map(df, n_queries, bin_of, n_bins, extnc_threshold=0.0, nd_threshold=0.07):
    """mAP per bin. bin_of(md, fa, weight) -> bin index or None. A matched prediction counts in the bin of its GT,
    an unmatched one in the bin of its own prediction. Returns (mAP per bin, GT count per bin)."""
    preds = [[] for _ in range(n_bins)]
    n_gt = np.zeros(n_bins, int)
    for idx, row in df.iterrows():
        gt_comps = gt_components(row)
        for g in gt_comps.values():
            b = bin_of(g["md"], g["fa"], g["weight"])
            if b is not None:
                n_gt[b] += 1
        for p in pred_components(row, n_queries, extnc_threshold).values():
            gt_idx, _ = nd_match(p, gt_comps, nd_threshold)
            src = gt_comps[gt_idx] if gt_idx is not None else p
            b = bin_of(src["md"], src["fa"], src["weight"])
            if b is not None:
                preds[b].append((p["extnc"], f"{idx}_{gt_idx}" if gt_idx is not None else None))
    return np.array([ap101(preds[b], n_gt[b]) for b in range(n_bins)]), n_gt


# ─────────────────────────────────────────────────────────────────────────
# Precision / recall / F1 at a hard existence threshold
# ─────────────────────────────────────────────────────────────────────────
def calculate_exact_metrics(df, n_queries, extnc_threshold=0.5):
    TP = FP = FN = total_preds = 0
    err = np.zeros(4)                                    # md, fa, angle, weight of the TPs
    for _, row in df.iterrows():
        for gt_idx, assigned in nd_assignment(row, n_queries, extnc_threshold=extnc_threshold).items():
            total_preds += len(assigned)
            if gt_idx == "unassigned":
                FP += len(assigned)
            elif assigned:
                best = max(assigned, key=lambda x: x[4])     # highest-confidence match is the TP
                TP += 1
                err += best[:4]
                FP += len(assigned) - 1                      # further predictions of the same GT
            else:
                FN += 1
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    err = err / TP if TP else err
    return {"threshold": float(extnc_threshold), "TP": TP, "FP": FP, "FN": FN, "Precision": precision,
            "Recall": recall, "F1 Score": f1, "Avg MD Diff": err[0], "Avg FA Diff": err[1],
            "Avg Angle Diff": err[2], "Avg Weight Diff": err[3],
            "Avg Pred Comp": total_preds / len(df) if len(df) else 0.0}


# ─────────────────────────────────────────────────────────────────────────
# Hungarian matching (cost of the training loss) for the scatter plots
# ─────────────────────────────────────────────────────────────────────────
def _hungarian_match_sample(row, n_queries, n_comp, md_w=7.5, fa_w=20.0, dir_w=10.0, wt_w=0.4, exs_w=0.7):
    gt = np.array([[row[f"MD_comp{j}"], row[f"FA_comp{j}"], row[f"dir_X_comp{j}"], row[f"dir_Y_comp{j}"],
                    row[f"dir_Z_comp{j}"], row[f"weight_comp{j}"]] for j in range(1, n_comp + 1)], np.float32)
    pred = np.array([[row[f"pred_q{i}_md"], row[f"pred_q{i}_fa"], row[f"pred_q{i}_dir_x"], row[f"pred_q{i}_dir_y"],
                      row[f"pred_q{i}_dir_z"], row[f"pred_q{i}_w"], row[f"pred_q{i}_extnc"]]
                     for i in range(1, n_queries + 1)], np.float32)
    pred_n = pred[:, 2:5] / (np.linalg.norm(pred[:, 2:5], axis=1, keepdims=True) + 1e-8)
    gt_n = gt[:, 2:5] / (np.linalg.norm(gt[:, 2:5], axis=1, keepdims=True) + 1e-8)
    dir_cost = (1 - np.clip(pred_n @ gt_n.T, -1.0, 1.0)) * gt[None, :, 1]          # FA-weighted
    cost = (md_w * (pred[:, 0:1] - gt[None, :, 0]) ** 2 + fa_w * (pred[:, 1:2] - gt[None, :, 1]) ** 2
            + dir_w * dir_cost) * gt[None, :, 5]
    cost += wt_w * (pred[:, 5:6] - gt[None, :, 5]) ** 2 + exs_w * (1 - pred[:, 6])[:, None]
    return linear_sum_assignment(cost)                                               # (query, GT), 0-based


def build_matched_arrays(df, n_queries):
    """pred_matched (N, nc_max, 7) predictions in GT order, ref (N, nc_max, 6) ground truth, n_comp (N,)."""
    n_comp = df["n_comp"].astype(int).to_numpy()
    pred_matched = np.full((len(df), n_comp.max(), 7), np.nan, np.float32)
    ref = np.zeros((len(df), n_comp.max(), 6), np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for j in range(1, n_comp[i] + 1):
            ref[i, j - 1] = [row[f"MD_comp{j}"], row[f"FA_comp{j}"], row[f"dir_X_comp{j}"], row[f"dir_Y_comp{j}"],
                             row[f"dir_Z_comp{j}"], row[f"weight_comp{j}"]]
        for r, c in zip(*_hungarian_match_sample(row, n_queries, n_comp[i])):
            q = r + 1
            pred_matched[i, c] = [row[f"pred_q{q}_md"], row[f"pred_q{q}_fa"], row[f"pred_q{q}_dir_x"],
                                  row[f"pred_q{q}_dir_y"], row[f"pred_q{q}_dir_z"], row[f"pred_q{q}_w"],
                                  row[f"pred_q{q}_extnc"]]
    return pred_matched, ref, n_comp


def matched_compartments(pred_matched, ref, n_comp, extnc_threshold):
    """(sample, compartment) indices of matched compartments with existence >= extnc_threshold."""
    valid = np.arange(ref.shape[1])[None] < n_comp[:, None]
    valid &= ~np.isnan(pred_matched[..., 0]) & (np.nan_to_num(pred_matched[..., 6], nan=-1) >= extnc_threshold)
    return np.nonzero(valid)


def angular_errors(pred_matched, ref, i, j):
    p, t = pred_matched[i, j, 2:5], ref[i, j, 2:5]
    pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
    ok = (pn >= 1e-8) & (tn >= 1e-8)
    cos = np.clip((p[ok] * t[ok]).sum(1) / (pn[ok] * tn[ok]), -1.0, 1.0)
    return ok, np.degrees(np.arccos(np.abs(cos)))


# ─────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────
def _save(fig, path, fmt=None):
    fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pr_curves(df, n_queries, path):
    fig, axes = plt.subplots(1, len(ND_THRESHOLDS), figsize=(18, 5))
    for ax, nd in zip(axes, ND_THRESHOLDS):
        m, precisions, recalls = calculate_map_101(df, n_queries, MAP_EXTNC_THRESHOLD, nd)
        t = int(nd * 100)
        ax.plot(recalls, precisions, label=f"PR Curve (mAP@{t}={m:.4f})", color="blue", linewidth=1.5)
        ax.plot(np.linspace(0, 1, 101), interpolated_precision(precisions, recalls), label="101-point Interpolated",
                color="orange", linestyle="--")
        ax.set_title(f"Precision-Recall (mAP@{t})", fontsize=14)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3); ax.legend(loc="lower left")
    axes[0].set_ylabel("Precision", fontsize=12)
    fig.tight_layout()
    _save(fig, path, "pdf")


def _spectrum(comps):
    """md, fa, marker size (weight * 1000), colour-FA colour of a list of compartments."""
    return ([c["md"] for c in comps], [c["fa"] for c in comps], [c["weight"] * 1000 for c in comps],
            [tuple(np.abs(c["dir"]) * c["fa"]) for c in comps])


def _spectrum_axes(ax, title, ylim=1.0):
    ax.set_title(title)
    ax.set_xlabel(MD_LABEL); ax.set_ylabel("Fractional Anisotropy (FA)")
    ax.set_xlim(0, 4); ax.set_ylim(0, ylim)
    ax.grid(alpha=0.3)


def plot_sample(df, n_queries, idx, extnc_threshold, path, combined=True):
    """Ground truth and predictions of one voxel in the MD-FA plane (size: weight, colour: colour FA)."""
    row = df.iloc[idx]
    gmd, gfa, gw, gc = _spectrum(gt_components(row).values())
    pmd, pfa, pw, pc = _spectrum(pred_components(row, n_queries, extnc_threshold).values())
    fig, axes = plt.subplots(1, 3 if combined else 2, figsize=(15 if combined else 10, 5))
    axes[0].scatter(gmd, gfa, s=gw, c=gc, alpha=0.9, edgecolors="k")
    _spectrum_axes(axes[0], f"Ground Truth (Sample {idx})")
    axes[1].scatter(pmd, pfa, s=pw, c=pc, alpha=0.9, edgecolors="k")
    _spectrum_axes(axes[1], f"Predictions (extnc threshold: {extnc_threshold})")
    if combined:
        ax = axes[2]
        ax.scatter(gmd, gfa, s=gw, c=gc, alpha=0.3, edgecolors="black", linewidths=2, linestyle="--")
        ax.scatter(pmd, pfa, s=pw, c=pc, alpha=0.9, edgecolors="k")
        _spectrum_axes(ax, "Combined View")
        handles = [ax.scatter([], [], s=100, c="gray", alpha=0.3, edgecolors="black", linewidths=2,
                              label="Ground Truth"),
                   ax.scatter([], [], s=100, c="gray", alpha=0.9, edgecolors="k", label="Prediction")]
        ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_signal_and_ground_truth(df, df_signals, idx, path_base):
    """Input signal of one voxel (blocks per b-value) and its ground-truth spectrum, as two figures."""
    b = np.loadtxt(PROTOCOL_FILE, delimiter=",")[:, 0]
    sig = df_signals[[c for c in df_signals.columns if c.startswith("mri_sig")]].iloc[idx].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(len(sig)), sig, marker="o", markersize=3, linewidth=1)
    ax.set_title(f"Example Signal (Sample {idx})"); ax.set_ylabel("Signal Value"); ax.grid(alpha=0.3)
    _, first = np.unique(b, return_index=True)
    ticks, labels = [], []
    for k, bv in enumerate(b[np.sort(first)]):
        ind = np.where(b == bv)[0]
        ax.axvspan(ind[0] - 0.5, ind[-1] + 0.5, facecolor=("lightgray", "lightblue")[k % 2], alpha=0.5, zorder=0,
                   linewidth=0, rasterized=True)
        ticks.append(ind.mean()); labels.append(str(int(bv)))
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlabel(r"b-value ($\mathrm{s/mm}^2$)")
    fig.tight_layout()
    _save(fig, path_base + "_signal.pdf")

    gmd, gfa, gw, gc = _spectrum(gt_components(df.iloc[idx]).values())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gmd, gfa, s=gw, c=gc, alpha=0.9, edgecolors="k")
    _spectrum_axes(ax, f"Ground Truth (Sample {idx})", ylim=1.05)
    fig.tight_layout()
    _save(fig, path_base + "_gt.pdf")


def _r2(true, pred):
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - np.sum((pred - true) ** 2) / ss_tot if ss_tot > 0 else float("nan")


def _scatter(x, y, c, xlabel, ylabel, lim, path, cbar_label="True Signal Fraction", median_line=False):
    norm, cmap = plt.Normalize(0, 1), plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.subplots_adjust(left=0.13, right=0.78, top=0.92, bottom=0.13)
    ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=8, alpha=0.5, rasterized=True)
    if median_line:                                          # angular error vs FA: binned median
        bins = np.linspace(0, 1, 20)
        cen, med = zip(*[((lo + hi) / 2, np.median(y[(x >= lo) & (x < hi)]))
                         for lo, hi in zip(bins[:-1], bins[1:]) if ((x >= lo) & (x < hi)).any()])
        ax.plot(cen, med, "k-", linewidth=1.5, label="Median (binned)", zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper right")
    else:                                                    # true vs predicted: identity and R^2
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="Identity")
        ax.text(0.03, 0.97, f"$R^2 = {_r2(x, y):.4f}$", transform=ax.transAxes, fontsize=10, va="top")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.035, pos.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel(cbar_label, fontsize=10); cax.tick_params(labelsize=9)
    _save(fig, path, "svg")


def plot_map_heatmap(heatmap, xedges, yedges, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.subplots_adjust(left=0.13, right=0.78, top=0.92, bottom=0.13)
    ax.set_facecolor("black")                                # bins without ground truth (NaN)
    ax.set_box_aspect(1)
    pcm = ax.pcolormesh(*np.meshgrid(xedges, yedges), heatmap.T, cmap="viridis", vmin=0, vmax=1.0, shading="flat",
                        edgecolors="face", linewidth=0, rasterized=True)
    fig.canvas.draw()
    pos = ax.get_position()
    fig.colorbar(pcm, cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.035, pos.height])).set_label("mAP Score")
    ax.set_xlabel(MD_LABEL); ax.set_ylabel("Fractional anisotropy (FA)")
    ax.set_xlim(0.0, 4.0); ax.set_ylim(0.0, 1.0)
    _save(fig, path, "svg")


# ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = lambda name: os.path.join(SAVE_DIR, name)
    df, df_val = load_predictions("test"), load_predictions("val")
    df_signals = pd.read_csv(TEST_SIGNALS)
    n_queries = sum(c.startswith("pred_q") and c.endswith("_extnc") for c in df.columns)
    results = {"experiment": EXPERIMENT, "n_queries": n_queries}

    # mAP over the test set and PR curves
    maps = {nd: calculate_map_101(df, n_queries, MAP_EXTNC_THRESHOLD, nd)[0] for nd in ND_THRESHOLDS}
    for nd, m in maps.items():
        print(f"mAP (101-point) at ND threshold {nd}: {m:.4f}")
    print(f"Average mAP over ND thresholds: {np.mean(list(maps.values())):.4f}")
    results["mAP"] = {str(nd): m for nd, m in maps.items()}
    results["mAP_mean"] = float(np.mean(list(maps.values())))
    plot_pr_curves(df, n_queries, out(f"{EXPERIMENT}_pr_curves_map101.pdf"))

    # existence threshold with the best F1 on the validation set, applied to the test set
    # (the notebook printed the validation metrics at that threshold; both are reported here)
    val = [calculate_exact_metrics(df_val, n_queries, t) for t in F1_THRESHOLDS]
    val_best = max(val, key=lambda m: m["F1 Score"])
    best_threshold = val_best["threshold"]
    metrics = calculate_exact_metrics(df, n_queries, best_threshold)
    print(f"--- Metrics at the best validation threshold (extnc >= {best_threshold:.2f}): validation | test ---")
    for k in metrics:
        f = (lambda v: f"{v:.4f}") if isinstance(metrics[k], float) else str
        print(f"{k:<16}: {f(val_best[k]):>10} | {f(metrics[k]):>10}")
    results["best_threshold_val"] = best_threshold
    results["val_metrics"] = val_best
    results["test_metrics"] = metrics

    # example voxel
    plot_sample(df, n_queries, SAMPLE_IDX, best_threshold, out(f"{EXPERIMENT}_sample_{SAMPLE_IDX}_evaluation.pdf"))
    plot_sample(df, n_queries, SAMPLE_IDX, best_threshold,
                out(f"{EXPERIMENT}_sample_{SAMPLE_IDX}_evaluation_gt_pred.pdf"), combined=False)
    plot_signal_and_ground_truth(df, df_signals, SAMPLE_IDX, out(f"sample_{SAMPLE_IDX}_signal_and_ground_truth"))

    # scatter plots after Hungarian matching
    print("Running Hungarian matching...")
    pred_matched, ref, n_comp = build_matched_arrays(df, n_queries)
    i, j = matched_compartments(pred_matched, ref, n_comp, best_threshold)
    frac = ref[i, j, 5]
    true_md, pred_md = ref[i, j, 0], pred_matched[i, j, 0]
    _scatter(true_md, pred_md, frac, r"True MD [$10^{-3}$ mm$^2$/s]", r"Predicted MD [$10^{-3}$ mm$^2$/s]",
             np.ceil(max(true_md.max(), pred_md.max()) * 10.0) / 10.0, out(f"{EXPERIMENT}_scatter_md.svg"))
    _scatter(ref[i, j, 1], pred_matched[i, j, 1], frac, "True FA", "Predicted FA", 1.0,
             out(f"{EXPERIMENT}_scatter_fa.svg"), cbar_label="Signal Fraction (ground truth)")
    ok, ang = angular_errors(pred_matched, ref, i, j)
    print(f"Angular error — Mean: {np.mean(ang):.2f}°  Median: {np.median(ang):.2f}°")
    _scatter(ref[i, j, 1][ok], ang, frac[ok], "True FA", "Angular Error (°)", None,
             out(f"{EXPERIMENT}_scatter_angular.svg"), median_line=True)
    results["r2"] = {"MD": float(_r2(true_md, pred_md)), "FA": float(_r2(ref[i, j, 1], pred_matched[i, j, 1]))}
    results["angular_error_mean_median"] = [float(np.mean(ang)), float(np.median(ang))]

    # mAP heatmap over MD-FA (ND threshold 0.1, all predictions)
    xedges, yedges = np.linspace(0, MD_MAX, 41), np.linspace(0, FA_MAX, 21)

    def md_fa_bin(md, fa, w):
        ix, iy = np.digitize(md, xedges) - 1, np.digitize(fa, yedges) - 1
        return ix * 20 + iy if 0 <= ix < 40 and 0 <= iy < 20 else None

    heat, _ = binned_map(df, n_queries, md_fa_bin, 40 * 20, 0.0, 0.1)
    plot_map_heatmap(heat.reshape(40, 20), xedges, yedges, out(f"{EXPERIMENT}_map_heatmap.svg"))

    # mAP by compartment size
    size_bin = lambda md, fa, w: 0 if w < W_SMALL else (1 if w < W_LARGE else 2)
    print(f"--- Weight-based mAP: small w < {W_SMALL} <= medium < {W_LARGE} <= large ---")
    results["mAP_by_weight"] = {}
    for nd in ND_THRESHOLDS:
        m, n = binned_map(df, n_queries, size_bin, 3, 0.0, nd)
        m = np.nan_to_num(m)
        print(f"ND {nd}: " + " | ".join(f"{s} mAP {a:.4f} (GT {c})" for s, a, c in zip(("small", "medium", "large"), m, n)))
        results["mAP_by_weight"][str(nd)] = {s: {"mAP": float(a), "n_gt": int(c)}
                                             for s, a, c in zip(("small", "medium", "large"), m, n)}

    with open(out("metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out('metrics.json')}")


if __name__ == "__main__":
    main()

"""
All settings of the in vivo module. The scripts take no arguments:

    python -m invivo.fetch      # optional: download the default data set, write and check the protocol
    python -m invivo.predict    # DETR on the in vivo volume -> results_invivo/predictions.npz
    python -m invivo.evaluate   # maps, pooled spectrum, clusters, cluster fraction maps, results.md

Paths are relative to the repository root, so the scripts also run from PyCharm (working directory = repo root).
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROTOCOL_DIR = os.path.join(REPO_ROOT, "datagen", "diffusion_protocols")
RESULTS_DIR = os.path.join(REPO_ROOT, "results_invivo")
PREDICTIONS = os.path.join(RESULTS_DIR, "predictions.npz")
SEED = 42

# ── trained model (output of DETR_training.main_training) ───────────────
MODEL_DIR = os.path.join(REPO_ROOT, "DETR_training", "experiments", "paper_model",
                         "comp_setup_dynamic__max_n_comp_5__min_n_comp_2__noise_0.01__n_signal_values_166")
MODEL_FILE = "detr_model_best"            # or "detr_model_last"
# protocol the model was trained on (datagen --diff_file, default: the CFIN protocol); predict.py stops if PROTOCOL
# differs from it
TRAIN_PROTOCOL = os.path.join(REPO_ROOT, "datagen", "diffusion_protocols", "cfin_multib_b1000.txt")
TRAIN_NOISE = 0.01                        # sigma / S(b=0) of the training data, i.e. SNR_b0 = 100

# ── in vivo data and protocol ───────────────────────────────────────────
# SOURCE "cfin_multib": human in vivo data of Hansen & Jespersen (Sci. Data 2016; refs. [17, 18] of the paper),
#   downloaded by dipy to ~/.dipy/cfin_multib on the first run. "files": INVIVO_FILES, 4D NIfTI (or .mat with one 4D
#   array) and FSL bval / bvec (3 x N or N x 3).
# PROTOCOL "cfin_multib": one b = 0 entry plus all CFIN volumes with b <= PROTOCOL_B_MAX, sorted by b, written to
#   CFIN_PROTOCOL_FILE on first use (datagen.main_datagen uses the same file by default); or the
#   path of a protocol file (rows b, gx, gy, gz) for own data, then train on it as well.
# The data volumes are matched one-to-one to the protocol entries (b within B_TOL, direction up to sign within
# ANGLE_TOL_DEG); all b = 0 volumes are averaged to S0 and every signal is divided by S0, as the training signals
# (S(b=0) = 1). MASK: NIfTI / .mat brain mask, or None = dipy median_otsu on S0. SLICES: None = all, or
# (first, last) z indices, 0-based as in FSLeyes / nibabel.
SOURCE = "cfin_multib"
INVIVO_FILES = {"dwi": os.path.join(REPO_ROOT, "invivo_data", "dwi.nii.gz"),
                "bval": os.path.join(REPO_ROOT, "invivo_data", "dwi.bval"),
                "bvec": os.path.join(REPO_ROOT, "invivo_data", "dwi.bvec")}
MASK = None
SLICES = None
PROTOCOL = "cfin_multib"
PROTOCOL_B_MAX = 1000.0
CFIN_PROTOCOL_FILE = TRAIN_PROTOCOL
B_TOL = 50.0                              # s/mm^2
B0_MAX = 50.0                             # volumes with b <= B0_MAX count as b = 0
ANGLE_TOL_DEG = 5.0

# ── DETR detections ─────────────────────────────────────────────────────
# existence threshold: best validation F1 of the paper (0.35). The sweep in results.md shows how much the in vivo
# result depends on it. Every training voxel has >= 2 compartments; with MIN_ONE_COMPARTMENT a voxel without any
# query above the threshold keeps its most confident one.
DETR_THRESHOLD = 0.35
THRESHOLD_SWEEP = (0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
MIN_ONE_COMPARTMENT = True
BATCH_SIZE = 4096

# ── noise: MP-PCA (dipy) on the raw volumes in predict.py ──────────────
# patches of (2r+1)^3 voxels, more than the 166 volumes; voxels within r of the volume border get the median sigma
# (RMSE / sigma only; the SNR histogram shows the estimated voxels)
NOISE = {"patch_radius": 3}

# ── clustering of the pooled DETR compartments ──────────────────────────
# Weighted GMM (weights = signal fraction) on features scaled to about [0, 1]:
#   MD / 4, FA, and the direction, multiplied by FA (direction of an isotropic compartment carries no information):
#   "axis":   FA * (x^2, y^2, z^2, sqrt2 xy, sqrt2 xz, sqrt2 yz) / sqrt2, the outer product e e^T; antipodally symmetric
#             (e and -e identical), no pole, no wrap-around; distance = FA * sin(angle) for equal FA
#   "angles": FA * (theta / 90 deg, phi / 180 deg) with e on the z >= 0 hemisphere; phi wraps around and is undefined
#             at the pole (fibres along z), kept for comparison
#   "none":   MD and FA only
# DIR_WEIGHT scales the direction block against MD and FA. n_clusters: number or tuple (one set of figures each).
CLUSTER = {"n_clusters": 6, "direction": "axis", "dir_weight": 1.0, "n_init": 3, "n_fit": 100_000, "min_std": 0.03}

# ── figures ─────────────────────────────────────────────────────────────
DISPLAY_SLICE = None                      # z index of the axial slice in all maps; None = middle of the mask
ROT90 = 1                                 # np.rot90(slice, ROT90): 1 shows a NIfTI axial slice anterior up
DISPLAY = {"md": (0.0, 3.0), "fa": (0.0, 1.0), "rmse": None, "rmse_sigma": (0.0, 3.0)}   # None: 0 to 99th pct
FIT_RESIDUAL = "rmse"                     # or "rmse_sigma": over the local MP-PCA noise level
COLORS = {"detr": "#0065BD", "dti": "#E37222"}

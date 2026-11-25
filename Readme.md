# Diffusion-DETR
Official code implementation for the paper **"Fiber-Resolved Microstructure Quantification
from Multi-Shell Diffusion MRI using Detection Transformers"**

## Repo structure
This repository is organized into three main sections: data generation, model training and inference, and evaluation.

### `datagen/`
The data generation pipeline that creates synthetic pairwise datasets containing an overall signal curve and the corresponding MD/FA spectrum with the individual compartments.
* **`diffusion_protocols/`**: Directory, where the acquisition schedule(s) are stored. The default protocol `cfin_multib_b1000.txt` (b ≤ 1000 s/mm² of the in vivo data of Hansen & Jespersen 2016) is derived from dipy's `cfin_multib` download on first use.
* **`data_generation_pipeline.py`**: Implementation of the data generation pipeline that synthetically creates a dataset for experimentation.

### `DETR_training/`
The deep learning training pipeline, utilizing a Detection Transformer (DETR) architecture mapping the decay curves to spectra components.
  * **`dl_models.py`**, **`train_utils.py`**: Model architecture definitions and standard training loop utilities.
  * **`match_loss.py`**: Implementations of custom loss functions and Hungarian matching.
  * **`run_inference.py`**: Script for model inference: Applies the model to each sample of the test dataset and stores predictions into a csv file to be used by the evaluation script.

### `evaluation/`
Evaluation framework calculating relevant metrics and creating visual evaluation plots.
* **`evaluate_synthetic.py`**: Evaluation of the synthetic test set: metrics (mAP, precision / recall / F1, parameter errors) and plots, saved to `evaluation/evaluation_plots/<experiment>/` (figures and `metrics.json`).



## How to use

### Installing required packages
All necessary packages are found in the requirements.txt file and need to be installed into the Python environment.


### Data generation
The data generation pipeline has to be executed three times, once for creating a train, validation and test dataset. The commands are as follows:

```python -m datagen.main_datagen --n_samples 200000 --seed 42 --save_folder_path "datagen/data/train/"```

```python -m datagen.main_datagen --n_samples 25000 --seed 43 --save_folder_path "datagen/data/validation/"```

```python -m datagen.main_datagen --n_samples 25000 --seed 44 --save_folder_path "datagen/data/test/"```


### Model training
All the default parameters are set as used for the experiment described in the paper.

```python -m DETR_training.main_training```


### Model inference
The inference script does not require any special parameters and can be run by:

```python -m DETR_training.run_inference```


### Evaluation
The evaluation is done by `python -m evaluation.evaluate_synthetic`. It reads the csv files produced by the inference script, selects the existence threshold with the best F1 on the validation set, and computes metrics and plots for the test set.


## In vivo (`invivo/`)
Applies a trained model to an in vivo volume and evaluates it without ground truth. All settings live in
`invivo/config.py`; the scripts take no arguments (run from the repository root, e.g. via PyCharm's run button):

```python -m invivo.fetch```     – optional: download the data, write the protocol, compare it with the training protocol

```python -m invivo.predict```   – DETR on the volume, saves `results_invivo/predictions.npz`

```python -m invivo.evaluate```  – maps, diagnostics, pooled spectrum, clustering, fraction maps

**Default data**: the human in vivo multi-shell data set of Hansen & Jespersen (Sci. Data 2016, refs. [17, 18] of the
paper), downloaded by dipy (`cfin_multib`, to `~/.dipy/cfin_multib`) on the first run, so the whole pipeline runs from
this repository alone. The input protocol is derived from it (`PROTOCOL = "cfin_multib"`: one b = 0 entry plus all
volumes with b ≤ 1000 s/mm², sorted by b, written to `datagen/diffusion_protocols/cfin_multib_b1000.txt`).
The data generation uses the same file by default (`--diff_file cfin_multib`), so training data and in vivo input
share one protocol; `predict` stops if the input protocol differs from the training protocol (`TRAIN_PROTOCOL`).

**Own data**: `SOURCE = "files"` with `INVIVO_FILES` (4D NIfTI or `.mat`, FSL `bval`/`bvec`) and `PROTOCOL` set to a
protocol file (train on it: `--diff_file <file>`, `TRAIN_PROTOCOL`); optional brain mask `MASK` (otherwise dipy's `median_otsu` on S0).

In both cases the volumes are matched one-to-one to the protocol entries (b within `B_TOL`, direction up to sign
within `ANGLE_TOL_DEG`); other shells (e.g. b = 2000) are ignored. All b0 volumes are averaged and every signal is
divided by S0, as in the training data (S(b=0) = 1).

**Outputs** (`results_invivo/`, figures as `.png` and `.pdf`, maps also as NIfTI):

| File | Content |
|---|---|
| `param_maps` | DETR: signal-fraction-weighted MD, FA, colour FA (Σ f FA \|e\|), fit RMSE of the re-simulated signal; DTI (WLLS on the same signals) as reference |
| `diagnostics` | SNR_b0 (MP-PCA) vs. training SNR, Σ predicted weights, compartments per voxel (histogram and map), existence scores: is the input inside the training distribution? |
| `av_spectrum_cluster` | compartments pooled over the volume in the MD-FA plane and their directions (equal-area projection), with the GMM clusters |
| `comp_sig_frac_maps` | compartmental signal fraction of each cluster per voxel |
| `results.md`, `.json` | all numbers, including a sweep over the existence threshold |

**Clustering**: weighted GMM (weights = signal fraction) on MD/4, FA and the direction multiplied by FA. The
direction is represented by its outer product e eᵀ (`CLUSTER["direction"] = "axis"`): antipodally symmetric and free
of the pole and wrap-around of spherical angles (`"angles"` is available for comparison, `"none"` clusters MD-FA only).

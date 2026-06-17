# Diffusion-DETR
Official code implementation for the paper **"Fiber-Resolved Microstructure Quantification
from Multi-Shell Diffusion MRI using Detection Transformers"**

## Repo structure
This repository is organized into three main sections: data generation, model training and inference, and evaluation.

### `datagen/`
The data generation pipeline that creates synthetic pairwise datasets containing an overall signal curve and the corresponding MD/FA spectrum with the individual compartments.
* **`diffusion_protocols/`**: Directory, where the acquisition schedule(s) are stored.
* **`data_generation_pipeline.py`**: Implementation of the data generation pipeline that synthetically creates a dataset for experimentation.

### `DETR_training/`
The deep learning training pipeline, utilizing a Detection Transformer (DETR) architecture mapping the decay curves to spectra components.
  * **`dl_models.py`**, **`train_utils.py`**: Model architecture definitions and standard training loop utilities.
  * **`match_loss.py`**: Implementations of custom matching losses (such as modified focal loss) and Hungarian matching.

### `evaluation/`
Evaluation framework calculating relevant metrics and creating visual evaluation plots.
* **`evaluation.ipynb`**: Central Jupyter Notebook for evaluating model performance, calculating metrics, and comparing different training run configurations.
* **`eval_data/`**: Directory where `*.csv` files containing network predictions on the test set (`eval_data/`) resulting from running `run_inference.py` are stored for the evaluation notebook.
* **`evaluation_plots_final/`**: The rendered, final evaluation plots organized by experiment and loss configuration (e.g. `focal_loss`, `mapeweight`, `angular_loss`).



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
The evaluation is done via a jupyter notebook that can be executed cell-by-cell. It accesses the output csv files produced by the inference script to compute relevant metrics and create plots to provide visual insights.

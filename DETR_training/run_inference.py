import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
import DETR_training.dl_models as models

def get_data(csv_path, b_size=256):
    """
    Loads data from a CSV, returning the X data wrapped in a DataLoader and
    the ground truth target columns in a DataFrame.
    """
    df = pd.read_csv(csv_path).fillna(0)

    # Extracting only the target columns (ground truth)
    gt_df = df.filter(like="comp")

    # Extracting MRI sequence for model input
    X_data = df.filter(like="mri_sig").to_numpy(dtype=np.float32)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

    return dataloader, gt_df

def run_inference_on_split(model, dataloader, gt_df, device, output_csv):
    """
    Runs inference on a single data split and saves predictions to a CSV.
    """
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            out = model(X)

            # If aux loss was active, out is a dict with 'pred' as the final layer output
            if isinstance(out, dict):
                out = out['pred']

            # Apply sigmoid to existence scores (index 6) to convert logits to probabilities
            out[:, :, 6] = torch.sigmoid(out[:, :, 6])

            all_preds.append(out.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)  # (N, n_queries, 7)

    # Flatten predictions for the CSV
    n_queries = preds_np.shape[1]
    pred_cols = []
    for q in range(n_queries):
        pred_cols.extend([
            f"pred_q{q+1}_md",
            f"pred_q{q+1}_fa",
            f"pred_q{q+1}_dir_x",
            f"pred_q{q+1}_dir_y",
            f"pred_q{q+1}_dir_z",
            f"pred_q{q+1}_w",
            f"pred_q{q+1}_extnc"
        ])

    flattened_preds = preds_np.reshape(preds_np.shape[0], -1)
    preds_df = pd.DataFrame(flattened_preds, columns=pred_cols)

    result_df = pd.concat([gt_df, preds_df], axis=1)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

def main():
    parser = argparse.ArgumentParser('DETR Inference Job')
    parser.add_argument('--b_size', default=256, type=int, help="Batch size for inference")
    job_args = parser.parse_args()

    experiments_dir = ".\\DETR_training\\experiments"
    test_data_dir = ".\\datagen\\data\\test"
    val_data_dir = ".\\datagen\\data\\validation"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        for dataset_base in os.listdir(exp_path):
            ds_path = os.path.join(exp_path, dataset_base)
            if not os.path.isdir(ds_path):
                continue

            config_path = os.path.join(ds_path, "detr_config.json")
            model_path = os.path.join(ds_path, "detr_model_last")

            if not os.path.exists(config_path) or not os.path.exists(model_path):
                print(f"Skipping {ds_path} (missing config or model_best)")
                continue

            test_csv_path = os.path.join(test_data_dir, f"{dataset_base}.csv")
            val_csv_path = os.path.join(val_data_dir, f"{dataset_base}.csv")

            if not os.path.exists(test_csv_path):
                print(f"Test data not found for {dataset_base} in {test_data_dir}")
                continue

            if not os.path.exists(val_csv_path):
                print(f"Validation data not found for {dataset_base} in {val_data_dir}")
                continue

            print(f"Running inference for {exp_name} on {dataset_base}")

            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Create a mock args object from the config dictionary
            class Args:
                pass
            args = Args()
            for k, v in config_dict.items():
                setattr(args, k, v)

            # Override specifics for inference
            args.device = device

            inference_eval_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "evaluation", "inference_eval")
            )
            os.makedirs(inference_eval_dir, exist_ok=True)

            # Model initialization (load once, run on both splits)
            model = models.DWI_DETR(args)
            models.load_state_dict_compat(
                model,
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model.to(device)
            model.eval()

            # --- Test split ---
            test_dataloader, test_gt_df = get_data(test_csv_path, b_size=job_args.b_size)
            test_output_csv = os.path.join(
                inference_eval_dir, f"test_predictions_{exp_name}_{dataset_base}.csv"
            )
            run_inference_on_split(model, test_dataloader, test_gt_df, device, test_output_csv)

            # --- Validation split (for threshold selection) ---
            val_dataloader, val_gt_df = get_data(val_csv_path, b_size=job_args.b_size)
            val_output_csv = os.path.join(
                inference_eval_dir, f"val_predictions_{exp_name}_{dataset_base}.csv"
            )
            run_inference_on_split(model, val_dataloader, val_gt_df, device, val_output_csv)

if __name__ == '__main__':
    main()

"""
Step 1: DETR on the in vivo volume (config.SOURCE; default: CFIN multi-b data, downloaded by dipy). Loads the data, matches the volumes to the training protocol,
normalizes by S0, runs the trained model and saves everything the evaluation needs to config.PREDICTIONS:
  P (V, Q, 7) [MD, FA, dir_x, dir_y, dir_z, weight, existence probability], X (V, 166) b0-normalized input,
  S0 (V,), sigma / S0 (V,, MP-PCA), mask, affine, matching report, runtime.
"""

import json
import os
import time

import numpy as np
import torch

import DETR_training.dl_models as models
from invivo import config as C
from invivo.data import load_invivo
from invivo.datasets import invivo_files, protocol_file
from invivo.fetch import compare
from invivo.tensor import load_protocol


def load_model(device):
    with open(os.path.join(C.MODEL_DIR, "detr_config.json")) as f:
        args = type("Args", (), json.load(f))()
    args.device = device
    model = models.DWI_DETR(args)
    models.load_state_dict_compat(model, torch.load(os.path.join(C.MODEL_DIR, C.MODEL_FILE), map_location=device,
                                                    weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, X, device, batch_size=C.BATCH_SIZE):
    out = []
    for i in range(0, len(X), batch_size):
        o = model(torch.from_numpy(X[i:i + batch_size]).to(device))
        o = o["pred"] if isinstance(o, dict) else o
        o[..., 6] = torch.sigmoid(o[..., 6])
        out.append(o.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    protocol = load_protocol(protocol_file())
    same, how = compare(protocol_file(), C.TRAIN_PROTOCOL)
    if not same:
        raise ValueError(f"input protocol {protocol_file()} differs from the training protocol {C.TRAIN_PROTOCOL} "
                         f"({how}): generate the training data with datagen --diff_file {protocol_file()} and set "
                         f"TRAIN_PROTOCOL to it")
    model = load_model(device)
    d = load_invivo(protocol, invivo_files())
    predict(model, d["X"][:256], device)                                   # warm-up
    t = time.perf_counter()
    P = predict(model, d["X"], device)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f"DETR: {t:.1f} s for {len(P)} voxels on {device} ({1e3 * t / len(P):.3f} ms per voxel)")
    np.savez_compressed(C.PREDICTIONS, P=P, X=d["X"], S0=d["S0"], sigma=d["sigma"], mask=d["mask"], affine=d["affine"],
                        report=json.dumps({**d["report"], "detr_s": t, "device": device,
                                           "model": os.path.join(C.MODEL_DIR, C.MODEL_FILE),
                                           "protocol": protocol_file(), "data": invivo_files()[0]}))
    print(f"Saved: {C.PREDICTIONS}")


if __name__ == "__main__":
    main()

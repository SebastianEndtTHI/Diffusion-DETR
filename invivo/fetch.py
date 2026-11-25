"""
Optional step 0: download the in vivo data (config.SOURCE), write the protocol (config.PROTOCOL) and compare it with
the protocol the model was trained on (config.TRAIN_PROTOCOL). predict.py does the first two on its own if needed.
"""

import os

import numpy as np

from invivo import config as C
from invivo.datasets import invivo_files, protocol_file
from invivo.tensor import load_protocol


def compare(path_a, path_b):
    """Entry-by-entry comparison of two protocols. Returns (same within B_TOL / ANGLE_TOL_DEG, description)."""
    (ba, ga), (bb, gb) = load_protocol(path_a), load_protocol(path_b)
    if len(ba) != len(bb):
        return False, f"{len(ba)} vs. {len(bb)} entries"
    ang = np.degrees(np.arccos(np.clip(np.abs((ga * gb).sum(1)), 0, 1)))[ba > C.B0_MAX]
    db = np.abs(ba - bb).max()
    return (db <= C.B_TOL and ang.max() <= C.ANGLE_TOL_DEG,
            f"{len(ba)} entries, max |b difference| {db:g} s/mm^2, max direction difference {ang.max():.3f} deg "
            f"(entry by entry, up to sign)")


def main():
    files = invivo_files()
    print("Data:", *files, sep="\n  ")
    p = protocol_file()
    same, how = compare(p, C.TRAIN_PROTOCOL)
    print(f"Protocol {os.path.basename(p)} vs. training protocol {os.path.basename(C.TRAIN_PROTOCOL)}: {how} -> "
          f"{'same' if same else 'DIFFERENT: retrain with datagen --diff_file ' + p}")


if __name__ == "__main__":
    main()

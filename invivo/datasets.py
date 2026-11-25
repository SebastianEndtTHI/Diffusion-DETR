"""
In vivo source and protocol. Default: the CFIN multi-b human brain data set of Hansen & Jespersen (Sci. Data 2016,
doi:10.1038/sdata.2016.72; paper refs. [17, 18]), downloaded by dipy ("cfin_multib", ~/.dipy/cfin_multib) on first
use. The training protocol is derived from it: one b = 0 entry plus all volumes with b <= PROTOCOL_B_MAX, sorted by b.

    python -m invivo.fetch    # download, write the protocol file, compare it with the training protocol
"""

import os

import numpy as np

from invivo import config as C


def invivo_files():
    """(dwi, bval, bvec) of config.SOURCE: dipy's cfin_multib (downloaded if missing) or config.INVIVO_FILES."""
    if C.SOURCE == "files":
        return C.INVIVO_FILES["dwi"], C.INVIVO_FILES["bval"], C.INVIVO_FILES["bvec"]
    if C.SOURCE != "cfin_multib":
        raise ValueError(f"unknown SOURCE {C.SOURCE!r}")
    return cfin_files()


def cfin_files():
    from dipy.data import get_fnames
    dwi, bval, bvec, _ = get_fnames(name="cfin_multib")
    return str(dwi), str(bval), str(bvec)


def protocol_from_gradients(b, g, b_max=C.PROTOCOL_B_MAX, b0_max=C.B0_MAX):
    """Protocol table (N, 4) [b, gx, gy, gz]: one b = 0 row, then all volumes with b0_max < b <= b_max, sorted by b
    (stable: acquisition order within a shell)."""
    dw = np.where((b > b0_max) & (b <= b_max))[0]
    dw = dw[np.argsort(b[dw], kind="stable")]
    return np.vstack([[0, 0, 0, 0], np.column_stack([np.round(b[dw], -1), g[dw]])])


def protocol_file():
    """Path of the input protocol per config.PROTOCOL."""
    return cfin_protocol_file() if C.PROTOCOL == "cfin_multib" else C.PROTOCOL


def cfin_protocol_file():
    """CFIN protocol file (training data and in vivo input), written from the dipy download on first use."""
    if not os.path.exists(C.CFIN_PROTOCOL_FILE):
        from invivo.data import load_gradients
        _, bval, bvec = cfin_files()
        t = protocol_from_gradients(*load_gradients(bval, bvec))
        os.makedirs(os.path.dirname(C.CFIN_PROTOCOL_FILE), exist_ok=True)
        np.savetxt(C.CFIN_PROTOCOL_FILE, t, delimiter=",", fmt=["%g", "%.5f", "%.5f", "%.5f"])
        print(f"Protocol: {len(t)} entries (b <= {C.PROTOCOL_B_MAX:g}) written to {C.CFIN_PROTOCOL_FILE}")
    return C.CFIN_PROTOCOL_FILE


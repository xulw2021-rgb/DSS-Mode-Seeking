"""
PanTaGruEl (Zenodo 5816149) -> project pickle format converter.

Input : HDF5 file, e.g. dataset_14_case_1.h5
        keys: V, theta, P, Q (N_samples x N_bus, per-unit on 100 MVA),
              epsilon (2 x E, 1-indexed line pairs), g, b (1 x E, pu),
              gsh, bsh (1 x N_bus, pu), id_slack, idgen (1-indexed)
Output: pickles {nodes, edges, labels, bus_param, edge_param, noise_param}
        in the exact format consumed by data.data_from_pickles.

Unit convention: per-unit quantities are converted to "MW" by multiplying
by S_base = 100.  All bus base voltages are set to vn_kv = 10.0 so that the
internal pu->MW factor V_lv^2 = 100 used by data.get_pflow coincides with
S_base: flows computed by get_pflow are then exactly the 100 MVA-base
per-unit flows times 100, consistent with the stored p_mw / p_from_mw.
(This is documented in the dataset README produced alongside the pickles.)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import h5py

# ---- noise model defaults (identical to toy_network.py constants) ----------
P_NOISE = 0.02
V_NOISE = 0.01
I_NOISE = 0.01
ZERO_INJ_COEF = 0.001
PM_NOISE = 0.15
SGEN_NOISE = 0.125

S_BASE = 100.0          # MVA base of the PanTaGruEl per-unit system
VN_KV = 10.0            # nominal kV: makes V_lv^2 == S_BASE in get_pflow
N_SAMPLES_DEFAULT = 2100

# IEEE-14 (MATPOWER case14) load buses, 1-indexed
IEEE14_LOAD_BUSES = {2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14}
# measurement placement (0-indexed): V at 6 buses, flows at 3 lines
MEAS_V = [0, 1, 3, 6, 8, 11]
MEAS_PF = [0, 4, 9]


def convert(src_h5: str, out_dir: str, n_samples: int = N_SAMPLES_DEFAULT,
            case_id: str = "14_case_1"):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(src_h5, "r") as f:
        V = f["V"][:n_samples]
        theta = f["theta"][:n_samples]
        P = f["P"][:n_samples]
        Q = f["Q"][:n_samples]
        epsilon = f["epsilon"][:].astype(int)          # (2, E) 1-indexed
        g = f["g"][0]
        b = f["b"][0]
        gsh = f["gsh"][0]
        bsh = f["bsh"][0]
        id_slack = int(f["id_slack"][0, 0]) - 1        # -> 0-indexed

    n_bus = V.shape[1]
    n_edge = epsilon.shape[1]
    from_bus = epsilon[0] - 1                          # 0-indexed
    to_bus = epsilon[1] - 1

    # per-edge shunt: total bus shunt split half to each end (pi-model),
    # consistent with get_pflow's (G + Gs/2)*V_i^2 convention
    Gs = gsh[from_bus] + gsh[to_bus]
    Bs = bsh[from_bus] + bsh[to_bus]

    # edge flows in "MW" (per-unit * S_BASE), same formula as data.get_pflow
    P_from = (-V[:, from_bus] * V[:, to_bus] * (g * np.cos(theta[:, from_bus] - theta[:, to_bus])
              + b * np.sin(theta[:, from_bus] - theta[:, to_bus]))
              + (g + Gs / 2) * V[:, from_bus] ** 2) * S_BASE
    Q_from = (V[:, from_bus] * V[:, to_bus] * (-g * np.sin(theta[:, from_bus] - theta[:, to_bus])
              + b * np.cos(theta[:, from_bus] - theta[:, to_bus]))
              - (b + Bs / 2) * V[:, from_bus] ** 2) * S_BASE

    # bus params
    bus_param = pd.DataFrame(index=np.arange(n_bus))
    bus_param["vn_kv"] = VN_KV
    bus_param["bool_slack"] = 0.0
    bus_param.loc[id_slack, "bool_slack"] = 1.0
    bus_param["bool_zero_inj"] = [
        1.0 if (i + 1 not in IEEE14_LOAD_BUSES) and (i != id_slack) else 0.0
        for i in range(n_bus)
    ]

    # edge params
    edge_param = pd.DataFrame({
        "from_bus": from_bus.astype(float), "to_bus": to_bus.astype(float),
        "G": g, "B": b, "Gs": Gs, "Bs": Bs,
        "closed line": 1.0, "phase shift": 0.0, "imax or sn": float(S_BASE),
    })

    noise_param = pd.DataFrame(
        {"p_noise": P_NOISE, "v_noise": V_NOISE, "i_noise": I_NOISE,
         "PM_NOISE": PM_NOISE, "SGEN_NOISE": SGEN_NOISE,
         "zero_inj_coef": ZERO_INJ_COEF},
        index=["def_value"])

    nodes_list, edges_list, labels_list = [], [], []
    for i in range(n_samples):
        nodes = pd.concat([bus_param,
                           pd.DataFrame({"vm_pu": V[i], "va_rad": theta[i],
                                         "p_mw": P[i] * S_BASE,
                                         "q_mvar": Q[i] * S_BASE},
                                        index=np.arange(n_bus))], axis=1)
        edges = pd.concat([edge_param,
                           pd.DataFrame({"p_from_mw": P_from[i],
                                         "q_from_mvar": Q_from[i]})], axis=1)
        nodes_list.append(nodes)
        edges_list.append(edges)
        labels_list.append(nodes[["vm_pu", "va_rad"]])

    for name, obj in [("nodes", nodes_list), ("edges", edges_list),
                      ("labels", labels_list), ("bus_param", bus_param),
                      ("edge_param", edge_param), ("noise_param", noise_param)]:
        with open(os.path.join(out_dir, name), "wb") as fo:
            import pickle
            pickle.dump(obj, fo)

    meta = {
        "source_file": os.path.abspath(src_h5),
        "source": "Zenodo 10.5281/zenodo.5816149 (PanTaGruEl), CC-BY-4.0",
        "case_id": case_id,
        "n_samples": n_samples,
        "n_bus": n_bus, "n_edge": n_edge,
        "id_slack_0idx": id_slack,
        "meas_v": MEAS_V, "meas_pf": MEAS_PF,
        "unit_note": ("per-unit values on 100 MVA base converted to MW by x100; "
                      "vn_kv set to 10.0 so get_pflow's V_lv^2 factor equals S_BASE"),
        "noise_param": {"p_noise": P_NOISE, "v_noise": V_NOISE,
                        "PM_NOISE": PM_NOISE, "SGEN_NOISE": SGEN_NOISE},
    }
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as fo:
        json.dump(meta, fo, indent=2)
    print(f"[convert_pantagruel] wrote {out_dir}: {n_samples} samples, "
          f"{n_bus} buses, {n_edge} edges")
    print(f"  V range {V.min():.3f}-{V.max():.3f}, theta range "
          f"{theta.min():.3f}-{theta.max():.3f}, P range {P.min():.2f}-{P.max():.2f}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="external/pantagruel/dataset_14_case_1.h5",
                    help="path to the PanTaGruEl HDF5 file (download from Zenodo 10.5281/zenodo.5816149; see docs/DATASETS.md)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    ap.add_argument("--case-id", default="14_case_1")
    args = ap.parse_args()
    convert(args.src, args.out_dir, args.n_samples, args.case_id)

"""
IEEE-30 data generation for E6-A (cross-system benchmark).

Adaptations vs the CIGRE pipeline (E6_IEEE30_ADAPTATION_LOG.md):
  1. bool_slack = bus connected to ext_grid (bus 0), NOT the max-voltage
     heuristic in data.get_bus_param (which would mark all 30 buses).
  2. No transformers in case30 -> no trafo shift handling.
  3. Load MC sampling: per-timestep independent draws N(p_mw, 0.15*p_mw)
     around the nominal pandapower case30 loads (flat sampling, no daily
     profiles); same noise-model constants as toy_network.py.
  4. Measurement placement chosen for this network: meas_v = 9 buses,
     meas_pf = 6 lines (documented in the adaptation log).

Output: Axioms_revision/experiments/data/ieee30/{nodes,edges,labels,
bus_param,edge_param,noise_param} + dataset_meta.json — the exact pickle
format consumed by data.data_from_pickles (unchanged core pipeline).
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import importlib.metadata as _im
_orig_d = _im.distribution
_im.distribution = lambda pkg: (type("FakeDist", (), {"version": np.__version__})()
                                if pkg == "numpy" else _orig_d(pkg))

import pandapower as pp

# noise-model constants identical to toy_network.py
P_NOISE, V_NOISE, I_NOISE = 0.02, 0.01, 0.01
ZERO_INJ_COEF = 0.001
PM_NOISE, SGEN_NOISE = 0.15, 0.125
POWER_COEF = 0.9

ITERATIONS = 90          # days x 24 h = 2160 scenarios
MEAS_V = [0, 2, 5, 9, 14, 19, 24, 27, 29]     # 9 voltage-measured buses
MEAS_PF = [0, 4, 8, 12, 16, 20]               # 6 flow-measured lines
SEED = 12345


def get_bus_param_ieee(net):
    """bus params with the ext_grid-based slack fix (adaptation #1)."""
    bp = pd.DataFrame(index=net.bus.index)
    bp["vn_kv"] = net.bus["vn_kv"]
    bp["bool_slack"] = 0.0
    for b in net.ext_grid.bus.values:
        bp.loc[b, "bool_slack"] = 1.0
    bp["bool_zero_inj"] = 0.0
    for i in net.bus.index:
        if i not in net.load["bus"].values and bp.loc[i, "bool_slack"] == 0.0:
            bp.loc[i, "bool_zero_inj"] = 1.0
    return bp


def get_edge_param_ieee(net):
    """line-only edge params (adaptation #2: no trafos)."""
    edge_r = net.line["r_ohm_per_km"] * net.line["length_km"]
    edge_x = net.line["x_ohm_per_km"] * net.line["length_km"]
    edge_c = net.line["c_nf_per_km"] * net.line["length_km"]
    edge_b = -2 * np.pi * net.f_hz * edge_c * 1e-9
    edge_g = net.line["g_us_per_km"] * net.line["length_km"] * 1e-6
    edge_Z = edge_r.values + 1j * edge_x.values
    edge_Y = np.reciprocal(edge_Z)
    edge_Ys = edge_g.values - 1j * edge_b.values
    bus_renum = {b: i for i, b in enumerate(net.bus.index)}
    ep = pd.DataFrame({
        "from_bus": net.line["from_bus"].map(bus_renum).astype(float).values,
        "to_bus": net.line["to_bus"].map(bus_renum).astype(float).values,
        "G": np.real(edge_Y), "B": np.imag(edge_Y),
        "Gs": np.nan_to_num(np.real(edge_Ys)),
        "Bs": np.nan_to_num(np.imag(edge_Ys)),
        "closed line": 1.0, "phase shift": 0.0,
        "imax or sn": net.line["max_i_ka"].values,
    })
    return ep


def generate(out_dir, n_days=ITERATIONS, seed=SEED):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    net = pp.networks.case30()
    n_bus = len(net.bus)

    bus_param = get_bus_param_ieee(net)
    edge_param = get_edge_param_ieee(net)
    noise_param = pd.DataFrame(
        {"p_noise": P_NOISE, "v_noise": V_NOISE, "i_noise": I_NOISE,
         "PM_NOISE": PM_NOISE, "SGEN_NOISE": SGEN_NOISE,
         "zero_inj_coef": ZERO_INJ_COEF}, index=["def_value"])

    load_p0 = net.load["p_mw"].values.copy()
    load_q0 = net.load["q_mvar"].values.copy()
    n_ts = 24 * n_days

    nodes_list, edges_list, labels_list = [], [], []
    t0 = time.time()
    for t in range(n_ts):
        net.load["p_mw"] = np.maximum(rng.normal(load_p0, PM_NOISE * np.abs(load_p0)), 0.0)
        net.load["q_mvar"] = net.load["p_mw"] * POWER_COEF * np.sign(load_q0)
        try:
            pp.runpp(net, calculate_voltage_angles=True, numba=False)
        except pp.LoadflowNotConverged:
            print(f"  WARNING: power flow did not converge at t={t}; skipping")
            continue
        nodes = pd.concat([bus_param,
                           pd.DataFrame({"vm_pu": net.res_bus["vm_pu"].values,
                                         "va_rad": (net.res_bus["va_degree"] * np.pi / 180).values,
                                         "p_mw": net.res_bus["p_mw"].values,
                                         "q_mvar": net.res_bus["q_mvar"].values},
                                        index=np.arange(n_bus))], axis=1)
        res_line = net.res_line
        edges = pd.concat([edge_param,
                           pd.DataFrame({"p_from_mw": res_line["p_from_mw"].values,
                                         "q_from_mvar": res_line["q_from_mvar"].values})],
                          axis=1)
        nodes_list.append(nodes)
        edges_list.append(edges)
        labels_list.append(nodes[["vm_pu", "va_rad"]])
        if t % 500 == 0:
            print(f"  t={t}/{n_ts} ({time.time()-t0:.0f}s)", flush=True)

    for name, obj in [("nodes", nodes_list), ("edges", edges_list),
                      ("labels", labels_list), ("bus_param", bus_param),
                      ("edge_param", edge_param), ("noise_param", noise_param)]:
        with open(os.path.join(out_dir, name), "wb") as f:
            pickle.dump(obj, f)

    meta = {
        "source": "pandapower case30 (standard IEEE 30-bus benchmark)",
        "n_samples": len(nodes_list), "n_bus": n_bus,
        "n_edges": len(edge_param),
        "slack_buses": [int(b) for b in net.ext_grid.bus.values],
        "meas_v": MEAS_V, "meas_pf": MEAS_PF,
        "adaptations": ["bool_slack from ext_grid (not max-voltage)",
                        "no trafos in case30; line-only edge params",
                        "flat per-timestep MC load sampling N(p, 0.15p)",
                        "q = 0.9*p*sign(q0)"],
        "seed": seed,
    }
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ieee30] wrote {out_dir}: {len(nodes_list)} samples, "
          f"{n_bus} buses, {len(edge_param)} edges")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(EXP, "data", "ieee30"))
    ap.add_argument("--n-days", type=int, default=ITERATIONS)
    args = ap.parse_args()
    generate(args.out_dir, args.n_days)

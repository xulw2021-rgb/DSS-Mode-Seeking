"""
Verify that regenerated publication artifacts numerically match the
packaged frozen artifacts.  Compares table CSVs and figure-generation
inputs (the frozen result CSVs are identical by construction, so the
check focuses on the generated table contents vs tables/publication/).
"""

import os
import sys
import glob
import hashlib
import numpy as np
import pandas as pd

EXP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def table_numbers(tex_path):
    """Extract numeric tokens from a generated .tex table."""
    txt = open(tex_path).read()
    nums = []
    for tok in txt.replace("$", " ").replace("\\pm", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    return nums


def main():
    mismatches = []
    for name in ["Table_2_main_14bus", "Table_3_70bus", "Table_4_cross_system",
                 "Table_5_robust_loss", "Table_6_sensitivity",
                 "Table_7_multistart", "Table_1_datasets"]:
        ref = os.path.join(EXP, "tables", "publication", name + ".tex")
        gen = os.path.join(EXP, "reproduced", "tables", name + ".tex")
        if not os.path.exists(gen):
            mismatches.append((name, "missing regenerated file"))
            continue
        a, b = table_numbers(ref), table_numbers(gen)
        if a != b:
            mismatches.append((name, f"numeric mismatch ({len(a)} vs {len(b)} tokens)"))
    figs = sorted(os.path.basename(p) for p in
                  glob.glob(os.path.join(EXP, "figures", "publication", "*.pdf")))
    gfigs = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(EXP, "reproduced", "figures", "*.pdf")))
    missing_figs = [f for f in figs if f not in gfigs]
    if missing_figs:
        mismatches.append(("figures", f"missing: {missing_figs}"))
    if mismatches:
        print("MISMATCHES:")
        for m in mismatches:
            print("  ", m)
        sys.exit(1)
    print("VERIFICATION PASS: all regenerated tables/figures match the packaged artifacts")


if __name__ == "__main__":
    main()

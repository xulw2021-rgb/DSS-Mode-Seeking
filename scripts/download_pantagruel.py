"""
Download helper for the PanTaGruEl external benchmark.

Source : Zenodo record 5816149 (CC-BY-4.0)
URL    : https://zenodo.org/records/5816149
File   : data_sets.tar.xz (384.6 MB archive containing dataset_14_case_1.h5
         and the other case files)

The dataset is NOT bundled with this repository.  Run:

    python scripts/download_pantagruel.py --out external/pantagruel

then convert with:

    python scripts/convert_pantagruel.py \
        --src external/pantagruel/dataset_14_case_1.h5 \
        --out-dir data/pantagruel_14_case1

For full reproduction of the paper's PanTaGruEl experiments only the
IEEE-14 case-1 file is required.
"""

import os
import sys
import tarfile
import argparse
import urllib.request

ZENODO_URL = "https://zenodo.org/records/5816149/files/data_sets.tar.xz/content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external/pantagruel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    archive = os.path.join(args.out, "data_sets.tar.xz")
    print(f"Downloading {ZENODO_URL} ...")
    urllib.request.urlretrieve(ZENODO_URL, archive)
    print(f"Extracting to {args.out} ...")
    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(args.out)
    print("Done. Files under", args.out)
    print("Required for the paper: dataset_14_case_1.h5")


if __name__ == "__main__":
    main()

# Third-Party Notices

## DSS² codebase (TU Delft AI Energy Lab)

Portions of `src/data.py`, `src/networks.py`, `src/loadsampling.py`,
`src/toy_network.py`, `src/experiment_runner.py`, and the WLS portions of
`src/modal_wls.py` derive from the DSS² (Deep Statistical Solver for
Distribution System State Estimation) codebase:

- Repository: https://github.com/TU-Delft-AI-Energy-Lab/Deep-Statistical-Solver-for-Distribution-System-State-Estimation
- Paper: B. Habib, E. Isufi, W. van Breda, A. Jongepier, J. L. Cremer, "Deep Statistical Solver for Distribution System State Estimation," IEEE Transactions on Power Systems, vol. 39, no. 2, pp. 4039–4050, 2024, DOI 10.1109/TPWRS.2023.3290358.
- Upstream license declaration (from the upstream README): "This work is licensed under a [MIT License]" (https://opensource.org/licenses/MIT).

MIT License terms of the upstream work:

Copyright (c) 2023 TU Delft AI Energy Lab (B. Habib, E. Isufi, W. van Breda, A. Jongepier, J. L. Cremer)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

In-code attribution comments from the upstream repository are retained in the
derived files. The PowerFlowNet "influence" note in `src/networks.py`
originates from the upstream DSS² repository.

## PanTaGruEl benchmark

Referenced by DOI only (not redistributed): Zenodo 10.5281/zenodo.5816149,
CC-BY-4.0. See docs/DATASETS.md.

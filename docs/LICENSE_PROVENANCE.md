# LICENSE_PROVENANCE.md

Provenance audit of third-party code in this repository (performed 2026-10-01, prior to the first public push).

| Public file/module | Origin | Original/reimplemented | Upstream repository/paper | Upstream license | MIT-compatible? | Action |
|---|---|---|---|---|---|---|
| src/data.py, src/networks.py, src/loadsampling.py, src/toy_network.py, src/experiment_runner.py (DSS2 portions), src/modal_wls.py (WLS portions) | DSS² codebase (TU Delft AI Energy Lab) | adapted | github.com/TU-Delft-AI-Energy-Lab/Deep-Statistical-Solver-for-Distribution-System-State-Estimation; Habib et al., IEEE TPWRS 2024, DOI 10.1109/TPWRS.2023.3290358 | **MIT** (declared by the upstream README: "This work is licensed under a [MIT License]" with the opensource.org badge; the repository currently lacks a LICENSE file — noted here for the record) | YES | Attribution preserved: existing in-code credits retained; see THIRD_PARTY_NOTICES.md |
| src/networks.py (PFN/SkipPFN classes) | PowerFlowNet "influence" as acknowledged by the DSS² authors themselves | not copied directly by this project (the note originates from the upstream DSS² repository; these classes are unused by the paper's experiments) | PowerFlowNet (Orfanoudakis, Vergara, Palensky, IEEE TPWRS 2024) | indirect influence only | n/a (second-order) | The DSS² authors published their repository under the MIT declaration including this note; no direct PowerFlowNet code was incorporated here |
| src/modal_losses.py, src/framework.py, src/modal_supervised_*.py, scripts/*, tests/* | this project | original | — | MIT (this repository) | YES | — |
| PanTaGruEl dataset | external data, referenced by DOI only | n/a | Zenodo 10.5281/zenodo.5816149 | CC-BY-4.0 | YES (not redistributed here) | download helper + docs/DATASETS.md |

Case classification: the direct upstream (DSS²) is Case B (adapted code with a permissively licensed source, MIT declared by the upstream authors). No Case C file remains.

LICENSE_GATE = PASS

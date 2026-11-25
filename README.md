# 27sysid-green

Code and results for the paper

> T. Nazaré, S. Martins and E. Nepomuceno, *Green-Box System Identification: Complexity-Aware Structure Selection for NARX Models*, submitted to the IFAC Journal of Systems and Control (special issue SYSID 2027).

The Green-Box method selects the regressors and the size of polynomial NARX models with a weighted criterion that combines the error reduction ratio with the arithmetic complexity of the model. This repository contains the scripts that produce the tables, figures and numbers of the paper, and the results obtained with them.

## Data (not redistributed)

The benchmark data belong to their authors. The scripts download them from the original sources.

| Benchmark | Source | How to obtain | Reference |
|---|---|---|---|
| DC motor | [sysidentpy-data](https://github.com/wilsonrljr/sysidentpy-data/tree/main/datasets/generator), files `x_cc.csv` and `y_cc.csv` | `python get_dc_data.py` | Lacerda Junior, W., Martins Almeida, V., Martins, S. (2017). Identificação de um motor/gerador CC por meio de modelos polinomiais autorregressivos e redes neurais artificiais. *XIII Simpósio Brasileiro de Automação Inteligente* |
| Coupled electric drives (CE8) | [nonlinear_benchmarks](https://github.com/MaartenSchoukens/nonlinear_benchmarks) 1.0.1 | run `CED.ipynb` | Wigren, T., Schoukens, M. (2017). Coupled electric drives data set and reference models. Technical report, Uppsala University |
| Silverbox | [nonlinear_benchmarks](https://github.com/MaartenSchoukens/nonlinear_benchmarks) 1.0.1 | run `Silverbox_arrow_validation_SysIdentPy.ipynb` | Wigren, T., Schoukens, J. (2013). Three free data sets for development and benchmarking in nonlinear system identification. *European Control Conference*, 2933–2938 |

## Third-party MATLAB functions (not redistributed)

The identification code calls the following functions of a NARMAX toolbox written by other authors (E. Mendes and L. A. Aguirre, ACSE Sheffield and UFMG). They are not included here. Place them in `code/` before running the MATLAB scripts. Please contact the authors of this repository for details.

`genterms.m`, `orthreg.m`, `get_info.m`, `build_pr.m`, `build_no.m`, `house.m`, `rowhouse.m`, `apply_qr.m`, `housels.m`, `shift_col.m`

## Requirements

- MATLAB (results obtained with R2025b).
- Python 3.14 with the packages in `requirements.txt`.
- `gcc` for the timing experiments (MinGW-w64 16.2 was used).
- Windows with the processor "Energy Meter" counters for the energy experiments (`code/e7_energy`).

All results in the paper were obtained on one core of an Intel Core i5-8500 with 32 GB of memory.

## Reproducing the paper

Run every command from `code/`. MATLAB scripts that take a benchmark read it from an environment variable, for example `setenv('FASE2_CASE','ced')`, with values `dc`, `ced` and `silverbox`.

| Step | Commands | Paper |
|---|---|---|
| Data and SysIdentPy references | `python get_dc_data.py`, then `CED.ipynb` and `Silverbox_arrow_validation_SysIdentPy.ipynb`, then `python d4_silverbox_sysidentpy_lag10.py` | Section 4 |
| Green-Box models | `green_dc_motor_sysidentpy.m`, `green_ced_exp2_sysidentpy_reference.m`, `green_silverbox_arrow_sysidentpy_reference.m`, then `fase4_figures.m` | Tables 2–5, Fig. 2 |
| Silverbox plateau | `diag_fase0_plateau.m` (`FASE0_CASE`), `diag_fase0_silverbox_plateau.m` | Section 5.1, Fig. 3(b) |
| Cost ratios, ablation and exhaustive search | `e1e2_sensitivity_ablation.m` (`FASE2_CASE`), `check_B32.m` | Section 5.3, Fig. 3(a) |
| NSGA-II | `e8_moo/export_green_sweep.m`, then `python e8_moo/e8_nsga2.py <case>` | Section 5.3 |
| Robustness and tie rule | `e5_robustness.m` (`FASE5_CASE`), `check_tie_rule_models.m`, `check_ced_endpoint.m`, then `fase4_analysis_figure.m` | Section 5.3, Fig. 3(c,d) |
| Reference time and evaluation time | `tempo_medio.m`, `e3b_e4_export.m`, then `python e3_timing/e3b_bench.py` | Section 2.1, Tables 2–4, Section 5.4 |
| Neural networks | `python e6c_nn_train.py <case> <mlp\|lstm\|subnet>`, then `python e3_timing/e6c_bench.py` and `python e6c_lifecycle.py` | Section 5.4, Fig. 4 |
| Energy | `python e7_energy/e7_eval_energy.py`, `e7_energy/e7_ident_energy.m`, `python e7_energy/e7_train_energy.py` | Section 5.4 |
| Cost figure | `fase4_cost_figure.m` | Fig. 4 |

## Results included

The results used in the paper are stored where the scripts write them, so the figure scripts run without repeating the experiments.

- `code/CED_outputs`, `code/Silverbox_arrow_validation_outputs`, `code/D4_outputs`: terms and coefficients of the SysIdentPy reference models and run manifests. The `.mat` model files embed parts of the measured data, so the notebooks regenerate them.
- `code/fase2_outputs`: cost-ratio sensitivity and ablation (`dc`, `ced`, `silverbox`), robustness (`e5`), evaluation and identification times (`e3b`, `e4`), neural networks (`e6c`) and energy (`e7`).
- `code/diag_fase0_outputs`: selection steps and final-stage criterion for each weight.
- `code/e8_moo`: weight sweeps, exhaustive DC motor front and NSGA-II runs.

Comments in the code are partly in Portuguese.

## Licence

The code and results written by the authors are released under the MIT licence (see `LICENSE`). The benchmark data and the third-party functions listed above remain under the terms of their owners.

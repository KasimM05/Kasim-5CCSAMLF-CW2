# Kasim-5CCSAMLF-CW2

Coursework split:

1. Implement the TPCRP algorithm from the coursework paper using CIFAR-10.
2. Write a two-page LaTeX report summarising the paper, results, plots, and analysis.
3. Propose, implement, and evaluate a modified TPCRP variant that improves performance.

Current status:

- Repo scaffolded for Task 1.
- CIFAR-10 training pipeline added.
- TPCRP-specific algorithm code is still blocked on the missing coursework paper / `CW` folder.

Task 1 interpretation:

- The implementation must be your own re-creation of the paper's method.
- It must run successfully without runtime errors.
- CIFAR-10 is the required dataset.
- You should keep evidence for the report and notebook printout.

Planned baseline workflow:

1. Read the TPCRP paper and extract the exact model/training procedure.
2. Implement the baseline TPCRP method in `tpcrp.py`.
3. Train/evaluate on CIFAR-10 via `train.py`.
4. Add a modified variant in `modified_tpcrp.py`.
5. Export results, plots, and tables for the report.

Files:

- `train.py`: training/evaluation entry point.
- `models.py`: baseline CNN backbone used by the current scaffold.
- `utils.py`: CIFAR-10 loading, training loop, evaluation helpers.
- `tpcrp.py`: paper-faithful TPCRP implementation placeholder.
- `modified_tpcrp.py`: modification placeholder.

Blocked item:

- Add the coursework paper or `CW` folder to this repo so the TPCRP implementation can be matched to the paper exactly.

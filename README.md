# Tabular Learning Reproduction

Small, from-scratch reproduction of two experiments from:

> L. Grinsztajn, E. Oyallon, and G. Varoquaux. **Why do tree-based models still
> outperform deep learning on tabular data?** *Advances in Neural Information
> Processing Systems (NeurIPS), Datasets and Benchmarks Track*, 2022.

Official paper repository: <https://github.com/LeoGrin/tabular-benchmark>

## Scope

This is a **partial reproduction**, not the full 45-dataset benchmark. It runs on
the OpenML `electricity` classification dataset with a 10,000-sample training
set (the paper's medium-size setting) and reproduces:

1. **Finding 2 — robustness to uninformative features**
   - removing the least important features (ranked by a Random Forest);
   - adding uncorrelated standard Gaussian noise features.
2. **Finding 3 — sensitivity to random orthogonal rotations**
   - Gaussianized features are rotated with random orthogonal matrices and the
     accuracy of an MLP is compared with the two tree models,
     **RandomForest** and **HistGradientBoosting (HistGB)**.

The neural model is a multilayer perceptron (**MLP**), and the two tree models
used throughout are **RandomForest** and **HistGradientBoosting (HistGB)**.

## Setup

Requires Python 3.10+ and internet access to download the OpenML dataset on the
first run.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python reproduce.py                 # writes CSVs to results/
python make_figures.py              # writes PNGs to figures/
```

## Figures

![Removing the least important features](figures/remove_features.png)

![Adding Gaussian noise features](figures/add_features.png)

![Random rotation of features](figures/rotations.png)

## Reproduced results

On `electricity` (10,000 training samples, fixed seed):

| Condition                 | MLP   | RandomForest | HistGB |
| ------------------------- | ----- | ------------ | ------ |
| All 8 features            | 0.815 | 0.865        | 0.882  |
| Keep 3 most important     | 0.795 | 0.846        | 0.855  |
| Add 20 noise features     | 0.763 | 0.816        | 0.851  |
| Add 50 noise features     | 0.745 | 0.792        | 0.843  |
| Original (rotation study) | 0.811 | 0.864        | 0.882  |
| Rotated (mean over seeds) | 0.814 | 0.826        | 0.818  |

These numbers are generated on a CPU with simple, fixed hyperparameters and
early stopping; the original paper uses a large random search on GPUs, so
absolute accuracy is lower, but the relative ranking and the qualitative effects
match the paper.

## Cross-dataset validation extension

The cross-dataset extension tests whether the response to irrelevant Gaussian
features is stable on two additional processed datasets from the paper's OpenML
suite: `eye_movements` (data id 44130) and `pol` (data id 44122). It uses three
split/model seeds (`42`, `123`, `2026`) and adds 0%, 25%, 50%, 75%, or 100% of
the original feature count as independent standard-Gaussian columns.

The extension preserves the original scripts and result files. Its default
experiment compares the same MLP and RandomForest configurations used above;
HistGB remains available as an optional model.

```bash
# Fast check: 1 dataset x 1 seed x 0% noise x 2 models
python cross_dataset_validation.py --datasets eye_movements --seeds 42 \
  --models MLP RandomForest --smoke

# Full experiment: 2 datasets x 3 seeds x 5 noise levels x 2 models
python cross_dataset_validation.py --datasets eye_movements pol \
  --seeds 42 123 2026 --models MLP RandomForest --resume

# Recreate figures from the saved real results without retraining
python plot_cross_dataset.py

# Tests
python -m pytest -q
```

Use `--resume` to skip completed conditions or `--overwrite` to explicitly
replace the extension CSVs. The 70%/9%/21% split is stratified. A
`QuantileTransformer(output_distribution="normal")` is fit on training rows
only. Train, validation, and test noise matrices use independent deterministic
random streams; every model/noise condition is fit from scratch.

Mean raw test accuracy over three seeds:

| Dataset | Model | 0% noise | 100% noise | Change |
| --- | --- | ---: | ---: | ---: |
| eye_movements | MLP | 0.581 | 0.554 | -0.027 |
| eye_movements | RandomForest | 0.637 | 0.603 | -0.035 |
| pol | MLP | 0.927 | 0.885 | -0.042 |
| pol | RandomForest | 0.982 | 0.968 | -0.014 |

`pol` follows the paper's aggregate observation: the MLP loses more accuracy
than RandomForest. `eye_movements` does not: RandomForest has the larger mean
drop, while the MLP change varies substantially across seeds. These mixed
results support reporting dataset dependence rather than claiming universal
robustness. Raw accuracy here must not be equated with Figure 5(b)'s normalized
cross-dataset score.

Extension artifacts:

- `results/cross_dataset_details.csv`: all 60 fitted conditions;
- `results/cross_dataset_summary.csv`: means, sample standard deviations, and
  changes from each matching 0% baseline;
- `figures/cross_dataset_eye_movements.png` and
  `figures/cross_dataset_pol.png`: raw accuracy curves;
- `figures/cross_dataset_accuracy_delta.png`: baseline-relative comparison;
- `REPRODUCTION_EXTENSION_NOTES.md`: methods, exact results, limitations, and
  report-ready observations.

## Own contributions vs. reused code

- All code in this directory was written for this reproduction. It uses
  standard libraries only: scikit-learn, NumPy, pandas, SciPy, and Matplotlib.
- No code was copied from the official `tabular-benchmark` repository. The
  experiment design and terminology follow the paper.
- The `electricity` dataset is from OpenML (data id 151), originally the UCI
  "Electricity" dataset.

## Acknowledgements

The experiments are adapted from Grinsztajn et al. (2022), NeurIPS Datasets and
Benchmarks Track. All credit for the original experimental design and findings
goes to the paper's authors.

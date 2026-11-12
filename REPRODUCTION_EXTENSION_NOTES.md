# Cross-dataset validation: reproduction extension notes

## Research question

Do the responses of an MLP and RandomForest to irrelevant Gaussian features
remain consistent across multiple real tabular datasets and random seeds?

This is a partial extension of Finding 2 in Section 5.3 of Grinsztajn et al.
(2022), not an exact reproduction of the paper's full model architectures,
hyperparameter search, 45-dataset benchmark, or normalized Figure 5(b) score.

## Division of work

The existing repository studied feature removal, Gaussian feature addition, and
random rotations on the original OpenML `electricity` dataset (data id 151).
This extension leaves that code and its CSV/PNG files unchanged and adds:

- `eye_movements`, processed OpenML data id 44130;
- `pol`, processed OpenML data id 44122;
- three split/model seeds rather than one fixed model seed;
- noise levels expressed as a fraction of the original feature dimension;
- detailed per-run records, seed-level baseline deltas, tests, and independent
  plotting entry points.

## Experimental setup

### Data

| Dataset | Samples | Original features | Target distribution | Missing values |
| --- | ---: | ---: | --- | ---: |
| eye_movements | 7,608 | 20 | 3,804 / 3,804 | 0 |
| pol | 10,082 | 26 | 5,041 / 5,041 | 0 |

OpenML returned 18 `float64` and 2 `uint8` features for `eye_movements`, and 26
`uint8` features for `pol`. Targets were stably encoded with `LabelEncoder`.

For every dataset and seed, indices were stratified into approximately 70%
train, 9% validation, and 21% test:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| eye_movements | 5,325 | 684 | 1,599 |
| pol | 7,057 | 907 | 2,118 |

All models and noise levels for a dataset/seed use exactly the same row
indices. The external validation split is reported but not used to select the
fixed hyperparameters. Scikit-learn MLP early stopping uses its own 10% subset
of the training rows.

### Preprocessing and irrelevant features

Original numerical features are transformed with
`QuantileTransformer(n_quantiles=1000, output_distribution="normal")`. The
transformer is fit only on training rows and then applied to validation and
test rows.

Noise ratios are `[0, .25, .50, .75, 1.0]`. With original dimension `d`, the
experiment appends `round(d * ratio)` independent standard-normal columns. For
each dataset/seed, train, validation, and test receive independent master noise
matrices. Increasing ratios use prefixes of those matrices, and both models use
the same matrices for a condition. Every noise level constructs and fits a
fresh model.

Split/model seeds are `42`, `123`, and `2026`. The recorded noise seed is
`dataset_id * 1000 + split_seed`, giving a separate deterministic stream for
each dataset/split pair.

### Fixed models

The configurations follow the existing teammate experiment:

- MLP: hidden layers `(256, 256)`, ReLU, Adam, batch size 512, learning rate
  0.001, L2 alpha 0.0001, maximum 250 iterations, early stopping, patience 15,
  and model `random_state=split_seed`.
- RandomForest: 300 estimators, default `max_features="sqrt"`, no depth limit,
  `n_jobs=-1`, and model `random_state=split_seed`.

No hyperparameters were selected using test accuracy. HistGB is implemented as
an optional model but was not included in the required 60-run experiment.

## Actual execution

The full command was:

```bash
python cross_dataset_validation.py --datasets eye_movements pol \
  --seeds 42 123 2026 --models MLP RandomForest --resume
python plot_cross_dataset.py
python -m pytest -q
```

All 60 expected combinations completed successfully. There were no failed rows
and no captured model warnings. Recorded model fit time summed to 248.17
seconds on the local CPU; download, preprocessing, prediction, CSV, and plotting
time are not included in that sum.

## Results

Values below are mean raw test accuracy with sample standard deviation across
three seeds.

### eye_movements

| Added noise | MLP | RandomForest |
| ---: | ---: | ---: |
| 0% | 0.5808 ± 0.0228 | 0.6375 ± 0.0094 |
| 25% | 0.5718 ± 0.0162 | 0.6239 ± 0.0167 |
| 50% | 0.5681 ± 0.0048 | 0.6070 ± 0.0104 |
| 75% | 0.5620 ± 0.0076 | 0.6100 ± 0.0160 |
| 100% | 0.5541 ± 0.0047 | 0.6029 ± 0.0109 |

At 100% noise, the seed-matched mean change is -0.0267 for MLP and -0.0346
for RandomForest. Thus this dataset does not reproduce the expected ordering:
RandomForest has the larger mean loss. The MLP delta also has high cross-seed
variation (SD 0.0259), driven partly by different 0% baselines.

### pol

| Added noise | MLP | RandomForest |
| ---: | ---: | ---: |
| 0% | 0.9268 ± 0.0037 | 0.9824 ± 0.0011 |
| 25% | 0.9160 ± 0.0086 | 0.9775 ± 0.0010 |
| 50% | 0.9021 ± 0.0035 | 0.9740 ± 0.0017 |
| 75% | 0.8950 ± 0.0031 | 0.9698 ± 0.0025 |
| 100% | 0.8850 ± 0.0063 | 0.9681 ± 0.0022 |

At 100% noise, the seed-matched mean change is -0.0419 for MLP and -0.0143
for RandomForest. This dataset agrees with the paper's qualitative aggregate
observation: the MLP is substantially more sensitive to irrelevant features.

Equally averaging the two dataset-level 100% changes gives -0.0343 for MLP and
-0.0245 for RandomForest. This two-dataset descriptive average is not a
benchmark score and is too small a sample for a general claim.

## Comparison with the paper and teammate experiment

The paper's Figure 5(b) aggregates normalized scores across datasets and model
families after extensive hyperparameter searches. Our figures show raw test
accuracy for two datasets and two fixed scikit-learn models. Numerical values
must therefore not be compared as if they were the same metric. Appendix Figure
22 is the more relevant paper reference because it exposes dataset-specific raw
accuracy behavior, but the architectures and tuning budgets still differ.

The teammate `electricity` experiment uses OpenML data id 151, a fixed 10,000
sample training set, one fixed model seed, and absolute noise widths
`[0, 5, 10, 20, 50]` for only eight original features. This extension uses the
paper's processed data ids, 70%/9%/21% splits, three seeds, and ratios no larger
than the original dimension. Absolute scores and x-axis values should not be
treated as directly comparable. The useful comparison is that sensitivity to
irrelevant features is dataset- and model-dependent.

## Discrepancies, challenges, and limitations

- `eye_movements` contradicts the expected MLP-vs-tree ordering, whereas `pol`
  supports it. This is evidence against presenting the phenomenon as universal.
- Only two added datasets and three seeds are used; uncertainty remains large,
  especially for the MLP on `eye_movements`.
- Scikit-learn MLP is a substitute for the paper's PyTorch MLP, ResNet, and
  FT-Transformer models, and no ~400-trial hyperparameter search was performed.
- Fixed hyperparameters may favor one model differently on each dataset.
- RandomForest reaches 1.0 training accuracy in every recorded condition, so
  its test changes reflect generalization rather than inability to fit training
  labels.
- OpenML downloads required network permission and returned uncompressed ARFF
  content. Dataset shapes, types, balance, and missingness were checked before
  training.
- GitHub CLI authentication on this computer was invalid during implementation;
  this affects pushing/PR creation, not the recorded experiment.

## Reproducibility and artifacts

- `results/cross_dataset_details.csv`: 60 per-condition records, model
  parameters, seeds, split sizes, timing, status, and warnings.
- `results/cross_dataset_summary.csv`: mean/SD accuracy and seed-matched
  baseline deltas.
- `figures/cross_dataset_eye_movements.png`
- `figures/cross_dataset_pol.png`
- `figures/cross_dataset_accuracy_delta.png`
- `tests/test_cross_dataset.py`: split integrity, leakage prevention, noise
  determinism/independence/sharing, seed changes, schema, and summary logic.

## Report-ready conclusion

> We extended the team's reproduction to two processed OpenML datasets and
> repeated each condition over three split and model seeds. On `pol`, adding as
> many independent Gaussian features as original features reduced MLP accuracy
> by 4.19 percentage points, compared with 1.43 points for RandomForest, in line
> with the paper's aggregate finding. On `eye_movements`, however, the mean loss
> was 2.67 points for MLP and 3.46 points for RandomForest. The mixed result shows
> that the observed robustness gap is not stable across every tested dataset and
> should be interpreted within the limited model, tuning, and dataset scope.

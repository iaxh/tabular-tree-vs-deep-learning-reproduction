"""Cross-dataset validation of robustness to irrelevant Gaussian features.

This extension preserves the original ``reproduce.py`` experiment and adds
two processed OpenML datasets from the paper's numerical-classification suite.
Every noise level trains a fresh model. Run ``python cross_dataset_validation.py
--help`` for the available controls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


DATASETS = {"eye_movements": 44130, "pol": 44122}
EXPECTED_SHAPES = {"eye_movements": (7608, 20), "pol": (10082, 26)}
NOISE_RATIOS = (0.00, 0.25, 0.50, 0.75, 1.00)
DEFAULT_SEEDS = (42, 123, 2026)
MODEL_NAMES = ("MLP", "RandomForest", "HistGB")
DETAIL_COLUMNS = (
    "dataset",
    "dataset_id",
    "model",
    "split_seed",
    "noise_seed",
    "noise_ratio",
    "n_original_features",
    "n_noise_features",
    "n_train",
    "n_val",
    "n_test",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "fit_seconds",
    "model_parameters",
    "warnings",
    "status",
)
RUN_KEY_COLUMNS = (
    "dataset",
    "model",
    "split_seed",
    "noise_seed",
    "noise_ratio",
)


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    dataset_id: int
    X: np.ndarray
    y: np.ndarray
    feature_dtypes: tuple[str, ...]
    target_name: str
    target_dtype: str
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class FeatureSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_dataset(name: str, cache_dir: str | Path = ".cache/openml") -> DatasetBundle:
    """Download one configured OpenML dataset and validate its schema."""

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")

    resolved_cache = Path(cache_dir).resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPENML_CACHE_DIR"] = str(resolved_cache)
    os.environ.setdefault("OPENML_SKIP_PARQUET", "true")
    import openml

    openml.config.set_root_cache_directory(resolved_cache)
    dataset_id = DATASETS[name]
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    target_name = dataset.default_target_attribute
    if not target_name:
        raise ValueError(f"OpenML dataset {dataset_id} has no default target")

    X_frame, y_series, _categorical, _feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=target_name,
    )
    if y_series is None or X_frame.empty:
        raise ValueError(f"OpenML dataset {dataset_id} returned empty data")

    expected_shape = EXPECTED_SHAPES[name]
    if X_frame.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {name}: got {X_frame.shape}, expected {expected_shape}"
        )
    non_numeric = [
        column
        for column in X_frame.columns
        if not pd.api.types.is_numeric_dtype(X_frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    feature_missing = X_frame.isna().sum()
    feature_missing = feature_missing[feature_missing > 0]
    target_missing = int(pd.isna(y_series).sum())
    if not feature_missing.empty or target_missing:
        raise ValueError(
            "Missing values are not silently imputed: "
            f"features={feature_missing.to_dict()}, target={target_missing}"
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(np.asarray(y_series))
    if len(encoder.classes_) != 2:
        raise ValueError(
            f"Expected binary labels, found {len(encoder.classes_)}: "
            f"{encoder.classes_.tolist()}"
        )
    X = X_frame.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(X).all():
        raise ValueError("Features contain non-finite values")

    return DatasetBundle(
        name=name,
        dataset_id=dataset_id,
        X=X,
        y=y.astype(np.int64, copy=False),
        feature_dtypes=tuple(str(dtype) for dtype in X_frame.dtypes),
        target_name=str(target_name),
        target_dtype=str(y_series.dtype),
        class_names=tuple(str(label) for label in encoder.classes_),
    )


def stratified_split_indices(y: np.ndarray, seed: int) -> SplitIndices:
    """Create a deterministic, stratified 70%/9%/21% split."""

    indices = np.arange(len(y))
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=0.70,
        random_state=seed,
        stratify=y,
    )
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=0.30,
        random_state=seed,
        stratify=y[remainder_idx],
    )
    return SplitIndices(
        train=np.sort(train_idx),
        validation=np.sort(validation_idx),
        test=np.sort(test_idx),
    )


def gaussianize_original_features(
    raw: FeatureSplits,
    *,
    seed: int,
) -> tuple[FeatureSplits, QuantileTransformer]:
    """Fit QuantileTransformer on training rows only and transform each split."""

    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(raw.train)),
        output_distribution="normal",
        random_state=seed,
    )
    transformed = FeatureSplits(
        train=transformer.fit_transform(raw.train),
        validation=transformer.transform(raw.validation),
        test=transformer.transform(raw.test),
    )
    return transformed, transformer


def derive_noise_seed(dataset_id: int, split_seed: int) -> int:
    """Give every dataset/split pair a recorded deterministic noise seed."""

    return dataset_id * 1000 + split_seed


def generate_master_noise(
    *,
    split_sizes: tuple[int, int, int],
    max_noise_features: int,
    noise_seed: int,
) -> FeatureSplits:
    """Generate independent train/validation/test master noise matrices."""

    child_seeds = np.random.SeedSequence(noise_seed).spawn(3)
    matrices = [
        np.random.default_rng(child_seed).normal(size=(n_rows, max_noise_features))
        for child_seed, n_rows in zip(child_seeds, split_sizes, strict=True)
    ]
    return FeatureSplits(*matrices)


def append_noise(
    features: FeatureSplits,
    master_noise: FeatureSplits,
    n_noise_features: int,
) -> FeatureSplits:
    """Append the same master-noise prefix for every model in a condition."""

    augmented = []
    for X, noise in zip(
        (features.train, features.validation, features.test),
        (master_noise.train, master_noise.validation, master_noise.test),
        strict=True,
    ):
        if len(X) != len(noise):
            raise ValueError("Feature and noise row counts differ")
        if not 0 <= n_noise_features <= noise.shape[1]:
            raise ValueError("Requested noise width is outside the master matrix")
        if n_noise_features:
            augmented.append(np.hstack((X, noise[:, :n_noise_features])))
        else:
            augmented.append(X.copy())
    return FeatureSplits(*augmented)


def make_models(names: list[str] | tuple[str, ...], seed: int) -> dict[str, object]:
    """Construct fresh models using the teammate experiment's parameters."""

    available = {
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=seed,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGB": lambda: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            random_state=seed,
        ),
    }
    unknown = set(names).difference(available)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}")
    return {name: available[name]() for name in names}


def summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy and per-seed changes from successful real runs."""

    successful = details.loc[details["status"] == "ok"].copy()
    baseline_keys = ["dataset", "model", "split_seed", "noise_seed"]
    baselines = successful.loc[
        np.isclose(successful["noise_ratio"], 0.0),
        baseline_keys + ["test_accuracy"],
    ].rename(columns={"test_accuracy": "baseline_accuracy"})
    if baselines.duplicated(baseline_keys).any():
        raise ValueError("Duplicate 0% baseline rows found")
    successful = successful.merge(
        baselines,
        how="left",
        on=baseline_keys,
        validate="many_to_one",
    )
    if successful["baseline_accuracy"].isna().any():
        raise ValueError("Every successful run group needs a 0% baseline")
    successful["delta_accuracy"] = (
        successful["test_accuracy"] - successful["baseline_accuracy"]
    )
    summary = (
        successful.groupby(["dataset", "model", "noise_ratio"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            std_delta_accuracy=("delta_accuracy", "std"),
            n_runs=("test_accuracy", "size"),
        )
        .sort_values(["dataset", "model", "noise_ratio"])
        .reset_index(drop=True)
    )
    summary[["std_accuracy", "std_delta_accuracy"]] = summary[
        ["std_accuracy", "std_delta_accuracy"]
    ].fillna(0.0)
    return summary


def _print_dataset_diagnostics(bundle: DatasetBundle) -> None:
    labels, counts = np.unique(bundle.y, return_counts=True)
    distribution = dict(zip(labels.tolist(), counts.tolist(), strict=True))
    print(
        f"Loaded {bundle.name} (OpenML {bundle.dataset_id}): X={bundle.X.shape}, "
        f"target={bundle.target_name!r}, target_dtype={bundle.target_dtype}"
    )
    print(
        f"  feature_dtypes={dict(Counter(bundle.feature_dtypes))}, missing=0, "
        f"classes={bundle.class_names}, encoded_distribution={distribution}"
    )


def _read_existing(path: Path, *, resume: bool, overwrite: bool) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    if overwrite:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    if not resume:
        raise FileExistsError(
            f"Refusing to overwrite {path}; pass --resume or --overwrite explicitly"
        )
    existing = pd.read_csv(path)
    missing = set(DETAIL_COLUMNS).difference(existing.columns)
    if missing:
        raise ValueError(f"Existing details CSV is missing columns: {sorted(missing)}")
    return existing.loc[:, DETAIL_COLUMNS]


def _write_outputs(details: pd.DataFrame, details_path: Path, summary_path: Path) -> None:
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details.loc[:, DETAIL_COLUMNS].to_csv(details_path, index=False)
    summary = summarize_details(details)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    details = _read_existing(
        args.output,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    completed = {
        tuple(row[column] for column in RUN_KEY_COLUMNS)
        for row in details.to_dict("records")
    }
    ratios = (0.0,) if args.smoke else NOISE_RATIOS

    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name, cache_dir=args.cache_dir)
        _print_dataset_diagnostics(bundle)
        for split_seed in args.seeds:
            indices = stratified_split_indices(bundle.y, split_seed)
            raw = FeatureSplits(
                train=bundle.X[indices.train],
                validation=bundle.X[indices.validation],
                test=bundle.X[indices.test],
            )
            y_train = bundle.y[indices.train]
            y_validation = bundle.y[indices.validation]
            y_test = bundle.y[indices.test]
            processed, _transformer = gaussianize_original_features(raw, seed=split_seed)
            noise_seed = derive_noise_seed(bundle.dataset_id, split_seed)
            master_noise = generate_master_noise(
                split_sizes=(len(indices.train), len(indices.validation), len(indices.test)),
                max_noise_features=bundle.X.shape[1],
                noise_seed=noise_seed,
            )

            for ratio in ratios:
                n_noise = round(bundle.X.shape[1] * ratio)
                augmented = append_noise(processed, master_noise, n_noise)
                print(
                    f"[{bundle.name}] split_seed={split_seed} noise_seed={noise_seed} "
                    f"ratio={ratio:.2f} ({n_noise} columns)"
                )
                for model_name, model in make_models(args.models, split_seed).items():
                    key = (bundle.name, model_name, split_seed, noise_seed, ratio)
                    if key in completed:
                        print(f"  {model_name}: already present, skipping (--resume)")
                        continue

                    caught_messages: list[str] = []
                    started = time.perf_counter()
                    status = "ok"
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(augmented.train, y_train)
                        caught_messages = [str(item.message) for item in caught]
                        fit_seconds = time.perf_counter() - started
                        train_accuracy = accuracy_score(
                            y_train, model.predict(augmented.train)
                        )
                        validation_accuracy = accuracy_score(
                            y_validation, model.predict(augmented.validation)
                        )
                        test_accuracy = accuracy_score(y_test, model.predict(augmented.test))
                    except Exception as error:
                        fit_seconds = time.perf_counter() - started
                        train_accuracy = np.nan
                        validation_accuracy = np.nan
                        test_accuracy = np.nan
                        status = "error"
                        caught_messages.append(f"{type(error).__name__}: {error}")

                    row = {
                        "dataset": bundle.name,
                        "dataset_id": bundle.dataset_id,
                        "model": model_name,
                        "split_seed": split_seed,
                        "noise_seed": noise_seed,
                        "noise_ratio": ratio,
                        "n_original_features": bundle.X.shape[1],
                        "n_noise_features": n_noise,
                        "n_train": len(indices.train),
                        "n_val": len(indices.validation),
                        "n_test": len(indices.test),
                        "train_accuracy": train_accuracy,
                        "validation_accuracy": validation_accuracy,
                        "test_accuracy": test_accuracy,
                        "fit_seconds": fit_seconds,
                        "model_parameters": json.dumps(
                            model.get_params(deep=False), sort_keys=True, default=str
                        ),
                        "warnings": " | ".join(caught_messages),
                        "status": status,
                    }
                    details = pd.concat(
                        [details, pd.DataFrame([row], columns=DETAIL_COLUMNS)],
                        ignore_index=True,
                    )
                    completed.add(key)
                    _write_outputs(details, args.output, args.summary)
                    print(
                        f"  {model_name}: train={train_accuracy:.4f}, "
                        f"validation={validation_accuracy:.4f}, "
                        f"test={test_accuracy:.4f}, fit={fit_seconds:.2f}s, "
                        f"status={status}"
                    )
                    if status == "error":
                        raise RuntimeError(caught_messages[-1])
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), required=True
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_NAMES),
        default=["MLP", "RandomForest"],
    )
    parser.add_argument("--smoke", action="store_true", help="Run only 0%% noise")
    parser.add_argument(
        "--output", type=Path, default=Path("results/cross_dataset_details.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/cross_dataset_summary.csv")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/openml"))
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--resume", action="store_true")
    control.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = run_experiment(args)
    print(f"Saved {len(details)} detailed rows to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()

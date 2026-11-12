import numpy as np
import pandas as pd

from cross_dataset_validation import (
    DETAIL_COLUMNS,
    FeatureSplits,
    append_noise,
    derive_noise_seed,
    gaussianize_original_features,
    generate_master_noise,
    make_models,
    stratified_split_indices,
    summarize_details,
)


def test_split_is_deterministic_stratified_and_disjoint() -> None:
    y = np.tile([0, 1], 500)
    first = stratified_split_indices(y, seed=42)
    second = stratified_split_indices(y, seed=42)

    assert (len(first.train), len(first.validation), len(first.test)) == (700, 90, 210)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)
    assert not set(first.validation) & set(first.test)
    assert set(np.concatenate((first.train, first.validation, first.test))) == set(
        range(1000)
    )
    assert y[first.train].sum() == 350
    assert y[first.validation].sum() == 45
    assert y[first.test].sum() == 105


def test_quantile_transformer_fits_training_rows_only() -> None:
    raw = FeatureSplits(
        train=np.arange(40, dtype=float).reshape(20, 2),
        validation=np.full((4, 2), 1_000_000.0),
        test=np.full((6, 2), -1_000_000.0),
    )

    transformed, transformer = gaussianize_original_features(raw, seed=42)

    assert transformed.train.shape == raw.train.shape
    assert transformed.validation.shape == raw.validation.shape
    assert transformed.test.shape == raw.test.shape
    np.testing.assert_allclose(transformer.quantiles_[0], raw.train.min(axis=0))
    np.testing.assert_allclose(transformer.quantiles_[-1], raw.train.max(axis=0))


def test_master_noise_is_deterministic_and_split_independent() -> None:
    kwargs = {
        "split_sizes": (12, 12, 12),
        "max_noise_features": 5,
        "noise_seed": 44_130_042,
    }
    first = generate_master_noise(**kwargs)
    second = generate_master_noise(**kwargs)

    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assert not np.array_equal(first.train, first.validation)
    assert not np.array_equal(first.train, first.test)
    assert not np.array_equal(first.validation, first.test)


def test_models_share_noise_prefix_and_original_features_are_preserved() -> None:
    features = FeatureSplits(
        train=np.arange(12, dtype=float).reshape(4, 3),
        validation=np.arange(6, dtype=float).reshape(2, 3),
        test=np.arange(9, dtype=float).reshape(3, 3),
    )
    master = generate_master_noise(
        split_sizes=(4, 2, 3), max_noise_features=3, noise_seed=123
    )

    for_model_a = append_noise(features, master, n_noise_features=2)
    for_model_b = append_noise(features, master, n_noise_features=2)

    np.testing.assert_array_equal(for_model_a.train, for_model_b.train)
    np.testing.assert_array_equal(for_model_a.validation, for_model_b.validation)
    np.testing.assert_array_equal(for_model_a.test, for_model_b.test)
    np.testing.assert_array_equal(for_model_a.train[:, :3], features.train)
    np.testing.assert_array_equal(for_model_a.train[:, 3:], master.train[:, :2])


def test_model_and_noise_seeds_really_change() -> None:
    first = make_models(["MLP", "RandomForest", "HistGB"], seed=42)
    second = make_models(["MLP", "RandomForest", "HistGB"], seed=123)

    assert {model.get_params()["random_state"] for model in first.values()} == {42}
    assert {model.get_params()["random_state"] for model in second.values()} == {123}
    assert derive_noise_seed(44130, 42) != derive_noise_seed(44130, 123)
    assert derive_noise_seed(44130, 42) != derive_noise_seed(44122, 42)


def test_detail_schema_and_summary_use_matching_seed_baselines() -> None:
    required = {
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
        "test_accuracy",
        "fit_seconds",
        "model_parameters",
        "warnings",
        "status",
    }
    assert required.issubset(DETAIL_COLUMNS)

    details = pd.DataFrame(
        [
            ["demo", "MLP", 1, 101, 0.0, 0.80, "ok"],
            ["demo", "MLP", 1, 101, 1.0, 0.70, "ok"],
            ["demo", "MLP", 2, 102, 0.0, 0.60, "ok"],
            ["demo", "MLP", 2, 102, 1.0, 0.55, "ok"],
        ],
        columns=[
            "dataset",
            "model",
            "split_seed",
            "noise_seed",
            "noise_ratio",
            "test_accuracy",
            "status",
        ],
    )

    summary = summarize_details(details)
    full_noise = summary.loc[summary["noise_ratio"] == 1.0].iloc[0]
    assert full_noise["mean_accuracy"] == 0.625
    np.testing.assert_allclose(full_noise["mean_delta_accuracy"], -0.075)
    assert full_noise["n_runs"] == 2

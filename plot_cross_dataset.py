"""Plot cross-dataset validation results without retraining models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


_cache = Path(".cache/matplotlib").resolve()
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"MLP": "#d62728", "RandomForest": "#1f77b4", "HistGB": "#2ca02c"}
MARKERS = {"MLP": "o", "RandomForest": "s", "HistGB": "^"}
MODEL_ORDER = ("MLP", "RandomForest", "HistGB")
REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "noise_ratio",
    "mean_accuracy",
    "std_accuracy",
    "mean_delta_accuracy",
    "n_runs",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("Summary CSV is empty")
    return summary


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_dataset(summary: pd.DataFrame, dataset: str, destination: Path) -> None:
    subset = summary.loc[summary["dataset"] == dataset]
    if subset.empty:
        raise ValueError(f"No rows found for dataset {dataset!r}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model in MODEL_ORDER:
        rows = subset.loc[subset["model"] == model].sort_values("noise_ratio")
        if rows.empty:
            continue
        ax.errorbar(
            rows["noise_ratio"] * 100,
            rows["mean_accuracy"],
            yerr=rows["std_accuracy"],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2,
            capsize=3,
            label=model,
        )
    style(
        ax,
        "Added Gaussian features (% of original dimension)",
        "Raw test accuracy",
        f"Cross-dataset validation: {dataset}",
    )
    ax.set_xticks(sorted(subset["noise_ratio"].unique() * 100))
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.01,
        "Team extension; not the original paper results. Error bars: sample SD across seeds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta(summary: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    datasets = sorted(summary["dataset"].unique())
    linestyles = ["-", "--", ":", "-."]
    for dataset_index, dataset in enumerate(datasets):
        for model in MODEL_ORDER:
            rows = summary.loc[
                (summary["dataset"] == dataset) & (summary["model"] == model)
            ].sort_values("noise_ratio")
            if rows.empty:
                continue
            ax.plot(
                rows["noise_ratio"] * 100,
                rows["mean_delta_accuracy"],
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=linestyles[dataset_index % len(linestyles)],
                linewidth=2,
                label=f"{dataset} — {model}",
            )
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    style(
        ax,
        "Added Gaussian features (% of original dimension)",
        "Test accuracy change from matching 0% baseline",
        "Cross-dataset change under irrelevant features",
    )
    ax.set_xticks(sorted(summary["noise_ratio"].unique() * 100))
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Team extension; raw accuracy deltas, not Figure 5(b)'s normalized aggregate score.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("results/cross_dataset_summary.csv")
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    for dataset in sorted(summary["dataset"].unique()):
        destination = args.outdir / f"cross_dataset_{dataset}.png"
        plot_dataset(summary, dataset, destination)
        print(f"Saved {destination}")
    delta_path = args.outdir / "cross_dataset_accuracy_delta.png"
    plot_delta(summary, delta_path)
    print(f"Saved {delta_path}")


if __name__ == "__main__":
    main()

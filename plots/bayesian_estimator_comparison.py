"""Графики сравнения байесовских оценок параметра с MLE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.geometric_bayesian_estimation import (
    ESTIMATOR_LABELS,
    BayesianEstimationExperiment,
)


def _prepare_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bayesian_comparison_plots(
    result: BayesianEstimationExperiment,
    output_dir: str | Path = "images/geometric_bayesian_estimation",
) -> list[Path]:
    """Построить сравнение оценок по трём функциям потерь."""

    output = _prepare_directory(output_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    saved: list[Path] = []
    metrics = (
        ("mse", "Среднеквадратический риск"),
        ("mae", "Средний абсолютный риск"),
        (
            "epsilon_zero_one_risk",
            rf"$P(|\widehat{{\theta}}-\theta|>{result.epsilon:g})$",
        ),
    )
    thetas = sorted(result.summary["theta"].unique())

    for metric, ylabel in metrics:
        fig, axes = plt.subplots(
            1,
            len(thetas),
            figsize=(5.2 * len(thetas), 4.3),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        for axis, theta in zip(axes, thetas):
            subset = result.summary[result.summary["theta"] == theta]
            for estimator, label in ESTIMATOR_LABELS.items():
                line = subset[subset["estimator"] == estimator]
                axis.plot(line["n"], line[metric], "o-", label=label)
            axis.set_xscale("log")
            axis.set_xlabel("n")
            axis.set_title(rf"$\theta={theta:g}$")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"Сравнение оценок: {ylabel}")
        fig.tight_layout()
        path = output / f"{metric}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    middle_theta = thetas[len(thetas) // 2]
    available_sizes = sorted(
        n for theta, n in result.estimates if theta == middle_theta
    )
    selected_sizes = sorted({available_sizes[0], available_sizes[-1]})
    fig, axes = plt.subplots(
        1,
        len(selected_sizes),
        figsize=(7 * len(selected_sizes), 4.5),
    )
    axes = np.atleast_1d(axes)
    for axis, n in zip(axes, selected_sizes):
        values = result.estimates[(middle_theta, n)]
        data = [values[key] for key in ESTIMATOR_LABELS]
        axis.boxplot(
            data,
            tick_labels=list(ESTIMATOR_LABELS.values()),
            showfliers=False,
        )
        axis.axhline(middle_theta, color="black", linestyle="--")
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(rf"$\theta={middle_theta:g},\ n={n}$")
    fig.suptitle("Распределения оценок")
    fig.tight_layout()
    path = output / "estimate_boxplots.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


"""Графики результатов эксперимента второго задания."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.bayes_style import (
    ESTIMATOR_COLORS,
    ESTIMATOR_LABELS,
    prepare_directory,
)


def save_metric_plot(
    experiment: dict,
    bayes_name: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str | Path,
) -> Path:
    """Сравнить риск байесовской оценки и MLE."""

    output = prepare_directory(output_dir)
    theta_values = experiment["theta_values"]
    sample_sizes = experiment["sample_sizes"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1,
        len(theta_values),
        figsize=(5.2 * len(theta_values), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, theta in zip(axes, theta_values):
        mle_values = []
        bayes_values = []

        for n in sample_sizes:
            current = experiment["results"][float(theta)][int(n)]
            mle_values.append(current["metrics"]["mle"][metric])
            bayes_values.append(current["metrics"][bayes_name][metric])

        axis.plot(
            sample_sizes,
            mle_values,
            "o-",
            color=ESTIMATOR_COLORS["mle"],
            label="MLE",
        )
        axis.plot(
            sample_sizes,
            bayes_values,
            "s--",
            color=ESTIMATOR_COLORS[bayes_name],
            label=ESTIMATOR_LABELS[bayes_name],
        )
        axis.set_xscale("log")
        axis.set_xlabel("объём выборки n")
        axis.set_title(rf"$\theta={theta:g}$")

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()

    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_bias_plot(
    experiment: dict,
    bayes_name: str,
    title: str,
    filename: str,
    output_dir: str | Path,
) -> Path:
    """Сравнить смещения байесовской оценки и MLE."""

    output = prepare_directory(output_dir)
    theta_values = experiment["theta_values"]
    sample_sizes = experiment["sample_sizes"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1,
        len(theta_values),
        figsize=(5.2 * len(theta_values), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, theta in zip(axes, theta_values):
        mle_bias = []
        bayes_bias = []

        for n in sample_sizes:
            current = experiment["results"][float(theta)][int(n)]
            mle_bias.append(current["metrics"]["mle"]["bias"])
            bayes_bias.append(current["metrics"][bayes_name]["bias"])

        axis.plot(
            sample_sizes,
            mle_bias,
            "o-",
            color=ESTIMATOR_COLORS["mle"],
            label="MLE",
        )
        axis.plot(
            sample_sizes,
            bayes_bias,
            "s--",
            color=ESTIMATOR_COLORS[bayes_name],
            label=ESTIMATOR_LABELS[bayes_name],
        )
        axis.axhline(0, color="gray", linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel("объём выборки n")
        axis.set_title(rf"$\theta={theta:g}$")

    axes[0].set_ylabel("смещение")
    axes[-1].legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()

    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_estimate_cloud_plot(
    experiment: dict,
    theta: float,
    sample_sizes: np.ndarray,
    bayes_name: str,
    title: str,
    filename: str,
    output_dir: str | Path,
) -> Path:
    """Показать отдельные значения двух оценок на общих выборках."""

    output = prepare_directory(output_dir)
    point_count = 250
    positions = np.arange(len(sample_sizes))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 5.5))

    for index, n in enumerate(sample_sizes):
        current = experiment["results"][float(theta)][int(n)]
        mle_values = current["estimates"]["mle"][:point_count]
        bayes_values = current["estimates"][bayes_name][:point_count]

        mle_x = np.full(mle_values.shape, positions[index] - 0.10)
        bayes_x = np.full(bayes_values.shape, positions[index] + 0.10)

        mle_label = None
        bayes_label = None
        if index == 0:
            mle_label = "отдельные оценки MLE"
            bayes_label = "отдельные байесовские оценки"

        axis.scatter(
            mle_x,
            mle_values,
            color=ESTIMATOR_COLORS["mle"],
            alpha=0.18,
            s=18,
            label=mle_label,
        )
        axis.scatter(
            bayes_x,
            bayes_values,
            color=ESTIMATOR_COLORS[bayes_name],
            alpha=0.24,
            s=18,
            label=bayes_label,
        )

        axis.scatter(
            positions[index] - 0.10,
            mle_values.mean(),
            color=ESTIMATOR_COLORS["mle"],
            marker="D",
            s=60,
            zorder=3,
        )
        axis.scatter(
            positions[index] + 0.10,
            bayes_values.mean(),
            color=ESTIMATOR_COLORS[bayes_name],
            marker="D",
            s=60,
            zorder=3,
        )

    axis.axhline(
        theta,
        color="tab:green",
        linestyle=":",
        linewidth=2,
        label=rf"истинное $\theta={theta:g}$",
    )
    axis.set_xticks(positions, [str(int(n)) for n in sample_sizes])
    axis.set_xlabel("объём выборки n")
    axis.set_ylabel(r"оценка $\widehat\theta$")
    axis.set_title(title)
    axis.legend()

    fig.tight_layout()
    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_risk_difference_plot(
    experiment: dict,
    bayes_name: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str | Path,
) -> Path:
    """Показать разность между байесовским риском и риском MLE."""

    output = prepare_directory(output_dir)
    theta_values = experiment["theta_values"]
    sample_sizes = experiment["sample_sizes"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.2))

    for theta in theta_values:
        differences = []
        for n in sample_sizes:
            current = experiment["results"][float(theta)][int(n)]
            bayes_risk = current["metrics"][bayes_name][metric]
            mle_risk = current["metrics"]["mle"][metric]
            differences.append(bayes_risk - mle_risk)

        axis.plot(
            sample_sizes,
            differences,
            "o-",
            label=rf"$\theta={theta:g}$",
        )

    axis.axhline(
        0,
        color="black",
        linestyle=":",
        label="риски равны",
    )
    axis.set_xscale("log")
    axis.set_xlabel("объём выборки n")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()

    fig.tight_layout()
    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_threshold_counts_plot(
    experiment: dict,
    output_dir: str | Path,
) -> Path:
    """Число оценок, вышедших за заданный epsilon-порог."""

    output = prepare_directory(output_dir)
    theta_values = experiment["theta_values"]
    sample_sizes = experiment["sample_sizes"]
    epsilon = experiment["epsilon"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1,
        len(theta_values),
        figsize=(5.2 * len(theta_values), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, theta in zip(axes, theta_values):
        for name in ESTIMATOR_LABELS:
            counts = []
            for n in sample_sizes:
                current = experiment["results"][float(theta)][int(n)]
                counts.append(
                    current["metrics"][name]["exceedances"]
                )

            linestyle = "-"
            if name == "zero_one":
                linestyle = "--"
            axis.plot(
                sample_sizes,
                counts,
                marker="o",
                linestyle=linestyle,
                color=ESTIMATOR_COLORS[name],
                label=ESTIMATOR_LABELS[name],
            )

        axis.set_xscale("log")
        axis.set_xlabel("объём выборки n")
        axis.set_title(rf"$\theta={theta:g}$")

    axes[0].set_ylabel("число превышений")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        rf"Число случаев $|\widehat\theta-\theta|>{epsilon:g}$"
    )
    fig.tight_layout()

    path = output / "threshold_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_threshold_cloud_plot(
    experiment: dict,
    theta: float,
    sample_sizes: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Отдельные MAP-оценки и допустимая epsilon-полоса."""

    output = prepare_directory(output_dir)
    epsilon = experiment["epsilon"]
    positions = np.arange(len(sample_sizes))
    point_count = 300

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 5.5))

    for index, n in enumerate(sample_sizes):
        current = experiment["results"][float(theta)][int(n)]
        estimates = current["estimates"]["zero_one"][:point_count]
        count = current["metrics"]["zero_one"]["exceedances"]
        x = np.full(estimates.shape, positions[index])

        label = None
        if index == 0:
            label = "отдельные MAP-оценки"

        axis.scatter(
            x,
            estimates,
            color=ESTIMATOR_COLORS["zero_one"],
            alpha=0.24,
            s=18,
            label=label,
        )
        axis.text(
            positions[index],
            theta + epsilon + 0.025,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.axhline(
        theta,
        color="black",
        linestyle=":",
        linewidth=2,
        label=rf"истинное $\theta={theta:g}$",
    )
    axis.axhline(
        theta - epsilon,
        color="tab:red",
        linestyle="--",
        label=rf"границы $\theta\pm{epsilon:g}$",
    )
    axis.axhline(
        theta + epsilon,
        color="tab:red",
        linestyle="--",
    )
    axis.set_xticks(positions, [str(int(n)) for n in sample_sizes])
    axis.set_xlabel("объём выборки n")
    axis.set_ylabel(r"оценка $\widehat\theta$")
    axis.set_title(
        "MAP = MLE: точки, допустимая полоса и число всех превышений"
    )
    axis.legend()

    fig.tight_layout()
    path = output / "threshold_cloud.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_all_metrics_plot(
    experiment: dict,
    theta: float,
    output_dir: str | Path,
) -> Path:
    """Итоговое сравнение всех оценок по трём критериям."""

    output = prepare_directory(output_dir)
    sample_sizes = experiment["sample_sizes"]
    epsilon = experiment["epsilon"]
    metrics = [
        ("mse", "MSE"),
        ("mae", "MAE"),
        (
            "zero_one",
            rf"$P(|\widehat\theta-\theta|>{epsilon:g})$",
        ),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))

    for axis, (metric, ylabel) in zip(axes, metrics):
        for name in ESTIMATOR_LABELS:
            values = []
            for n in sample_sizes:
                current = experiment["results"][float(theta)][int(n)]
                values.append(current["metrics"][name][metric])

            linestyle = "-"
            if name == "zero_one":
                linestyle = "--"
            axis.plot(
                sample_sizes,
                values,
                marker="o",
                linestyle=linestyle,
                color=ESTIMATOR_COLORS[name],
                label=ESTIMATOR_LABELS[name],
            )

        axis.set_xscale("log")
        axis.set_xlabel("объём выборки n")
        axis.set_ylabel(ylabel)

    axes[-1].legend(fontsize=8)
    fig.suptitle(rf"Все оценки при $\theta={theta:g}$")
    fig.tight_layout()

    path = output / "all_metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

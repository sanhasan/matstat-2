"""Графики для оценки параметра Beta(theta, 1) методом моментов."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from utils.moment_estimation import MomentEstimationExperiment


def _prepare_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_sizes(result: MomentEstimationExperiment) -> np.ndarray:
    return np.array(sorted(result.estimates))


def _plot_points(axis, sizes, values_by_size, label):
    for index, n in enumerate(sizes):
        values = values_by_size[int(n)]
        axis.scatter(
            np.full(values.shape, n),
            values,
            color="tab:purple",
            alpha=0.24,
            s=18,
            label=label if index == 0 else None,
        )


def _format_sample_size_axis(axis, sizes: np.ndarray) -> None:
    if sizes[0] >= 100_000:
        axis.set_xticks(sizes)

        def format_value(value: float, _: int) -> str:
            if np.isclose(value, 1_000_000):
                return "10⁶"
            return f"{value / 100_000:g}×10⁵"

        axis.xaxis.set_major_formatter(
            FuncFormatter(format_value)
        )


def _save_statistic_plot(
    result: MomentEstimationExperiment,
    sample_values: np.ndarray,
    theoretical_values: np.ndarray,
    ylabel: str,
    title: str,
    filename: str,
    sample_label: str,
    output_dir: str | Path,
    show_estimates: bool = False,
) -> Path:
    output = _prepare_directory(output_dir)
    sizes = _sample_sizes(result)

    plt.style.use("seaborn-v0_8-whitegrid")
    if show_estimates:
        fig, (estimate_axis, statistic_axis) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        _plot_points(
            estimate_axis,
            sizes,
            result.estimates,
            "отдельные оценки",
        )
        estimate_axis.axhline(
            result.theta,
            color="black",
            linestyle=":",
            label=rf"истинное $\theta={result.theta:g}$",
        )
        estimate_axis.set_ylabel(r"оценка $\widehat{\theta}_n$")
        estimate_axis.legend()
    else:
        fig, statistic_axis = plt.subplots(figsize=(9, 5.2))

    statistic_axis.plot(
        sizes,
        sample_values,
        "o-",
        color="tab:blue",
        linewidth=2,
        label=sample_label,
    )
    statistic_axis.plot(
        sizes,
        theoretical_values,
        "s--",
        color="tab:orange",
        label="точное теоретическое значение",
    )
    statistic_axis.set_xlabel("объём выборки n")
    statistic_axis.set_ylabel(ylabel)
    statistic_axis.legend()
    _format_sample_size_axis(statistic_axis, sizes)
    fig.suptitle(title)
    fig.tight_layout()

    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_beta_density_plot(
    theta: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Построить плотность Beta(theta, 1)."""

    output = _prepare_directory(output_dir)
    grid = np.linspace(0, 1, 800)
    density = theta * grid ** (theta - 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, density, color="tab:blue", linewidth=2)
    axis.fill_between(grid, density, alpha=0.2)
    axis.set_xlabel("t")
    axis.set_ylabel(r"$p_\theta(t)$")
    axis.set_title(rf"Плотность $\mathrm{{Beta}}({theta:g}, 1)$")
    axis.set_xlim(0, 1)
    fig.tight_layout()

    path = output / "density.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_estimator_benchmark_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки, их среднее и истинное значение."""

    output = _prepare_directory(output_dir)
    sizes = _sample_sizes(result)
    sample_means = np.array(
        [result.estimates[int(n)].mean() for n in sizes]
    )
    repetitions = len(result.estimates[int(sizes[0])])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.2))
    _plot_points(axis, sizes, result.estimates, "отдельные оценки")
    axis.plot(
        sizes,
        sample_means,
        "o-",
        color="tab:blue",
        linewidth=2,
        label=f"среднее по {repetitions} оценкам",
    )
    axis.axhline(
        result.theta,
        color="black",
        linestyle=":",
        linewidth=2,
        label=rf"теоретическое значение $\theta={result.theta:g}$",
    )
    axis.set_xlabel("объём выборки n")
    axis.set_ylabel(r"оценка $\widehat{\theta}_n$")
    axis.set_title("Оценка метода моментов при разных объёмах выборки")
    axis.legend()
    _format_sample_size_axis(axis, sizes)
    fig.tight_layout()

    path = output / "estimator_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_bias_comparison_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Сравнить выборочное и точное теоретическое смещение."""

    sizes = _sample_sizes(result)
    sample_bias = np.array(
        [result.estimates[int(n)].mean() - result.theta for n in sizes]
    )
    return _save_statistic_plot(
        result,
        sample_bias,
        np.array([result.theoretical[int(n)].bias for n in sizes]),
        "смещение",
        rf"Смещение оценки, $\theta={result.theta:g}$",
        "bias.png",
        "выборочное смещение",
        output_dir,
    )


def save_variance_comparison_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки и выборочную дисперсию."""

    sizes = _sample_sizes(result)
    sample_variance = np.array(
        [result.estimates[int(n)].var(ddof=1) for n in sizes]
    )
    return _save_statistic_plot(
        result,
        sample_variance,
        np.array([result.theoretical[int(n)].variance for n in sizes]),
        "дисперсия",
        rf"Дисперсия оценки, $\theta={result.theta:g}$",
        "variance.png",
        "выборочная дисперсия",
        output_dir,
        show_estimates=True,
    )


def save_mse_comparison_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Сравнить выборочную и точную теоретическую MSE."""

    sizes = _sample_sizes(result)
    sample_mse = np.array(
        [
            np.mean((result.estimates[int(n)] - result.theta) ** 2)
            for n in sizes
        ]
    )
    theoretical_mse = np.array(
        [result.theoretical[int(n)].mse for n in sizes]
    )
    return _save_statistic_plot(
        result,
        sample_mse,
        theoretical_mse,
        "MSE",
        rf"Среднеквадратическая ошибка, $\theta={result.theta:g}$",
        "mse.png",
        "выборочная MSE",
        output_dir,
    )


def save_estimate_distributions_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Построить распределения оценки для трёх объёмов выборки."""

    output = _prepare_directory(output_dir)
    sizes = sorted(result.estimates)
    selected_sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, n in zip(axes, selected_sizes):
        axis.hist(result.estimates[n], bins=15, alpha=0.75)
        axis.axvline(
            result.theta,
            color="black",
            linestyle="--",
            label=rf"истинное $\theta={result.theta:g}$",
        )
        axis.set_title(f"n = {n}")
        axis.set_xlabel(r"$\widehat{\theta}$")
        axis.legend()
    fig.suptitle("Распределение оценки")
    fig.tight_layout()

    path = output / "estimate_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_threshold_exceedance_plot(
    result: MomentEstimationExperiment,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки и выборочную долю превышений."""

    output = _prepare_directory(output_dir)
    sizes = _sample_sizes(result)
    counts = np.array(
        [
            np.count_nonzero(
                np.abs(result.estimates[int(n)] - result.theta)
                > result.threshold
            )
            for n in sizes
        ]
    )
    repetitions = len(result.estimates[int(sizes[0])])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (estimate_axis, share_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    _plot_points(
        estimate_axis,
        sizes,
        result.estimates,
        "отдельные оценки",
    )
    estimate_axis.axhline(
        result.theta,
        color="black",
        linestyle=":",
        label=rf"истинное $\theta={result.theta:g}$",
    )
    estimate_axis.axhspan(
        result.theta - result.threshold,
        result.theta + result.threshold,
        color="tab:green",
        alpha=0.12,
        label="допустимый интервал",
    )
    estimate_axis.set_ylabel(r"оценка $\widehat{\theta}_n$")
    estimate_axis.legend()

    share_axis.plot(
        sizes,
        counts,
        "o-",
        color="tab:blue",
        linewidth=2,
        label="число превышений",
    )
    for n, count in zip(sizes, counts):
        share_axis.annotate(
            str(count),
            (n, count),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    share_axis.set_xlabel("объём выборки n")
    share_axis.set_ylabel("число превышений")
    share_axis.set_ylim(-0.5, repetitions + 0.5)
    share_axis.legend()
    _format_sample_size_axis(share_axis, sizes)
    fig.suptitle(
        rf"$|\widehat{{\theta}}-\theta|>{result.threshold:g}$"
    )
    fig.tight_layout()

    path = output / "threshold_exceedance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

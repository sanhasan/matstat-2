"""Графики результатов эксперимента."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def _prepare_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_sizes(result: dict) -> np.ndarray:
    return np.array(sorted(result["results"]))


def _plot_points(axis, sizes, results_by_size, label):
    for index, n in enumerate(sizes):
        results = results_by_size[int(n)]
        point_label = None
        if index == 0:
            point_label = label

        axis.scatter(
            np.full(results.shape, n),
            results,
            color="tab:purple",
            alpha=0.24,
            s=18,
            label=point_label,
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
    result: dict,
    theta: float,
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
            result["results"],
            "отдельные оценки",
        )
        estimate_axis.axhline(
            theta,
            color="black",
            linestyle=":",
            label=rf"истинное $\theta={theta:g}$",
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
        label="асимптотическое значение",
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
    result: dict,
    theta: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки, их среднее и истинное значение."""

    output = _prepare_directory(output_dir)
    sizes = _sample_sizes(result)
    sample_means = []
    for n in sizes:
        mean = result["results"][int(n)].mean()
        sample_means.append(mean)
    sample_means = np.array(sample_means)

    repetitions = len(result["results"][int(sizes[0])])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.2))
    _plot_points(axis, sizes, result["results"], "отдельные оценки")
    axis.plot(
        sizes,
        sample_means,
        "o-",
        color="tab:blue",
        linewidth=2,
        label=f"среднее по {repetitions} оценкам",
    )
    axis.axhline(
        theta,
        color="black",
        linestyle=":",
        linewidth=2,
        label=rf"теоретическое значение $\theta={theta:g}$",
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
    result: dict,
    theta: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Сравнить смещение в эксперименте и его асимптотику."""

    sizes = _sample_sizes(result)
    experiment_bias = []
    asymptotic_bias = []
    for n in sizes:
        n = int(n)
        mean_theta_hat = result["results"][n].mean()
        bias = mean_theta_hat - theta
        experiment_bias.append(bias)

        bias = result["theory"][n]["bias"]
        asymptotic_bias.append(bias)

    experiment_bias = np.array(experiment_bias)
    asymptotic_bias = np.array(asymptotic_bias)

    return _save_statistic_plot(
        result,
        theta,
        experiment_bias,
        asymptotic_bias,
        "смещение",
        rf"Смещение оценки, $\theta={theta:g}$",
        "bias.png",
        "смещение по результатам эксперимента",
        output_dir,
    )


def save_variance_comparison_plot(
    result: dict,
    theta: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки и выборочную дисперсию S_M^2."""

    sizes = _sample_sizes(result)
    sample_variance = []
    theoretical_variance = []
    for n in sizes:
        n = int(n)
        variance = result["results"][n].var(ddof=0)
        sample_variance.append(variance)
        theoretical_variance.append(
            result["theory"][n]["variance"]
        )

    sample_variance = np.array(sample_variance)
    theoretical_variance = np.array(theoretical_variance)

    return _save_statistic_plot(
        result,
        theta,
        sample_variance,
        theoretical_variance,
        "дисперсия",
        rf"Дисперсия оценки, $\theta={theta:g}$",
        "variance.png",
        r"$S_M^2$ по результатам эксперимента",
        output_dir,
        show_estimates=True,
    )


def save_mse_comparison_plot(
    result: dict,
    theta: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Сравнить MSE в эксперименте и её асимптотику."""

    sizes = _sample_sizes(result)
    sample_mse = []
    theoretical_mse = []
    for n in sizes:
        n = int(n)
        errors = result["results"][n] - theta
        mse = np.mean(errors**2)
        sample_mse.append(mse)
        theoretical_mse.append(result["theory"][n]["mse"])

    sample_mse = np.array(sample_mse)
    theoretical_mse = np.array(theoretical_mse)

    return _save_statistic_plot(
        result,
        theta,
        sample_mse,
        theoretical_mse,
        "MSE",
        rf"Среднеквадратическая ошибка, $\theta={theta:g}$",
        "mse.png",
        "выборочная MSE",
        output_dir,
    )


def save_threshold_exceedance_plot(
    result: dict,
    theta: float,
    threshold: float,
    output_dir: str | Path = "images/moment_estimation",
) -> Path:
    """Показать отдельные оценки и выборочную долю превышений."""

    output = _prepare_directory(output_dir)
    sizes = _sample_sizes(result)
    counts = []
    for n in sizes:
        n = int(n)
        errors = np.abs(result["results"][n] - theta)
        count = np.count_nonzero(errors > threshold)
        counts.append(count)
    counts = np.array(counts)

    repetitions = len(result["results"][int(sizes[0])])

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
        result["results"],
        "отдельные оценки",
    )
    estimate_axis.axhline(
        theta,
        color="black",
        linestyle=":",
        label=rf"истинное $\theta={theta:g}$",
    )
    estimate_axis.axhspan(
        theta - threshold,
        theta + threshold,
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
        rf"$|\widehat{{\theta}}-\theta|>{threshold:g}$"
    )
    fig.tight_layout()

    path = output / "threshold_exceedance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

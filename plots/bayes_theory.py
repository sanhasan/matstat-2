"""Графики к теоретическим выкладкам второго задания."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distribution

from plots.bayes_style import (
    ESTIMATOR_COLORS,
    ESTIMATOR_LABELS,
    prepare_directory,
)
from utils.bayes_estimates import find_estimates


def save_distribution_plot(
    theta_values: np.ndarray,
    a: float,
    b: float,
    output_dir: str | Path,
) -> Path:
    """Геометрические распределения и априорная плотность."""

    output = prepare_directory(output_dir)
    x = np.arange(16)
    theta_grid = np.linspace(0.001, 0.999, 800)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (model_axis, prior_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    for theta in theta_values:
        probabilities = theta * (1 - theta) ** x
        model_axis.plot(
            x,
            probabilities,
            "o-",
            label=rf"$\theta={theta:g}$",
        )

    model_axis.set_xlabel("число неудач x до первого успеха")
    model_axis.set_ylabel(r"$p_\theta(x)$")
    model_axis.set_title("Геометрическое распределение")
    model_axis.legend()

    prior_density = beta_distribution.pdf(theta_grid, a, b)
    prior_axis.plot(
        theta_grid,
        prior_density,
        color="tab:green",
        linewidth=2,
    )
    prior_axis.fill_between(
        theta_grid,
        prior_density,
        color="tab:green",
        alpha=0.18,
    )
    prior_axis.set_xlabel(r"$\theta$")
    prior_axis.set_ylabel(r"$\pi(\theta)$")
    prior_axis.set_title(rf"Априорное $\mathrm{{Beta}}({a:g},{b:g})$")
    prior_axis.set_xlim(0, 1)

    fig.tight_layout()
    path = output / "distribution_and_prior.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_posterior_plot(
    theta: float,
    sample_sizes: np.ndarray,
    a: float,
    b: float,
    output_dir: str | Path,
) -> Path:
    """Показать, как апостериорная плотность сжимается около theta."""

    output = prepare_directory(output_dir)
    grid = np.linspace(0.001, 0.999, 1000)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(
        grid,
        beta_distribution.pdf(grid, a, b),
        color="gray",
        linestyle="--",
        linewidth=2,
        label=rf"приор $\mathrm{{Beta}}({a:g},{b:g})$",
    )

    for n in sample_sizes:
        n = int(n)
        expected_sum = int(round(n * (1 - theta) / theta))
        alpha = a + n
        beta = b + expected_sum
        density = beta_distribution.pdf(grid, alpha, beta)
        axis.plot(
            grid,
            density,
            linewidth=2,
            label=rf"$n={n},\ S={expected_sum}$",
        )

    axis.axvline(
        theta,
        color="black",
        linestyle=":",
        linewidth=2,
        label=rf"истинное $\theta={theta:g}$",
    )
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel(r"$\pi(\theta\mid x)$")
    axis.set_title("Концентрация апостериорного распределения")
    axis.set_xlim(0, 1)
    axis.legend()

    fig.tight_layout()
    path = output / "posterior_concentration.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_squared_loss_plot(
    n: int,
    total_failures: int,
    a: float,
    b: float,
    output_dir: str | Path,
) -> Path:
    """Апостериорный риск при квадратичной потере."""

    output = prepare_directory(output_dir)
    alpha = a + n
    beta = b + total_failures
    posterior_mean = alpha / (alpha + beta)
    posterior_variance = (
        alpha * beta
        / ((alpha + beta) ** 2 * (alpha + beta + 1))
    )
    decisions = np.linspace(0, 1, 800)
    risk = posterior_variance + (decisions - posterior_mean) ** 2

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(decisions, risk, color="tab:blue", linewidth=2)
    axis.axvline(
        posterior_mean,
        color="tab:blue",
        linestyle="--",
        label=rf"минимум: $d={posterior_mean:.4f}$",
    )
    axis.scatter(
        posterior_mean,
        posterior_variance,
        color="tab:blue",
        s=70,
        zorder=3,
    )
    axis.set_xlabel("решение d")
    axis.set_ylabel(r"$\rho_2(d\mid x)$")
    axis.set_title(
        rf"Квадратичный апостериорный риск, $n={n},\ S={total_failures}$"
    )
    axis.legend()

    fig.tight_layout()
    path = output / "squared_posterior_risk.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_absolute_loss_plot(
    n: int,
    total_failures: int,
    a: float,
    b: float,
    output_dir: str | Path,
) -> Path:
    """Апостериорный риск при абсолютной потере."""

    output = prepare_directory(output_dir)
    alpha = a + n
    beta = b + total_failures
    posterior_mean = alpha / (alpha + beta)
    posterior_median = beta_distribution.ppf(0.5, alpha, beta)
    decisions = np.linspace(0, 1, 800)

    cdf = beta_distribution.cdf(decisions, alpha, beta)
    partial_mean = posterior_mean * beta_distribution.cdf(
        decisions,
        alpha + 1,
        beta,
    )
    risk = (
        decisions * (2 * cdf - 1)
        + posterior_mean
        - 2 * partial_mean
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(decisions, risk, color="tab:orange", linewidth=2)
    axis.axvline(
        posterior_median,
        color="tab:orange",
        linestyle="--",
        label=rf"минимум: $d={posterior_median:.4f}$",
    )
    minimum_risk = np.interp(posterior_median, decisions, risk)
    axis.scatter(
        posterior_median,
        minimum_risk,
        color="tab:orange",
        s=70,
        zorder=3,
    )
    axis.set_xlabel("решение d")
    axis.set_ylabel(r"$\rho_1(d\mid x)$")
    axis.set_title(
        rf"Абсолютный апостериорный риск, $n={n},\ S={total_failures}$"
    )
    axis.legend()

    fig.tight_layout()
    path = output / "absolute_posterior_risk.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_zero_one_loss_plot(
    n: int,
    total_failures: int,
    a: float,
    b: float,
    epsilon: float,
    output_dir: str | Path,
) -> Path:
    """Плотность и риск непрерывной epsilon-версии 0–1 потери."""

    output = prepare_directory(output_dir)
    alpha = a + n
    beta = b + total_failures
    mode = (alpha - 1) / (alpha + beta - 2)
    decisions = np.linspace(0, 1, 1200)
    density = beta_distribution.pdf(decisions, alpha, beta)

    left = np.clip(decisions - epsilon, 0, 1)
    right = np.clip(decisions + epsilon, 0, 1)
    interval_probability = (
        beta_distribution.cdf(right, alpha, beta)
        - beta_distribution.cdf(left, alpha, beta)
    )
    risk = 1 - interval_probability
    epsilon_estimate = decisions[np.argmin(risk)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (density_axis, risk_axis) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7),
        sharex=True,
    )

    density_axis.plot(
        decisions,
        density,
        color="tab:purple",
        linewidth=2,
    )
    density_axis.axvline(
        mode,
        color="tab:purple",
        linestyle="--",
        label=rf"MAP: $d={mode:.4f}$",
    )
    density_axis.set_ylabel(r"$\pi(\theta\mid x)$")
    density_axis.set_title(
        rf"Предельная 0–1 оценка, $n={n},\ S={total_failures}$"
    )
    density_axis.legend()

    risk_axis.plot(
        decisions,
        risk,
        color="tab:red",
        linewidth=2,
    )
    risk_axis.axvline(
        epsilon_estimate,
        color="tab:red",
        linestyle="--",
        label=rf"минимум при $\varepsilon={epsilon:g}$: "
        rf"$d={epsilon_estimate:.4f}$",
    )
    risk_axis.axvline(
        mode,
        color="tab:purple",
        linestyle=":",
        label=r"MAP — предел при $\varepsilon\to0$",
    )
    risk_axis.set_xlabel("решение d")
    risk_axis.set_ylabel(r"$\rho_{0-1,\varepsilon}(d\mid x)$")
    risk_axis.legend()

    fig.tight_layout()
    path = output / "zero_one_posterior_risk.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_estimator_formula_plot(
    n: int,
    max_total_failures: int,
    a: float,
    b: float,
    output_dir: str | Path,
) -> Path:
    """Сравнить формулы оценок при разных значениях достаточной статистики."""

    output = prepare_directory(output_dir)
    total_failures = np.arange(max_total_failures + 1)
    estimates = find_estimates(total_failures, n, a, b)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.2))

    for name in estimates:
        linestyle = "-"
        linewidth = 2
        if name == "zero_one":
            linestyle = "--"
            linewidth = 2.5
        axis.plot(
            total_failures,
            estimates[name],
            color=ESTIMATOR_COLORS[name],
            linestyle=linestyle,
            linewidth=linewidth,
            label=ESTIMATOR_LABELS[name],
        )

    axis.set_xlabel(r"сумма $S=\sum X_i$")
    axis.set_ylabel(r"оценка $\widehat\theta$")
    axis.set_title(rf"Зависимость оценок от достаточной статистики, $n={n}$")
    axis.legend()

    fig.tight_layout()
    path = output / "estimators_by_sum.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path



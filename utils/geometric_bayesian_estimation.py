"""Байесовские оценки параметра геометрического распределения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


ESTIMATOR_LABELS = {
    "mle": "MLE",
    "bayes_squared": "Bayes, квадратичная",
    "bayes_absolute": "Bayes, абсолютная",
    "bayes_zero_one": "Bayes, 0–1 (MAP)",
}


@dataclass(frozen=True)
class BayesianEstimationExperiment:
    """Результаты сравнения байесовских оценок с MLE."""

    prior_a: float
    prior_b: float
    repetitions: int
    epsilon: float
    summary: pd.DataFrame
    estimates: dict[tuple[float, int], dict[str, np.ndarray]]


def mle_estimate(total_failures: np.ndarray, n: int) -> np.ndarray:
    """MLE параметра успеха для Geom(theta) на {0, 1, ...}."""

    return n / (n + total_failures)


def posterior_parameters(
    total_failures: np.ndarray,
    n: int,
    prior_a: float,
    prior_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Параметры Beta-апостериорного распределения."""

    alpha = np.full_like(total_failures, prior_a + n, dtype=float)
    beta = prior_b + total_failures.astype(float)
    return alpha, beta


def posterior_mode(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Одна из мод Beta(alpha, beta), включая граничные случаи."""

    mode = np.empty_like(alpha, dtype=float)
    interior = (alpha > 1.0) & (beta > 1.0)
    mode[interior] = (alpha[interior] - 1.0) / (
        alpha[interior] + beta[interior] - 2.0
    )

    uniform = np.isclose(alpha, 1.0) & np.isclose(beta, 1.0)
    left = (alpha < 1.0) & (beta >= 1.0)
    left |= np.isclose(alpha, 1.0) & (beta > 1.0)
    right = (alpha >= 1.0) & (beta < 1.0)
    right |= (alpha > 1.0) & np.isclose(beta, 1.0)
    two_boundary_modes = (alpha < 1.0) & (beta < 1.0)
    mode[left] = 0.0
    mode[right] = 1.0
    mode[two_boundary_modes] = 0.0
    mode[uniform] = 0.5
    return mode


def calculate_bayesian_estimates(
    total_failures: np.ndarray,
    n: int,
    prior_a: float,
    prior_b: float,
) -> dict[str, np.ndarray]:
    """Оценки для квадратичной, абсолютной и предельной 0–1 потерь."""

    alpha, beta = posterior_parameters(
        total_failures,
        n,
        prior_a,
        prior_b,
    )
    return {
        "mle": mle_estimate(total_failures, n),
        "bayes_squared": alpha / (alpha + beta),
        "bayes_absolute": beta_distribution.ppf(0.5, alpha, beta),
        "bayes_zero_one": posterior_mode(alpha, beta),
    }


def run_bayesian_comparison_experiment(
    true_thetas: Iterable[float],
    sample_sizes: Iterable[int],
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    repetitions: int = 20_000,
    epsilon: float = 0.05,
    seed: int = 467_394 + 467_920,
) -> BayesianEstimationExperiment:
    """Сравнить оценки по MSE, MAE и epsilon-версии 0–1 риска."""

    thetas = tuple(float(theta) for theta in true_thetas)
    sizes = tuple(int(n) for n in sample_sizes)
    if any(not 0.0 < theta < 1.0 for theta in thetas):
        raise ValueError("Каждое theta должно лежать в (0, 1)")
    if any(n <= 0 for n in sizes):
        raise ValueError("Объёмы выборки должны быть положительными")
    if prior_a <= 0 or prior_b <= 0:
        raise ValueError("Параметры Beta-приора должны быть положительными")

    rng = np.random.default_rng(seed)
    estimates: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | int | str]] = []

    for theta in thetas:
        for n in sizes:
            total_failures = rng.negative_binomial(
                n=n,
                p=theta,
                size=repetitions,
            )
            current = calculate_bayesian_estimates(
                total_failures,
                n,
                prior_a,
                prior_b,
            )
            estimates[(theta, n)] = current

            for estimator, values in current.items():
                errors = values - theta
                rows.append(
                    {
                        "theta": theta,
                        "n": n,
                        "estimator": estimator,
                        "label": ESTIMATOR_LABELS[estimator],
                        "bias": float(np.mean(errors)),
                        "mse": float(np.mean(errors**2)),
                        "mae": float(np.mean(np.abs(errors))),
                        "epsilon_zero_one_risk": float(
                            np.mean(np.abs(errors) > epsilon)
                        ),
                    }
                )

    return BayesianEstimationExperiment(
        prior_a=prior_a,
        prior_b=prior_b,
        repetitions=repetitions,
        epsilon=epsilon,
        summary=pd.DataFrame(rows),
        estimates=estimates,
    )


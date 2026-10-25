"""Оценки параметра геометрического распределения."""

import numpy as np
from scipy.stats import beta as beta_distribution


def find_mle(total_failures: np.ndarray, n: int) -> np.ndarray:
    """Оценка максимального правдоподобия."""

    return n / (n + total_failures)


def find_posterior_parameters(
    total_failures: np.ndarray,
    n: int,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Параметры Beta-распределения после наблюдения выборки."""

    alpha = np.full(total_failures.shape, a + n, dtype=float)
    beta = b + total_failures
    return alpha, beta


def find_posterior_mode(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Мода Beta-распределения с учётом граничных случаев."""

    mode = np.empty(alpha.shape, dtype=float)

    inside = (alpha > 1) & (beta > 1)
    mode[inside] = (
        (alpha[inside] - 1)
        / (alpha[inside] + beta[inside] - 2)
    )

    uniform = (alpha == 1) & (beta == 1)
    mode[uniform] = 0.5

    left_boundary = (alpha <= 1) & (beta >= 1) & ~uniform
    mode[left_boundary] = 0.0

    right_boundary = (alpha >= 1) & (beta <= 1) & ~uniform
    mode[right_boundary] = 1.0

    two_boundaries = (alpha < 1) & (beta < 1)
    mode[two_boundaries] = 0.0

    return mode


def find_estimates(
    total_failures: np.ndarray,
    n: int,
    a: float,
    b: float,
) -> dict[str, np.ndarray]:
    """MLE и три байесовские оценки."""

    alpha, beta = find_posterior_parameters(
        total_failures,
        n,
        a,
        b,
    )

    estimates = {
        "mle": find_mle(total_failures, n),
        "squared": alpha / (alpha + beta),
        "absolute": beta_distribution.ppf(0.5, alpha, beta),
        "zero_one": find_posterior_mode(alpha, beta),
    }

    return estimates

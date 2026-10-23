"""Оценка параметра Beta(theta, 1) методом моментов."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp1f1


@dataclass(frozen=True)
class TheoreticalCharacteristics:
    """Точные теоретические характеристики оценки."""

    bias: float
    variance: float
    mse: float


@dataclass(frozen=True)
class MomentEstimationExperiment:
    """Оценки параметра, полученные для каждого объёма выборки."""

    theta: float
    threshold: float
    estimates: dict[int, np.ndarray]
    theoretical: dict[int, TheoreticalCharacteristics]


def estimate_beta_shape_from_mean(samples: np.ndarray) -> np.ndarray:
    """Посчитать оценку theta_hat по каждой строке выборок."""

    sample_means = samples.mean(axis=1)
    return sample_means / (1 - sample_means)


def _laplace_transform(theta: float, s: float) -> float:
    """Преобразование Лапласа величины 1 - X."""

    if s < 1e-3:
        value = 1.0
        numerator = 1.0
        denominator = 1.0
        for k in range(1, 7):
            numerator *= -s
            denominator *= theta + k
            value += numerator / denominator
        return value
    return float(hyp1f1(1, theta + 1, -s))


def exact_characteristics(
    theta: float,
    n: int,
) -> TheoreticalCharacteristics:
    """Вычислить точные характеристики через обратные моменты суммы."""

    def laplace_power(u: float) -> float:
        value = _laplace_transform(theta, u / n)
        return float(np.exp(n * np.log(value)))

    first_scaled_moment = quad(
        laplace_power,
        0,
        np.inf,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=200,
    )[0]
    second_scaled_moment = quad(
        lambda u: u * laplace_power(u),
        0,
        np.inf,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=200,
    )[0]

    bias = first_scaled_moment - 1 - theta
    variance = second_scaled_moment - first_scaled_moment**2
    mse = variance + bias**2
    return TheoreticalCharacteristics(bias, variance, mse)


def run_monte_carlo_experiment(
    theta: float,
    sample_sizes: Iterable[int],
    repetitions: int,
    threshold: float,
    seed: int,
) -> MomentEstimationExperiment:
    """Сгенерировать выборки и посчитать оценки параметра."""

    rng = np.random.default_rng(seed)
    estimates = {
        int(n): estimate_beta_shape_from_mean(
            rng.beta(theta, 1, size=(repetitions, int(n)))
        )
        for n in sample_sizes
    }
    theoretical = {
        n: exact_characteristics(theta, n)
        for n in estimates
    }
    return MomentEstimationExperiment(
        theta,
        threshold,
        estimates,
        theoretical,
    )

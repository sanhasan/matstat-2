"""Вычислительный эксперимент для байесовских оценок."""

import numpy as np

from utils.bayes_estimates import find_estimates


def calculate_metrics(
    estimates: np.ndarray,
    theta: float,
    epsilon: float,
) -> dict[str, float | int]:
    """Характеристики ошибки оценки."""

    errors = estimates - theta

    return {
        "bias": float(errors.mean()),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "zero_one": float(np.mean(np.abs(errors) > epsilon)),
        "exceedances": int(np.sum(np.abs(errors) > epsilon)),
    }


def run_bayes_experiment(
    theta_values: np.ndarray,
    sample_sizes: np.ndarray,
    repetitions: int,
    a: float,
    b: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    """Сгенерировать выборки и сравнить оценки на одних данных."""

    results = {}

    for theta in theta_values:
        theta = float(theta)
        results[theta] = {}

        for n in sample_sizes:
            n = int(n)
            total_failures = rng.negative_binomial(
                n,
                theta,
                size=repetitions,
            )
            estimates = find_estimates(
                total_failures,
                n,
                a,
                b,
            )

            metrics = {}
            for name in estimates:
                metrics[name] = calculate_metrics(
                    estimates[name],
                    theta,
                    epsilon,
                )

            results[theta][n] = {
                "total_failures": total_failures,
                "estimates": estimates,
                "metrics": metrics,
            }

    return {
        "theta_values": theta_values,
        "sample_sizes": sample_sizes,
        "repetitions": repetitions,
        "a": a,
        "b": b,
        "epsilon": epsilon,
        "results": results,
    }

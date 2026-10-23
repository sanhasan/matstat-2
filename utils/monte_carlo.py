import numpy as np

from utils.theta_hat import find_theta_hat
from utils.theory import asymptotic_values


def run_monte_carlo(
    theta: float,
    sizes: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> dict:
    results = {}
    for n in sizes:
        n = int(n)
        samples = rng.beta(theta, 1, size=(iters, n))
        results[n] = find_theta_hat(samples)

    theory = {}
    for n in results:
        theory[n] = asymptotic_values(theta, n)

    return {
        "results": results,
        "theory": theory,
    }

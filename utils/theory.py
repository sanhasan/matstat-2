def asymptotic_values(theta: float, n: int) -> dict[str, float]:
    bias = theta * (theta + 1) / ((theta + 2) * n)
    expectation = theta + bias
    variance = theta * (theta + 1) ** 2 / ((theta + 2) * n)
    mse = variance

    return {
        "expectation": expectation,
        "bias": bias,
        "variance": variance,
        "mse": mse,
    }

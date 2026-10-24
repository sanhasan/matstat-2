def asymptotic_values(theta: float, n: int) -> dict[str, float]:
    bias = theta * (theta + 1) / ((theta + 2) * n)
    variance = theta * (theta + 1) ** 2 / ((theta + 2) * n)
    mse = variance

    return {
        "bias": bias,
        "variance": variance,
        "mse": mse,
    }

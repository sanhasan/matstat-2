"""Общие подписи и цвета для графиков второго задания."""

from pathlib import Path


ESTIMATOR_LABELS = {
    "mle": "MLE",
    "squared": "байесовская для квадратичной потери",
    "absolute": "байесовская для абсолютной потери",
    "zero_one": "байесовская для 0–1 потери (MAP)",
}

ESTIMATOR_COLORS = {
    "mle": "black",
    "squared": "tab:blue",
    "absolute": "tab:orange",
    "zero_one": "tab:purple",
}


def prepare_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

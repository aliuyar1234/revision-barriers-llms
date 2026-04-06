from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_C_GRID = [0.01, 0.1, 1.0, 10.0]


@dataclass
class StandardScaler1D:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def fit_standard_scaler(train_x: np.ndarray, epsilon: float = 1e-6) -> StandardScaler1D:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < epsilon, epsilon, std)
    return StandardScaler1D(mean=mean, std=std)


def fit_logistic_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
        class_weight=None,
        random_state=0,
    )
    model.fit(train_x, train_y)
    return model


def logistic_scores(model: LogisticRegression, x_values: np.ndarray) -> np.ndarray:
    return model.decision_function(x_values)


def select_logistic_by_dev(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    c_grid: list[float] | None = None,
) -> dict[str, Any]:
    if c_grid is None:
        c_grid = DEFAULT_C_GRID

    scaler = fit_standard_scaler(train_x)
    train_scaled = scaler.transform(train_x)
    dev_scaled = scaler.transform(dev_x)

    candidates: list[dict[str, Any]] = []
    for c_value in c_grid:
        model = fit_logistic_model(train_scaled, train_y, c_value)
        dev_scores = logistic_scores(model, dev_scaled)
        candidates.append(
            {
                "C": float(c_value),
                "model": model,
                "dev_scores": dev_scores,
                "dev_auroc": safe_auroc(dev_y, dev_scores),
                "dev_auprc": safe_auprc(dev_y, dev_scores),
            }
        )

    best_auroc = max(candidate["dev_auroc"] for candidate in candidates)
    near_best = [candidate for candidate in candidates if best_auroc - candidate["dev_auroc"] <= 0.002]
    selected = min(near_best, key=lambda candidate: candidate["C"])
    return {
        "selected_C": selected["C"],
        "selected_model": selected["model"],
        "selected_dev_scores": selected["dev_scores"],
        "selected_dev_auroc": selected["dev_auroc"],
        "selected_dev_auprc": selected["dev_auprc"],
        "scaler": scaler,
        "candidates": [
            {
                "C": candidate["C"],
                "dev_auroc": candidate["dev_auroc"],
                "dev_auprc": candidate["dev_auprc"],
            }
            for candidate in candidates
        ],
    }


def refit_logistic_on_train_dev(
    train_dev_x: np.ndarray,
    train_dev_y: np.ndarray,
    selected_c: float,
) -> tuple[LogisticRegression, StandardScaler1D]:
    scaler = fit_standard_scaler(train_dev_x)
    train_dev_scaled = scaler.transform(train_dev_x)
    model = fit_logistic_model(train_dev_scaled, train_dev_y, selected_c)
    return model, scaler


def evaluate_logistic_predictions(
    model: LogisticRegression,
    scaler: StandardScaler1D,
    x_values: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    x_scaled = scaler.transform(x_values)
    scores = logistic_scores(model, x_scaled)
    return {
        "scores": scores,
        "auroc": safe_auroc(y_true, scores),
        "auprc": safe_auprc(y_true, scores),
    }


def bootstrap_metric_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    metric: str = "auroc",
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)

    if metric == "auroc":
        metric_fn = safe_auroc
    elif metric == "auprc":
        metric_fn = safe_auprc
    else:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")

    base_metric_a = metric_fn(y_true, score_a)
    base_metric_b = metric_fn(y_true, score_b)
    base_delta = base_metric_a - base_metric_b

    if math.isnan(base_delta):
        return {
            "metric": metric,
            "mean_delta": float("nan"),
            "ci": {"low": float("nan"), "high": float("nan")},
            "n_resamples": 0,
        }

    estimates: list[float] = []
    n_examples = len(y_true)
    for _ in range(n_resamples):
        sample_index = rng.integers(0, n_examples, size=n_examples)
        sample_y = y_true[sample_index]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_a = score_a[sample_index]
        sample_b = score_b[sample_index]
        estimates.append(metric_fn(sample_y, sample_a) - metric_fn(sample_y, sample_b))

    if not estimates:
        return {
            "metric": metric,
            "mean_delta": base_delta,
            "ci": {"low": float("nan"), "high": float("nan")},
            "n_resamples": 0,
        }

    ordered = sorted(estimates)
    low_index = int(math.floor(0.025 * (len(ordered) - 1)))
    high_index = int(math.ceil(0.975 * (len(ordered) - 1)))
    return {
        "metric": metric,
        "mean_delta": base_delta,
        "ci": {"low": ordered[low_index], "high": ordered[high_index]},
        "n_resamples": len(ordered),
    }

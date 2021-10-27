#!/usr/bin/env python3
"""
A collection of functions that take estimates and true labels, returning useful
model evaluation metrics, and helper functions to parse particularly complex
returning payloads.

Invoke a demo of training a model by running a python interpreter with access to
this module as `$ python -m gizmo.metrics`.
"""

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score as roc,
)

from sklearn.calibration import calibration_curve

from . import log

import numpy as np
import pandas as pd


def estimate_threshold(y_probs, y_labels, target_recall):
    """Return the score threshold value that mostly closey achieves the desired
    recall."""
    # Estimate recall values
    _, recalls, thresholds = precision_recall_curve(y_labels, y_probs)

    # Thresholds are ordered increasing with index, so subtracting one
    # guarantees the target threshold produces a recall greater than the
    # target.
    target_threshold_index = np.argmin(np.abs(recalls - target_recall)) - 1
    threshold = thresholds[max(target_threshold_index, 0)]
    recall_at_threshold = recalls[max(target_threshold_index, 0)]

    # The percent sent to manual review is equal to the fraction of predicted
    # true values.
    percent_predicted_true = (y_probs > threshold).mean()

    recall_info = dict(
        threshold=threshold,
        recall_at_threshold=recall_at_threshold,
        percent_predicted_true=percent_predicted_true,
    )

    return recall_info


def calculate_binary_classification_metrics(
    y_true=None, y_score=None, y_hat=None, **kwargs
):
    """
    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.

    y_hat : array, shape = [n_samples] or [n_samples, n_classes]
        y_score > threshold = yhat: predicted binary labels, which are then
        consumed by classification metrics.
    """

    n_top_ten = int(len(y_score) * 0.1)
    inds_by_score = np.argsort(y_score)[::-1][:n_top_ten]

    return {
        "n": len(y_true),
        "y_true_mean": y_true.mean(),
        "auc": roc(y_true, y_score),
        "accuracy": (y_true == y_hat).astype(int).mean(),
        "recall": recall_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat),
        "precision_at_90th_score_percentile": y_true[inds_by_score].mean(),
        "recall_at_90th_score_percentile": (y_true[inds_by_score].sum() / y_true.sum()),
        "calibration": calibration_curve(y_true, y_score, n_bins=10),
        "brier_score": brier_score_loss(y_true, y_score),
        "prob_pos": y_score,
    }


# TODO Support linear models / multiple model types?
def collect_feature_data(model):
    # NOTE Assumes a pipeline object or dict-like interface.
    try:
        fimp = getattr(model["estimator"], "feature_importances_")
    except (AttributeError, TypeError) as e:
        log.error(
            f"Provided model of type {type(model)} does not support `feature_importance_` attribute."
        )
        return None

    return fimp


def collect_roc_curve_data(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def collect_precision_recall_curve_data(y_true, y_score):
    prc, rcl, thresholds = precision_recall_curve(y_true, y_score)
    return {
        "prc": prc,
        "rcl": rcl,
        "thresholds": thresholds,
    }


def collect_hit_rate_curve_data(y_true, y_score):
    hit_rate_df = pd.DataFrame({"probability": y_score, "truth": y_true})
    hit_rate_df.sort_values(by="probability", ascending=False, inplace=True)
    hit_rate_df["hits"] = hit_rate_df["truth"].cumsum()
    hit_rate_df.index = range(1, len(hit_rate_df) + 1)
    hit_rate_df["hit_rate"] = [
        hit_rate_df.loc[i, "hits"] / i for i in hit_rate_df.index
    ]
    hit_rate_df = hit_rate_df.reset_index().rename(columns={"index": "Digs"})
    return {column: hit_rate_df[column].values for column in hit_rate_df.columns}


def collect_curve_data(model, y_true, y_score):
    """Returns a dictionary of vectors required to generate common charts for
    binary classification. This includes ROC and PR Curves"""
    return {
        "roc_curve": collect_roc_curve_data(y_true, y_score),
        "precision_recall": collect_precision_recall_curve_data(y_true, y_score),
        "hit_rate": collect_hit_rate_curve_data(y_true, y_score),
    }


def consolidate_metrics(metrics_list):
    """Take a list of dictionaries with key-value pairs of metrics such as
    `calculate_binary_classification_metrics` and return summary statistics.
    Useful for reducing lists of dictionaries returned from different
    cross-validation folds."""
    return (
        pd.DataFrame(metrics_list)
        .describe()
        .drop("n", axis="columns")
        .drop("count", axis="rows")
    )


if __name__ == "__main__":
    from .bc_logger import get_simple_logger
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    log = get_simple_logger(__name__)
    data = load_iris()

    # Form binary problem from ternary by masking with membership in first class
    X = data.data
    y = data.target == 0

    log.info("Training Model on Iris Dataset")

    lr = LogisticRegression().fit(X, y)

    # Not worried about train test splitting here; merely demonstrating the metrics.
    y_true, y_score, y_hat = y, lr.predict_proba(X)[:, 1], lr.predict(X)
    metrics = calculate_binary_classification_metrics(y_true, y_score, y_hat)

    log.info("Scores", **metrics)

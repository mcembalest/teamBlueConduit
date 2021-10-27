#!/usr/bin/env python3
"""
Model evaluation utilities, comprising resampling and testing schemes like cross
validation, bootstrapping, and single train test split runs.
"""

import pandas as pd
import geopandas as gpd
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    KFold,
    StratifiedKFold,
)
from typing import Dict, Any
from gizmo.metrics import (
    calculate_binary_classification_metrics,
    collect_curve_data,
    collect_feature_data,
)
from functools import partial

import logging

log = logging.getLogger(__name__)


def test_model(model, X, y, test_size=0.2) -> Dict[str, Any]:
    """Take a model and a dataset. Split the data and return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return calculate_binary_classification_metrics(y_test, y_score, y_hat)


def run_classifier(
    clf_name,
    clf,
    X,
    y,
    cv_splitter,
    weights=None,
    spatial_features_obj=None,
    ignore_test_inds=False,
):
    """Simple experiment, run for each classifier in the list, which returns the
    result of a cross validation strategy, provided as an argument."""

    if type(X) not in [pd.DataFrame, gpd.GeoDataFrame]:
        log.warning("X isn't a dataframe? Your model might require one.")

    metrics_per_fold = list()

    for train_inds, test_inds in cv_splitter(X):
        if spatial_features_obj is None:
            X_copy = X.copy()

        # add spatial features as new columns to X.
        elif spatial_features_obj is not None and not ignore_test_inds:
            X_copy = spatial_features_obj.add_partition_stats_features(
                full_test_parcel_gdf=X, test_inds=test_inds
            )

        elif spatial_features_obj is not None and ignore_test_inds:
            X_copy = spatial_features_obj.add_partition_stats_features(
                full_test_parcel_gdf=X, test_inds=[]
            )

        X_train = X_copy.iloc[train_inds]
        X_test = X_copy.iloc[test_inds]
        y_train, y_test = y[train_inds], y[test_inds]

        debias_kwargs = {}

        if weights is not None and len(weights) > 0:
            train_weights = weights[train_inds]
            test_weights = weights[test_inds]

            debias_kwargs = {clf.steps[-1][0] + "__sample_weight": train_weights}

        clf.fit(X_train, y_train, **debias_kwargs)

        y_hat = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]

        evaluation_dict = {
            "cv_model_metrics": calculate_binary_classification_metrics(
                y_test, y_score, y_hat
            ),
            "curve_data": collect_curve_data(clf, y_test, y_score),
            "feature_data": {
                "feature_names": list(X.columns),
                "importance_score": collect_feature_data(clf),
            },
        }

        metrics_per_fold.append(evaluation_dict)

    metric_names = metrics_per_fold[0].keys()
    # Transform our list of dictionaries to a dictionary of lists.
    evaluation_results = {
        metric_name: [metrics_dict[metric_name] for metrics_dict in metrics_per_fold]
        for metric_name in metric_names
    }

    return evaluation_results


# TODO rename to reflect CV and metrics returned?
# TODO pass scoring function?
def run_classifiers(clfs, X, y, cv):
    """Simple experiment, run for each classifier in the list, which returns the
    result of a cross validation strategy, provided as an argument."""
    reports = dict()

    for name, clf in clfs:
        reports[name] = run_classifier(name, clf, X, y, cv)

    return reports


def generate_learning_curve(clf, X, y, cv=StratifiedKFold(10)):
    return dict(
        zip(
            ["train_sizes", "train_scores", "test_scores"],
            learning_curve(clf, X, y, cv=cv, scoring="roc_auc"),
        )
    )


# TODO Collect more robust metrics ala run_classifiers /w
# `calculate_binary_classification_metrics` and `get_curve_data`.
def generate_learning_curves(clfs, X, y):
    return {name: generate_learning_curve(clf, X, y) for name, clf in clfs}

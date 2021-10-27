#!/usr/bin/env python3
import numpy as np
from scipy.stats import laplace
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, GroupShuffleSplit
from gizmo.ml import preprocessor_from_config
from gizmo.utils import get_in
from gizmo.debias import debias
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from service_line_pipeline import RANDOM_STATE
from service_line_pipeline.utils import load_client_configs
from typing import Dict, Any
from functools import partial
from . import log, RANDOM_STATE
from gizmo.evaluation import run_classifiers, generate_learning_curves
from gizmo.ml import (
    estimator_from_config,
    preprocessor_from_config,
    pipeline_from_config,
)
from pathlib import Path


def build_models(features_config, estimators_config, models_config):
    feature_pipelines = {
        feature_set_name: preprocessor_from_config(feature_set_config_list)
        for feature_set_name, feature_set_config_list in features_config.items()
    }

    estimators = {
        estimator_name: estimator_from_config(estimator_config)
        for estimator_name, estimator_config in estimators_config.items()
    }

    models = {
        model_name: Pipeline(
            steps=[
                (
                    "preprocessor",
                    FeatureUnion(
                        transformer_list=[
                            (
                                feature_pipeline_name,
                                feature_pipelines[feature_pipeline_name],
                            )
                            for feature_pipeline_name in config["features"]
                        ]
                    ),
                ),
                ("estimator", estimators[config["estimator"]]),
            ]
        )
        for model_name, config in models_config.items()
    }

    return models


def build_model(client_name: str, model_config: Dict[str, Any]):
    """Take any supported client name and call the model constructing function,
    returning an unfit model for that client."""
    client_model = {
        "flint": build_flint_model,
        "toledo": build_toledo_model,
        "halifax": build_halifax_model,
        "benton_harbor": build_benton_harbor_model,
        "detroit": build_detroit_model,
        "trenton": build_trenton_model,
        # "helios": build_helios_model,
        # ...
    }[client_name](model_config)
    return client_model


def get_models(client_name, client_config_path='configs/clients.yaml'):
    """
    Return a dictionary: keys are model names; values are instantiated models.
    """
    client_configs = load_client_configs(Path(client_config_path))
    client_config = client_configs[client_name]
    features_config = client_config.get("features") or {}
    estimators_config = client_config.get("estimators") or {}
    models_config = client_config.get("models") or {}

    models = build_models(features_config, estimators_config, models_config)

    return models


def build_detroit_model(model_config):
    """Build and return a fresh pipeline, configured for the Detroit data"""
    return pipeline_from_config(model_config)


def build_flint_model(model_config):
    """Build and return a fresh pipeline, configured for the Flint data"""
    return pipeline_from_config(model_config)


# TODO add RFE code
def build_toledo_model(model_config):
    return pipeline_from_config(model_config)


def build_halifax_model(model_config):
    return pipeline_from_config(model_config)


def build_benton_harbor_model(model_config):
    return pipeline_from_config(model_config)


def build_trenton_model(model_config):
    return pipeline_from_config(model_config)


# TODO This is almost certainly in the wrong place; move to evaluation.py?
def construct_cv_from_config(constructor=None, params={}, groups=None):
    """Takes configration options specified in config, validates them, and
    returns a function, partially applying special arugments depending on the
    split strategy. Returned cv_splitter is used like so:
    `train_inds, test_inds in cv_splitter(X)`"""
    split_strategies = [
        "KFold",
        "GroupShuffleSplit",
        "SKCV",
        "spatial-basic",
        "spatial-neighborhood",
    ]
    assert constructor in split_strategies

    # Initialize splitters
    cv_splitter = None
    if constructor == "KFold":
        cv = KFold(**params)
    elif constructor == "GroupShuffleSplit":
        params["random_state"] = RANDOM_STATE
        cv = GroupShuffleSplit(**params)

        # Special argument to handle difference in cv.split API
        cv_splitter = partial(cv.split, groups=groups)
    elif constructor == "spatial-basic":
        log.info("hello from spatial-basic!")
        cv = KFold(**params)
    elif constructor == "spatial-neighborhood":
        cv = KFold(**params)
    else:
        raise ValueError(
            f"Unsupported cv Strategy {constructor}."
            f" try {', '.join(split_strategies)}"
        )

    # If not set above, set by default here to normal split api
    cv_splitter = cv_splitter or cv.split
    return cv_splitter


def make_weights_laplace(
    vec=None, decision_boundary=None, scale=0.1, lower=0.1, upper=0.9
):
    # Clip cuts off values below lower and above upper, so that weights stay between 0 and 1.
    return np.clip(laplace.pdf(vec, decision_boundary, scale), lower, upper)


def get_iwal_weights(probs, lower=0.1, upper=0.9, decision_boundary=0.7, scale=0.1):
    """Returns a vector of Laplace weights associated with a probability vector."""
    if probs is None:
        # Maybe we should throw an error here?
        return None
    # We make the weights Laplace.
    wts = make_weights_laplace(probs, decision_boundary, scale, lower, upper)
    return wts


def calculate_debias_weights(data, debias_config, label_col):

    X = data

    debiasing_preprocessor = preprocessor_from_config(
        get_in(debias_config, ["preprocessor"])
    )

    X_transform = debiasing_preprocessor.fit_transform(X)

    sample_weights = debias(
        X_transform,
        label_col,
        figure=False,
        metrics=False,
        est=get_in(debias_config, ["method"]),
        params=None,
    )

    #TODO: Send params through the config
    return sample_weights

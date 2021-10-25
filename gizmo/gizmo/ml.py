#!/usr/bin/env python3
"""
Module for basic machine learning pipeline construction at BlueConduit.


TODO
- Custom Feature Transformers
- Custom Models (ex: debiasing model)
"""

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
    PowerTransformer,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
from gizmo import log
from stringcase import snakecase


class YrBuiltRangeTform(BaseEstimator, TransformerMixin):
    """Custom scaling transformer for construction year"""

    def __init__(self, earliest=1600, latest=2021):
        self.earliest = earliest
        self.latest = latest

    def get_feature_names(self):
        return X.columns.tolist()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[X < self.earliest] = np.nan
        X[X > self.latest] = np.nan
        return X


class ZeroTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer setting 0 to nan"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.replace(0, np.nan)


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer returning sqrt of data"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.sqrt(X)


def pipeline_from_config(config):
    # TODO validate basic config

    preprocessor_config = config.get("preprocessor", {})
    estimator_config = config.get("estimator", {})

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_from_config(preprocessor_config)),
            ("estimator", estimator_from_config(estimator_config)),
        ]
    )
    return model


def transformer_list_from_config(tfmr_list):
    steps = [
        (
            snakecase(config["constructor"]),
            eval(config["constructor"])(**config.get("params") or {}),
        )
        for config in tfmr_list
    ]
    return Pipeline(steps=steps)


def preprocessor_from_config(tfmr_config_list):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                tfmr_config.get("name"),
                transformer_list_from_config(tfmr_config.get("transforms")),
                tfmr_config.get("target_columns"),
            )
            for tfmr_config in tfmr_config_list
        ]
    )
    return preprocessor


def estimator_from_config(config):
    estimator = eval(config["constructor"])(**config.get("params") or {})
    return estimator


class RandomGuessClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):

        # TODO Fails on dataframes (nan read as float for string col)
        # X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        return self

    # TODO call predict_proba and round
    def predict(self, X):
        return np.random.choice(self.classes_, size=X.shape[0])

    # TODO sample uniform dist. instead of "max uncertainty always"
    def predict_proba(self, X):
        halfs = np.zeros(shape=X.shape[0]) + 0.5
        return np.array([halfs, halfs]).T

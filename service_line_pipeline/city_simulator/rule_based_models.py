import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedModel(BaseEstimator, ClassifierMixin):
    """
    Rule-based model, otherwise empty, mimics an sklearn model
    or pipeline object for the purposes of the simulator: it supports a
    .fit(X, y) method and a .predict_proba(X) method.
    """

    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba needs to be replaced!")


class YearBuiltBefore(RuleBasedModel):
    def __init__(
        self,
        year_built_col_name="year_built",
        year_built_cutoff=1950,
    ):
        self.year_built_col_name = year_built_col_name
        self.year_built_cutoff = year_built_cutoff

    def predict_proba(self, X):
        greater_than_year = (
            X[self.year_built_col_name].ge(self.year_built_cutoff).astype(int)
        )
        less_than_year = (~greater_than_year.astype(bool)).astype(int)

        # return pd.DataFrame({"a1": greater_than_year, "a2": less_than_year}).to_numpy()
        return np.array([greater_than_year, less_than_year]).T

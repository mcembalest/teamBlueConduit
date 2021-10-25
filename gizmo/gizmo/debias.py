#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns

from functools import partial, reduce
from multiprocessing import cpu_count
from pathlib import Path

from scipy.spatial.distance import cdist, pdist
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from KDEpy import TreeKDE

from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LogisticRegressionCV


import matplotlib.pyplot as plt

import pandas as pd

from gizmo.bc_logger import get_simple_logger

from densratio import densratio


log = get_simple_logger(__name__)


class DensityRatioEstimator:
    """
    Class to accomplish direct density estimation implementing the original KLIEP
    algorithm from Direct Importance Estimation with Model Selection
    and Its Application to Covariate Shift Adaptation by Sugiyama et al.

    The training set is distributed via
                                            train ~ p(x)
    and the test set is distributed via
                                            test ~ q(x).

    The KLIEP algorithm and its variants approximate w(x) = q(x) / p(x) directly. The predict function returns the
    estimate of w(x). The function w(x) can serve as sample weights for the training set during
    training to modify the expectation function that the model's loss function is optimized via,
    i.e.

            E_{x ~ w(x)p(x)} loss(x) = E_{x ~ q(x)} loss(x).

    Usage :
        The fit method is used to run the KLIEP algorithm using LCV and returns value of J
        trained on the entire training/test set with the best sigma found.

        Use the predict method on the training set to determine the sample weights from the KLIEP algorithm.
    """

    def __init__(
        self,
        max_iter=5000,
        num_params=[0.1, 0.2],
        epsilon=1e-4,
        cv=3,
        sigmas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        random_state=None,
        verbose=0,
    ):
        """
        Direct density estimation using an inner LCV loop to estimate the proper model. Can be used with sklearn
        cross validation methods with or without storing the inner CV. To use a standard grid search.


        max_iter : Number of iterations to perform
        num_params : List of number of test set vectors used to construct the approximation for inner LCV.
                     Must be a float. Original paper used 10%, i.e. =.1
        sigmas : List of sigmas to be used in inner LCV loop.
        epsilon : Additive factor in the iterative algorithm for numerical stability.
        """
        self.max_iter = max_iter
        self.num_params = num_params
        self.epsilon = epsilon
        self.verbose = verbose
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = 0

    def fit(self, X_train, X_test, alpha_0=None):
        """Uses cross validation to select sigma as in the original paper (LCV).
        In a break from sklearn convention, y=X_test.
        The parameter cv corresponds to R in the original paper.
        Once found, the best sigma is used to train on the full set."""

        # LCV loop, shuffle a copy in place for performance.
        cv = self.cv
        chunk = int(X_test.shape[0] / float(cv))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled)

        j_scores = {}

        if type(self.sigmas) != list:
            self.sigmas = [self.sigmas]

        if type(self.num_params) != list:
            self.num_params = [self.num_params]

        if len(self.sigmas) * len(self.num_params) > 1:
            # Inner LCV loop
            for num_param in self.num_params:
                for sigma in self.sigmas:
                    j_scores[(num_param, sigma)] = np.zeros(cv)
                    for k in range(1, cv + 1):
                        if self.verbose > 0:
                            print("Training: sigma: %s    R: %s" % (sigma, k))
                        X_test_fold = X_test_shuffled[(k - 1) * chunk : k * chunk, :]
                        j_scores[(num_param, sigma)][k - 1] = self._fit(
                            X_train=X_train,
                            X_test=X_test_fold,
                            num_parameters=num_param,
                            sigma=sigma,
                        )
                    j_scores[(num_param, sigma)] = np.mean(j_scores[(num_param, sigma)])

            sorted_scores = sorted(
                [x for x in j_scores.items() if np.isfinite(x[1])],
                key=lambda x: x[1],
                reverse=True,
            )
            if len(sorted_scores) == 0:
                warnings.warn("LCV failed to converge for all values of sigma.")
                return self
            self._sigma = sorted_scores[0][0][1]
            self._num_parameters = sorted_scores[0][0][0]
            self._j_scores = sorted_scores
        else:
            self._sigma = self.sigmas[0]
            self._num_parameters = self.num_params[0]
            # best sigma
        self._j = self._fit(
            X_train=X_train,
            X_test=X_test_shuffled,
            num_parameters=self._num_parameters,
            sigma=self._sigma,
        )

        return self  # Compatibility with sklearn

    def _fit(self, X_train, X_test, num_parameters, sigma, alpha_0=None):
        """Fits the estimator with the given parameters w-hat and returns J"""

        num_parameters = num_parameters

        if type(num_parameters) == float:
            num_parameters = int(X_test.shape[0] * num_parameters)

        self._select_param_vectors(
            X_test=X_test, sigma=sigma, num_parameters=num_parameters
        )

        X_train = self._reshape_X(X_train)
        X_test = self._reshape_X(X_test)

        if alpha_0 is None:
            alpha_0 = np.ones(shape=(num_parameters, 1)) / float(num_parameters)

        self._find_alpha(
            X_train=X_train,
            X_test=X_test,
            num_parameters=num_parameters,
            epsilon=self.epsilon,
            alpha_0=alpha_0,
            sigma=sigma,
        )

        return self._calculate_j(X_test, sigma=sigma)

    def _calculate_j(self, X_test, sigma):
        return np.log(self.predict(X_test, sigma=sigma)).sum() / X_test.shape[0]

    def score(self, X_test):
        """Return the J score, similar to sklearn's API"""
        return self._calculate_j(X_test=X_test, sigma=self._sigma)

    @staticmethod
    def _reshape_X(X):
        """Reshape input from mxn to mx1xn to take advantage of numpy broadcasting."""
        if len(X.shape) != 3:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def _select_param_vectors(self, X_test, sigma, num_parameters):
        """X_test is the test set. b is the number of parameters."""
        indices = np.random.choice(X_test.shape[0], size=num_parameters, replace=False)
        self._test_vectors = X_test[indices, :].copy()
        self._phi_fitted = True

    def _phi(self, X, sigma=None):

        if sigma is None:
            sigma = self._sigma

        if self._phi_fitted:
            return np.exp(
                -np.sum((X - self._test_vectors) ** 2, axis=-1) / (2 * sigma ** 2)
            )
        raise Exception("Phi not fitted.")

    def _find_alpha(self, alpha_0, X_train, X_test, num_parameters, sigma, epsilon):
        A = np.zeros(shape=(X_test.shape[0], num_parameters))
        b = np.zeros(shape=(num_parameters, 1))

        A = self._phi(X_test, sigma)
        b = self._phi(X_train, sigma).sum(axis=0) / X_train.shape[0]
        b = b.reshape((num_parameters, 1))

        out = alpha_0.copy()
        for k in range(self.max_iter):
            out += epsilon * np.dot(np.transpose(A), 1.0 / np.dot(A, out))
            out += b * (
                ((1 - np.dot(np.transpose(b), out)) / np.dot(np.transpose(b), b))
            )
            out = np.maximum(0, out)
            out /= np.dot(np.transpose(b), out)

        self._alpha = out
        self._fitted = True

    def predict(self, X, sigma=None):
        """Equivalent of w(X) from the original paper."""

        X = self._reshape_X(X)
        if not self._fitted:
            raise Exception("Not fitted!")
        return np.dot(self._phi(X, sigma=sigma), self._alpha).reshape((X.shape[0],))


def cramer_metric(x, y, n_jobs=None):
    """
    TODO: Docs
    From Baringhaus and Franz:
    The test statistic is the difference of the sum of all the Euclidean
    interpoint distances between the random variables from the two different
    samples and one-half of the two corresponding sums of distances of the
    variables within the same sample
    """
    # (sum of all Euclidean interpoint distances between random
    #  variables from 2 different samples)
    #  - (1/2)(sums of distances of the variable within sample 1)
    #  - (1/2)(sums of distances of the variable within sample 2)

    m = x.shape[0]
    n = y.shape[0]

    # Parallel by default, but don't use all CPUs.
    n_jobs = n_jobs or max(int(cpu_count() * 7 / 8), 1)

    distance_matrix_generator = partial(
        pairwise_distances_chunked,
        n_jobs=n_jobs,
        metric="euclidean",
    )

    sum_chunks = lambda dist, dmc: dist + np.sum(dmc)
    xyterm = (1 / (m * n)) * reduce(sum_chunks, distance_matrix_generator(x, y), 0)
    xterm = (1 / (2 * m * m)) * reduce(sum_chunks, distance_matrix_generator(x, x), 0)
    yterm = (1 / (2 * n * n)) * reduce(sum_chunks, distance_matrix_generator(y, y), 0)

    cramer = (m * n / (m + n)) * (xyterm - xterm - yterm)

    log.info(
        "Cramer statistic",
        value=cramer,
    )

    return cramer


class DensityRatioEstimator:
    """
    Class to accomplish direct density estimation implementing the original KLIEP
    algorithm from Direct Importance Estimation with Model Selection
    and Its Application to Covariate Shift Adaptation by Sugiyama et al.

    The training set is distributed via
                                            train ~ p(x)
    and the test set is distributed via
                                            test ~ q(x).

    The KLIEP algorithm and its variants approximate w(x) = q(x) / p(x) directly. The predict function returns the
    estimate of w(x). The function w(x) can serve as sample weights for the training set during
    training to modify the expectation function that the model's loss function is optimized via,
    i.e.

            E_{x ~ w(x)p(x)} loss(x) = E_{x ~ q(x)} loss(x).

    Usage :
        The fit method is used to run the KLIEP algorithm using LCV and returns value of J
        trained on the entire training/test set with the best sigma found.

        Use the predict method on the training set to determine the sample weights from the KLIEP algorithm.
    """

    def __init__(
        self,
        max_iter=5000,
        num_params=[0.1, 0.2],
        epsilon=1e-4,
        cv=3,
        sigmas=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
        random_state=None,
        verbose=0,
    ):
        """
        Direct density estimation using an inner LCV loop to estimate the proper model. Can be used with sklearn
        cross validation methods with or without storing the inner CV. To use a standard grid search.


        max_iter : Number of iterations to perform
        num_params : List of number of test set vectors used to construct the approximation for inner LCV.
                     Must be a float. Original paper used 10%, i.e. =.1
        sigmas : List of sigmas to be used in inner LCV loop.
        epsilon : Additive factor in the iterative algorithm for numerical stability.
        """
        self.max_iter = max_iter
        self.num_params = num_params
        self.epsilon = epsilon
        self.verbose = verbose
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = 0

    def fit(self, X_train, X_test, alpha_0=None):
        """Uses cross validation to select sigma as in the original paper (LCV).
        In a break from sklearn convention, y=X_test.
        The parameter cv corresponds to R in the original paper.
        Once found, the best sigma is used to train on the full set."""

        # LCV loop, shuffle a copy in place for performance.
        cv = self.cv
        chunk = int(X_test.shape[0] / float(cv))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled)

        j_scores = {}

        if type(self.sigmas) != list:
            self.sigmas = [self.sigmas]

        if type(self.num_params) != list:
            self.num_params = [self.num_params]

        if len(self.sigmas) * len(self.num_params) > 1:
            # Inner LCV loop
            for num_param in self.num_params:
                for sigma in self.sigmas:
                    j_scores[(num_param, sigma)] = np.zeros(cv)
                    for k in range(1, cv + 1):
                        if self.verbose > 0:
                            print("Training: sigma: %s    R: %s" % (sigma, k))
                        X_test_fold = X_test_shuffled[(k - 1) * chunk : k * chunk, :]
                        j_scores[(num_param, sigma)][k - 1] = self._fit(
                            X_train=X_train,
                            X_test=X_test_fold,
                            num_parameters=num_param,
                            sigma=sigma,
                        )
                    j_scores[(num_param, sigma)] = np.mean(j_scores[(num_param, sigma)])

            sorted_scores = sorted(
                [x for x in j_scores.iteritems() if np.isfinite(x[1])],
                key=lambda x: x[1],
                reverse=True,
            )
            if len(sorted_scores) == 0:
                warnings.warn("LCV failed to converge for all values of sigma.")
                return self
            self._sigma = sorted_scores[0][0][1]
            self._num_parameters = sorted_scores[0][0][0]
            self._j_scores = sorted_scores
        else:
            self._sigma = self.sigmas[0]
            self._num_parameters = self.num_params[0]
            # best sigma
        self._j = self._fit(
            X_train=X_train,
            X_test=X_test_shuffled,
            num_parameters=self._num_parameters,
            sigma=self._sigma,
        )

        return self  # Compatibility with sklearn

    def _fit(self, X_train, X_test, num_parameters, sigma, alpha_0=None):
        """Fits the estimator with the given parameters w-hat and returns J"""

        num_parameters = num_parameters

        if type(num_parameters) == float:
            num_parameters = int(X_test.shape[0] * num_parameters)

        self._select_param_vectors(
            X_test=X_test, sigma=sigma, num_parameters=num_parameters
        )

        X_train = self._reshape_X(X_train)
        X_test = self._reshape_X(X_test)

        if alpha_0 is None:
            alpha_0 = np.ones(shape=(num_parameters, 1)) / float(num_parameters)

        self._find_alpha(
            X_train=X_train,
            X_test=X_test,
            num_parameters=num_parameters,
            epsilon=self.epsilon,
            alpha_0=alpha_0,
            sigma=sigma,
        )

        return self._calculate_j(X_test, sigma=sigma)

    def _calculate_j(self, X_test, sigma):
        return np.log(self.predict(X_test, sigma=sigma)).sum() / X_test.shape[0]

    def score(self, X_test):
        """Return the J score, similar to sklearn's API"""
        return self._calculate_j(X_test=X_test, sigma=self._sigma)

    @staticmethod
    def _reshape_X(X):
        """Reshape input from mxn to mx1xn to take advantage of numpy broadcasting."""
        if len(X.shape) != 3:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def _select_param_vectors(self, X_test, sigma, num_parameters):
        """X_test is the test set. b is the number of parameters."""
        indices = np.random.choice(X_test.shape[0], size=num_parameters, replace=False)
        self._test_vectors = X_test[indices, :].copy()
        self._phi_fitted = True

    def _phi(self, X, sigma=None):

        if sigma is None:
            sigma = self._sigma

        if self._phi_fitted:
            return np.exp(
                -np.sum((X - self._test_vectors) ** 2, axis=-1) / (2 * sigma ** 2)
            )
        raise Exception("Phi not fitted.")

    def _find_alpha(self, alpha_0, X_train, X_test, num_parameters, sigma, epsilon):
        A = np.zeros(shape=(X_test.shape[0], num_parameters))
        b = np.zeros(shape=(num_parameters, 1))

        A = self._phi(X_test, sigma)
        b = self._phi(X_train, sigma).sum(axis=0) / X_train.shape[0]
        b = b.reshape((num_parameters, 1))

        out = alpha_0.copy()
        for k in range(self.max_iter):
            out += epsilon * np.dot(np.transpose(A), 1.0 / np.dot(A, out))
            out += b * (
                ((1 - np.dot(np.transpose(b), out)) / np.dot(np.transpose(b), b))
            )
            out = np.maximum(0, out)
            out /= np.dot(np.transpose(b), out)

        self._alpha = out
        self._fitted = True

    def predict(self, X, sigma=None):
        """Equivalent of w(X) from the original paper."""

        X = self._reshape_X(X)
        if not self._fitted:
            raise Exception("Not fitted!")
        return np.dot(self._phi(X, sigma=sigma), self._alpha).reshape((X.shape[0],))


class PCDensityRatioEstimator:
    def __init__(self, classifier=LogisticRegressionCV(max_iter=1000)):

        self.classifier = classifier

    def fit(self, X, y):

        self.classifier.fit(X, y)

        return self

    def predict(self, X):

        # p(x)/(alpha * p(x) + (1 - alpha) * q(x))

        probs = self.classifier.predict_proba(X)

        alpha = 0.9

        return probs[:, 0] / (alpha * probs[:, 0] + (1 - alpha) * probs[:, 1])


class uLSIFDensityRatioEstimator:

    def __init__(self, alpha = 0.5):

        self.alpha = alpha

    def fit(self, X, y):

        self.densratio_obj = densratio(X[y], X[~y], alpha=self.alpha, sigma_range = [1.000], lambda_range = [0.100])
        print(self.densratio_obj)
        return self

    def predict(self, X):

        # p(x)/(alpha * p(x) + (1 - alpha) * q(x))
        return self.densratio_obj.compute_density_ratio(X)


class KLIEPDensityRatioEstimator:

    def __init__(self):

        self.kliep = DensityRatioEstimator(max_iter=10)

    def fit(self, X, y):

        self.kliep.fit(X[y], X[~y]) # keyword arguments are X_train and X_test

    def predict(self, X):

        return self.kliep.predict(X)


def plot_debias(data_all, data_labeled, weights=[], figures_directory="./figures/"):

    data_all, data_labeled = pd.DataFrame(data_all), pd.DataFrame(data_labeled)

    if len(weights) != 0:
        # Resample according to the weights
        resample_pop = data_labeled.sample(
            data_all.shape[0], weights=weights, replace=True
        )

        fig, axes = plt.subplots(len(data_all.columns), 2)

        fig.set_figheight(10)
        fig.set_figwidth(15)

        for index, var in enumerate(data_all.columns):

            axes[index][0].set_title(str(var))

            h1 = axes[index][0].hist(
                data_all[var], bins=32, density=True, alpha=0.5, label="Population"
            )

            h2 = axes[index][0].hist(
                data_labeled[var], bins=32, density=True, alpha=0.5, label="Labeled"
            )

            axes[index][0].legend(loc="upper left")

            h3 = axes[index][1].hist(
                data_all[var], bins=32, density=True, alpha=0.5, label="Population"
            )

            h4 = axes[index][1].hist(
                resample_pop[var],
                bins=32,
                density=True,
                alpha=0.5,
                label="Labeled (debiased)",
            )

            axes[index][1].legend(loc="upper left")

        figures_directory = Path(figures_directory).resolve()
        figures_directory.mkdir(parents=True, exist_ok=True)

        fig.savefig(figures_directory / Path("debias_histograms.png"))

        # TODO: The datashaders (joint distributions)


def hist_ndim(x, x_grid, bins=10, **kwargs):

    hist, edges = np.histogramdd(x, bins=bins)

    lookup = lambda hist, edges, coords: hist[
        tuple(
            [
                np.clip(np.searchsorted(edges[i], coords[i]) - 1, 0, len(edges[i]) - 2)
                for i in range(len(hist.shape))
            ]
        )
    ]

    density_eval = lookup(hist, edges, tuple(xi for xi in x_grid.T))

    return density_eval, edges


def debias(data, labels, method='PCRE', figure=False, metrics=False, **kwargs):

    # kliep = DensityRatioEstimator(max_iter=10)
    # kliep.fit(data_labeled, data)
    # weights = kliep.predict(data_labeled)

    # for densratio
    # TODO: Get `method` here from the config yaml and branch on it
    #    We now need to support logreg, KLIEP, RuLSIF etc.

    #import pdb; pdb.set_trace()

    methodswitch = {'PCRE' : PCDensityRatioEstimator,
                    'KLIEP' : DensityRatioEstimator,
                    'RuLSIF' : uLSIFDensityRatioEstimator}

    est = methodswitch.get(method, PCDensityRatioEstimator)()

    x_num = data[~labels]
    x_denom = data[labels]

    est = est.fit(data, labels)
    weights = est.predict(x_denom)

    return weights


if __name__ == "__main__":
    pass

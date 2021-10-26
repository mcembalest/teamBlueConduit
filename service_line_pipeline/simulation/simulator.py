"""The Simulator class takes a client city and a model to emulate the exploration
(inspection) and exploitation (replacement) mechanics. The subclasses in this
file implement the different inspection and replacement steps according to the
associated policy."""

import pandas as pd
import numpy as np

from scipy.stats import laplace

from ..models import calculate_debias_weights


class Simulator:
    def __init__(
        self,
        name=None,
        city=None,
        model=None,
        max_samples=8000,
        train_stride=1000,
        eps=1.0,
        decision_boundary=None,
        max_explore_steps=None,
        client_config=None,
        verbose=True,
    ):

        self.name = name

        self.city = city
        self.model = model
        self.eps = eps

        self.step_count = 0
        self.curr_yhat = None
        self.max_samples = max_samples

        self.train_stride = train_stride
        self.decision_boundary = decision_boundary
        self.max_explore_steps = max_explore_steps

        self.sim_state = self.city.data[
            [self.city.unique_id, self.city.target_column]
        ].assign(explored=False, excavated=False)

        self.total_cost = 0.0

        self.verbose = verbose

        self.client_config = client_config

    # TODO Make total_cost a read only attribute that just calculates the cost.
    #      Want to double check logic that excavating doesn't double count
    #      things because their explored.
    #
    # @property
    # def total_cost(self):
    #     explore_costs = self.sim_state.explored.fillna(False).sum() * self.city.explore_cost
    #     excavate_costs = self.sim_state.excavated.fillna(False).sum() * self.city.excavate_cost
    #     return explore_costs + excavate_costs

    def explore_lines(self, explore_ids):

        self.total_cost += len(explore_ids) * self.city.explore_cost
        self.sim_state.loc[explore_ids, "explored"] = True

        return self.city.data.loc[explore_ids]

    def replace_lines(self, excavate_list):
        num_replaced = self.sim_state.loc[excavate_list, self.city.target_column].sum()

        num_excavated_not_replaced = len(excavate_list) - num_replaced

        self.total_cost += (
            num_replaced * self.city.replace_cost
            + num_excavated_not_replaced * self.city.excavate_cost
        )

        self.sim_state.loc[excavate_list, "explored"] = True
        self.sim_state.loc[excavate_list, "excavated"] = True

        replace_index = self.sim_state.loc[excavate_list][
            self.sim_state.loc[excavate_list, self.city.target_column].astype(bool)
        ].index

        self.sim_state.loc[replace_index, "replaced"] = True

        return self.sim_state.loc[excavate_list]

    def explore_step(self):
        raise NotImplementedError("explore_step must be implemented")

    def excavate_step(self):
        raise NotImplementedError("excavate_step must be implemented")

    # TODO Read the model config from clients
    def train_step(self, wts=None):
        Xdata = self.city.data.loc[self.sim_state.explored].drop(
            [self.city.target_column, self.city.unique_id], axis=1
        )
        Ydata = self.sim_state.loc[
            self.sim_state.explored, self.city.target_column
        ].astype(int)

        # HACK Pipeline.fit does not take a sample weights param, but the
        # estimator does. Must build kwarg dynamically to get the full parameter
        # name correct.
        if wts is not None:
            sample_weight_kwargs = {
                self.model.steps[-1][0]
                + "__sample_weight": wts[self.sim_state.explored]
            }
        else:
            sample_weight_kwargs = {self.model.steps[-1][0] + "__sample_weight": wts}

        self.model.fit(Xdata, Ydata, **sample_weight_kwargs)

        yhat = self.model.predict_proba(
            self.city.data.drop([self.city.target_column, self.city.unique_id], axis=1)
        )[:, 1]

        return pd.Series(yhat, index=self.city.data.index)

    def simulate(self):

        self.total_cost = 0
        self.step_count = 0
        self.trained = False

        self.sim_state = self.city.data[
            [self.city.unique_id, self.city.target_column]
        ].assign(explored=False, excavated=False)

        hv_hit_rates = []
        sl_hit_rates = []
        cum_avg_cost = []
        replacements = []

        # Set the sim_state
        self.sim_state = self.city.data[
            [self.city.unique_id, self.city.target_column]
        ].assign(explored=False, excavated=False, replaced=False)

        if self.verbose:
            print("\nStarting %s" % self.__str__())

        while self.sim_state.excavated.sum() < self.max_samples:
            explored_data = self.explore_step()
            if explored_data is not None:

                self.sim_state.loc[explored_data.index, "sim_round"] = self.step_count
                hv_hit_rates.append(
                    explored_data[self.city.target_column].sum()
                    / float(len(explored_data))
                )
            else:
                hv_hit_rates.append(np.nan)

            if (self.step_count % self.train_stride) == 0:
                if self.get_iwal_weights(self.curr_yhat) is None:
                    if (self.step_count != 0) and (explored_data is not None):
                        self.curr_yhat = self.train_step()
                        self.trained = True
                else:
                    self.curr_yhat = self.train_step(
                        wts=self.get_iwal_weights(
                            self.curr_yhat, decision_boundary=self.decision_boundary
                        )
                    )
                    self.trained = True

            excavated_data = self.excavate_step()

            if excavated_data is not None:

                self.sim_state.loc[excavated_data.index, "sim_round"] = self.step_count
                sl_hit_rates.append(
                    excavated_data[self.city.target_column].sum()
                    / float(len(excavated_data))
                )
                replacements.append(excavated_data[self.city.target_column].sum())

                cum_avg_cost.append(
                    float(self.total_cost) / self.sim_state.replaced.sum()
                )

            self.step_count += 1

        if self.verbose:
            self.print_stats()

        return (
            self.sim_state,
            (sl_hit_rates, hv_hit_rates, cum_avg_cost, replacements),
            (
                self.sim_state.loc[
                    self.sim_state.excavated == True, self.city.target_column
                ].mean(),
                self.sim_state.loc[
                    self.sim_state.excavated == True, self.city.target_column
                ].mean(),
                self.sim_state.loc[
                    self.sim_state.explored == True, self.city.target_column
                ].mean(),
                (
                    float(self.total_cost)
                    / self.sim_state[self.city.target_column].sum()
                ),
            ),
        )

    def print_stats(self):
        print("\n")
        print(
            "Excavation Hit-rate: \t\t %0.2f"
            % (
                self.sim_state.loc[
                    self.sim_state.excavated.fillna(False), self.city.target_column
                ].mean()
            )
        )
        print(
            "Inspection Hit-rate: \t\t %0.2f"
            % (
                self.sim_state.loc[
                    self.sim_state.explored.fillna(False), self.city.target_column
                ].mean()
            )
        )
        print("Total Number of Excavations: \t%d" % self.sim_state.excavated.sum())
        print("Total Number of Explorations: \t%d" % self.sim_state.explored.sum())
        # NOTE At times, replaced is explicitly referenced. Others, it is
        # implicitly derived from excavated and explored columns
        print("Total Number of Replacements: \t%d" % self.sim_state.replaced.sum())
        print(
            "Avg cost of Replacement: \t%0.2f"
            % (float(self.total_cost) / self.sim_state.replaced.sum())
        )

    def get_iwal_weights(self, probs):
        return None

    def __str__(self):
        return self.name


# TODO Compliment the sim subclassing. Then consider making a "Policy" class
# that implements a `sample` method.
class UniformRandomExplore(Simulator):
    def explore_step(self, num_samples=100):

        # TODO Simplify: the only thing changing is the number of samples
        if sum(self.sim_state.excavated == False) < num_samples:
            explore_ids = (
                self.sim_state[self.sim_state.explored == False]
                .sample(sum(self.sim_state.explored == False))
                .index
            )
        else:
            explore_ids = (
                self.sim_state[self.sim_state.explored == False]
                .sample(num_samples)
                .index
            )

        explored_data = self.explore_lines(explored_ids)

        return explored_data

    def excavate_step(self):

        if (
            self.sim_state["excavated"].sum()
            + len(
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state[self.city.target_column] == True)
                ]
            )
            > self.max_samples
        ):
            excavate_ids = (
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state[self.city.target_column] == True)
                ]
                .sample(self.max_samples - self.sim_state["excavated"].sum())
                .index
            )
        else:
            excavate_ids = self.sim_state[
                (self.sim_state.excavated == False)
                & (self.sim_state[self.city.target_column] == True)
            ].index

        excavated_data = self.replace_lines(excavate_ids)
        return excavated_data


class UniformRandomExcavate(Simulator):

    # NOTE We see the abstraction breaking down here. Want to avoid definitions
    # like this.
    def explore_step(self, num_samples=100):
        pass

    def excavate_step(self, num_samples=100):
        excavate_ids = (
            self.sim_state[self.sim_state.excavated == False].sample(num_samples).index
        )
        excavated_data = self.replace_lines(excavate_ids)

        return excavated_data


class GreedyExcavate(Simulator):
    def explore_step(self, num_samples=100):
        pass

    def excavate_step(self, num_samples=100):

        n_greedy = int((1 - self.eps) * num_samples)

        if self.trained == True:
            excavate_ids = (
                self.curr_yhat[
                    self.curr_yhat.index.isin(
                        self.sim_state[self.sim_state.excavated == False].index
                    )
                ]
                .sort_values(ascending=False)[:n_greedy]
                .index
            )
            excavated_data = self.replace_lines(excavate_ids)

            excavate_ids_ = (
                self.sim_state[(self.sim_state.explored == False)]
                .sample(num_samples - n_greedy)
                .index
            )
            excavated_data_ = self.replace_lines(excavate_ids)

            return excavated_data.append(excavated_data_)

        else:
            excavate_ids = (
                self.sim_state[(self.sim_state.explored == False)]
                .sample(num_samples)
                .index
            )
            excavated_data = self.replace_lines(excavate_ids)

            return excavated_data


class UniformExpGreedyAdaptive(Simulator):
    def explore_step(self, num_samples=100):

        # If we're about to explore all the lines ...
        if (self.sim_state.excavated == False).sum() < num_samples:
            # just return the rest of the unexplored ids
            explore_ids = self.sim_state[self.sim_state.explored == False].index
        else:
            explore_ids = (
                self.sim_state[self.sim_state.explored == False]
                .sample(num_samples)
                .index
            )

        explored_data = self.explore_lines(explore_ids)
        return explored_data

    def excavate_step(self, num_samples=100):
        if (
            self.sim_state["excavated"].sum()
            + len(
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
            )
            > self.max_samples
        ):
            excavate_ids = (
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
                .sample(
                    self.max_samples - self.sim_state["excavated"].sum().astype(int)
                )
                .index
            )
        else:
            excavate_ids = self.sim_state[
                (self.sim_state.excavated == False)
                & (self.sim_state.explored == True)
                & (self.sim_state[self.city.target_column] == True)
            ].index

        if self.trained == True:
            if len(excavate_ids) < num_samples:
                addtl_samples = num_samples - len(excavate_ids)
                addtl_ids = (
                    self.curr_yhat[
                        self.curr_yhat.index.isin(
                            self.sim_state[self.sim_state.excavated == False].index
                        )
                    ]
                    .sort_values(ascending=False)[:addtl_samples]
                    .index
                )

                excavate_ids = excavate_ids.append(addtl_ids)

        excavated_data = self.replace_lines(excavate_ids)

        return excavated_data


class UniformGreedyModel(Simulator):
    def explore_step(self, num_samples=100):

        if sum(self.sim_state.excavated == False) < num_samples:
            explore_ids = (
                self.sim_state[self.sim_state.explored == False]
                .sample(sum(self.sim_state.explored == False))
                .index
            )
        else:
            explore_ids = (
                self.sim_state[self.sim_state.explored == False]
                .sample(num_samples)
                .index
            )

        explored_data = self.explore_lines(explore_ids)

        self.sim_state.loc[explore_ids, "explored"] = True
        self.total_cost = self.sim_state.explored.sum() * self.city.explore_cost

        return self.sim_state.loc[self.sim_state.explored]

        return explored_data

    def excavate_step(self, num_samples=100):

        if (
            self.sim_state["excavated"].sum()
            + len(
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
            )
            > self.max_samples
        ):
            excavate_ids = (
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
                .sample(self.max_samples - self.sim_state["excavated"].sum())
                .index
            )
        else:
            excavate_ids = self.sim_state[
                (self.sim_state.excavated == False)
                & (self.sim_state.explored == True)
                & (self.sim_state[self.city.target_column] == True)
            ].index

        if self.trained == True:
            if len(excavate_ids) < num_samples:
                addtl_samples = num_samples - len(excavate_ids)
                addtl_ids = (
                    self.curr_yhat[
                        self.curr_yhat.index.isin(
                            self.sim_state[
                                self.sim_state[self.city.target_column].isna()
                            ].index
                        )
                    ]
                    .sort_values(ascending=False)[:addtl_samples]
                    .index
                )

                excavate_ids = excavate_ids.append(addtl_ids)

        excavated_data = self.replace_lines(excavate_ids)

        return excavated_data


class EpsUniformGreedyModel(Simulator):
    def explore_step(self, num_samples=100):

        if self.trained == False:
            if sum(self.sim_state.excavated == False) < num_samples:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(sum(self.sim_state.explored == False))
                    .index
                )
            else:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(num_samples)
                    .index
                )

            explored_data = self.explore_lines(explore_ids)
            return explored_data
        else:
            n_greedy = int((self.eps) * num_samples)
            if sum(self.sim_state.excavated == False) < num_samples:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(sum(self.sim_state.explored == False))
                    .index
                )
                explored_data = self.explore_lines(explore_ids)
            else:
                cand_probs = self.curr_yhat[
                    self.curr_yhat.index.isin(
                        self.sim_state[
                            self.sim_state[self.city.target_column].isna()
                        ].index
                    )
                ].copy()
                explore_ids = cand_probs.nlargest(n_greedy).index
                explored_data = self.explore_lines(explore_ids)

                explore_ids = (
                    self.sim_state[
                        (self.sim_state.explored == False)
                        & (~self.sim_state.index.isin(explore_ids))
                    ]
                    .sample(num_samples - n_greedy)
                    .index
                )
                explored_data_ = self.explore_lines(explore_ids)

            return explored_data.append(explored_data_)

    def excavate_step(self, num_samples=100):
        if (
            self.sim_state["excavated"].sum()
            + len(
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
            )
            > self.max_samples
        ):
            excavate_ids = (
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
                .sample(int(self.max_samples - self.sim_state.excavated.sum()))
                .index
            )
        else:
            excavate_ids = self.sim_state[
                (self.sim_state.excavated == False)
                & (self.sim_state.explored == True)
                & (self.sim_state[self.city.target_column] == True)
            ].index

        if self.trained == True:
            if len(excavate_ids) < num_samples:
                addtl_samples = num_samples - len(excavate_ids)
                addtl_ids = (
                    self.curr_yhat[
                        self.curr_yhat.index.isin(
                            self.sim_state[
                                self.sim_state[self.city.target_column].isna()
                            ].index
                        )
                    ]
                    .sort_values(ascending=False)[:addtl_samples]
                    .index
                )
                excavate_ids = excavate_ids.append(addtl_ids)

        excavated_data = self.replace_lines(excavate_ids)

        return excavated_data


class ALGreedyModel(Simulator):
    def explore_step(self, num_samples=100):

        if self.trained == False:
            if sum(self.sim_state.excavated == False) < num_samples:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(sum(self.sim_state.explored == False))
                    .index
                )
            else:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(num_samples)
                    .index
                )
            explored_data = self.explore_lines(explore_ids)

            return explored_data
        else:
            cand_probs = self.curr_yhat[
                self.curr_yhat.index.isin(
                    self.sim_state[self.sim_state.explored == False].index
                )
            ].copy()
            wts = self.get_iwal_weights(
                cand_probs, decision_boundary=self.decision_boundary
            )
            if sum(self.sim_state.excavated == False) < num_samples:
                explore_ids = cand_probs.sample(
                    sum(self.sim_state.explored == False), weights=wts
                ).index
            else:
                explore_ids = cand_probs.sample(num_samples, weights=wts).index
            explored_data = self.explore_lines(explore_ids)
            return explored_data

    def excavate_step(self, num_samples=100):

        if (
            self.sim_state["excavated"].sum()
            + len(
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
            )
            > self.max_samples
        ):
            excavate_ids = (
                self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ]
                .sample(self.max_samples - self.sim_state["excavated"].sum())
                .index
            )
        else:
            excavate_ids = self.sim_state[
                (self.sim_state.excavated == False)
                & (self.sim_state.explored == True)
                & (self.sim_state[self.city.target_column] == True)
            ].index

        if self.trained == True:
            if len(excavate_ids) < num_samples:
                addtl_samples = num_samples - len(excavate_ids)
                addtl_ids = (
                    self.curr_yhat[
                        self.curr_yhat.index.isin(
                            self.sim_state[self.sim_state.excavated == False].index
                        )
                    ]
                    .sort_values(ascending=False)[:addtl_samples]
                    .index
                )

                excavate_ids = excavate_ids.append(addtl_ids)

        excavated_data = self.replace_lines(excavate_ids)

        return excavated_data

    def make_weights_laplace(
        self, vec=None, decision_boundary=None, scale=0.1, lower=0.1, upper=0.9
    ):
        return np.clip(laplace.pdf(vec, decision_boundary, scale), lower, upper)

    def get_iwal_weights(
        self, probs, lower=0.1, upper=0.9, decision_boundary=0.7, scale=0.1
    ):

        if probs is None:
            return None

        wts = self.make_weights_laplace(probs, decision_boundary, scale, lower, upper)
        wts_inv = 1.0 / wts
        return wts


class UniformInspectAdaptiveExcavate(Simulator):
    def explore_step(self, num_samples=100):

        if self.step_count < self.max_explore_steps:
            unexplored = self.sim_state.explored == False
            try:
                explore_ids = self.sim_state[unexplored].sample(num_samples).index
            except ValueError:
                # ValueError: Cannot take a larger sample than population
                explore_ids = self.sim_state[unexplored].index
            return self.explore_lines(explore_ids)

        return None

    def excavate_step(self, num_samples=100):

        unexcavated = self.sim_state.excavated == False
        targetcol_t = self.sim_state[self.city.target_column] == True

        if self.step_count >= self.max_explore_steps:
            if (
                self.sim_state["excavated"].sum()
                + len(
                    self.sim_state[
                        (self.sim_state.excavated == False)
                        & (self.sim_state.explored == True)
                        & (self.sim_state[self.city.target_column] == True)
                    ]
                )
                > self.max_samples
            ):
                excavate_ids = (
                    self.sim_state[
                        (self.sim_state.excavated == False)
                        & (self.sim_state.explored == True)
                        & (self.sim_state[self.city.target_column] == True)
                    ]
                    .sample(self.max_samples - self.sim_state["excavated"].sum())
                    .index
                )
            else:
                excavate_ids = self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ].index

            if self.trained == True:
                if len(excavate_ids) < num_samples:
                    addtl_samples = num_samples - len(excavate_ids)
                    addtl_ids = (
                        self.curr_yhat[
                            self.curr_yhat.index.isin(
                                self.sim_state[
                                    # self.sim_state[self.city.target_column].isna()
                                    (self.sim_state.explored == False)
                                    & (self.sim_state.excavated == False)
                                ].index
                            )
                        ]
                        .sort_values(ascending=False)[:addtl_samples]
                        .index
                    )

                    excavate_ids = excavate_ids.append(addtl_ids)

            excavated_data = self.replace_lines(excavate_ids)

            return excavated_data
        else:
            return None


class UniformInspectOneShot(Simulator):
    def explore_step(self, num_samples=100):

        if self.step_count < self.max_explore_steps:
            if sum(self.sim_state.excavated == False) < num_samples:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(sum(self.sim_state.explored == False))
                    .index
                )
            else:
                explore_ids = (
                    self.sim_state[self.sim_state.explored == False]
                    .sample(num_samples)
                    .index
                )
            explored_data = self.explore_lines(explore_ids)

            return explored_data
        else:
            return None

    def excavate_step(self, num_samples=100):

        num_samples = self.max_samples

        if self.step_count >= self.max_explore_steps:
            if (
                self.sim_state["excavated"].sum()
                + len(
                    self.sim_state[
                        (self.sim_state.excavated == False)
                        & (self.sim_state.explored == True)
                        & (self.sim_state[self.city.target_column] == True)
                    ]
                )
                > self.max_samples
            ):
                excavate_ids = (
                    self.sim_state[
                        (self.sim_state.excavated == False)
                        & (self.sim_state.explored == True)
                        & (self.sim_state[self.city.target_column] == True)
                    ]
                    .sample(self.max_samples - self.sim_state["excavated"].sum())
                    .index
                )
            else:
                excavate_ids = self.sim_state[
                    (self.sim_state.excavated == False)
                    & (self.sim_state.explored == True)
                    & (self.sim_state[self.city.target_column] == True)
                ].index

            if self.trained == True:
                if len(excavate_ids) < num_samples:
                    addtl_samples = num_samples - len(excavate_ids)
                    addtl_ids = (
                        self.curr_yhat[
                            self.curr_yhat.index.isin(
                                self.sim_state[
                                    self.sim_state[self.city.target_column].isna()
                                ].index
                            )
                        ]
                        .sort_values(ascending=False)[:addtl_samples]
                        .index
                    )

                    excavate_ids = excavate_ids.append(addtl_ids)

            excavated_data = self.replace_lines(excavate_ids)

            return excavated_data
        else:
            return None


class BiasedInspectAdaptiveExcavate(UniformInspectAdaptiveExcavate):
    def __init__(self, bias_weights=None, **kwargs):

        self.bias_weights = bias_weights
        super().__init__(**kwargs)

    def train_step(self):
        Xdata = self.city.data.loc[self.sim_state.explored].drop(
            [self.city.target_column, self.city.unique_id], axis=1
        )
        Ydata = self.sim_state.loc[
            self.sim_state.explored, self.city.target_column
        ].astype(int)

        # WIP: This is coming along - just need to make sure
        #    we do the correct preprocessing etc. here (we are on our own!)
        # Should pass the correct numbers thru the constructor?
        # wts = self.debias_est().fit(self.city.data.drop(
        #    [self.city.target_column, self.city.unique_id], axis=1
        # ), self.sim_state.explored).predict(Xdata)

        # TODO: Where do we get the client model config?
        # TODO: Where do we get the target columns and label columns?
        # Presumably from the config - but what do target_columns and label_column mean
        if self.client_config.get("debias"):
            wts = calculate_debias_weights(
                self.city.data,
                self.client_config["debias"],
                self.sim_state.explored,
            )
        else:
            wts = np.ones(self.sim_state.explored.sum())

        #import pdb; pdb.set_trace()

        sample_weight_kwargs = {
                self.model.steps[-1][0]
                + "__sample_weight": wts
            }
        #    sample_weight_kwargs = {self.model.steps[-1][0] + "__sample_weight": wts}

        self.model.fit(Xdata, Ydata, **sample_weight_kwargs)

        yhat = self.model.predict_proba(
            self.city.data.drop([self.city.target_column, self.city.unique_id], axis=1)
        )[:, 1]

        return pd.Series(yhat, index=self.city.data.index)

    def explore_step(self, num_samples=100):

        if self.step_count < self.max_explore_steps:

            unexplored = self.sim_state.explored == False
            num_samples = min(num_samples, len(self.sim_state[unexplored]))
            explore_ids = (
                self.sim_state[unexplored]
                .sample(num_samples, weights=self.bias_weights[unexplored])
                .index
            )
            return self.explore_lines(explore_ids)

        return None

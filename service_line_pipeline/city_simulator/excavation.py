import numpy as np
import pandas as pd


class excavator_object:
    def __init__(self, batch_size=100, random_state=42, policy="default"):
        self.policy = policy
        self.batch_size = batch_size
        self.random_state = random_state
        self.excavation_funcs = {
            "default": self.get_known_hazards_then_highest_probs,
            "random-guess": self.random_guess,
            "no-inspection;-excavate-top-yhat": self.get_highest_probs,
            "inspect-randomly;-excavate-only-inspected": self.get_known_y_hazardous,
            "inspect-top-yhat;-excavate-only-inspected": self.get_known_y_hazardous,
            "inspect-randomly;-excavate-top-yhat": self.get_known_hazards_then_highest_probs,
            "inspect-med-yhat;-excavate-top-yhat": self.get_known_hazards_then_highest_probs,
        }

    ### get_excavation_locs() is the primary I/O for this module. ###
    ### Should not need to be changed unless for architectural reasons!
    ##################################################################

    def get_excavation_locs(self, simstate):
        # only keep rows where excavated_cycle is nan AND y is (1 or nan).
        simstate = simstate.loc[
            simstate.excavated_cycle.isna() & ((simstate.y.isna()) | (simstate.y == 1))
        ]

        excavation_locs = self.excavation_funcs[self.policy](simstate)

        if len(excavation_locs) < self.batch_size:
            return_length = len(excavation_locs)
        else:
            return_length = self.batch_size

        return excavation_locs.index[:return_length]

    ##################################################################
    ### These, below, are the custom policy functions. ###############
    ### Each of these functions returns a dataframe    ###############
    ### ordered (descending) by excavation location.   ###############

    def get_known_y_hazardous(self, sim_df):
        known_y_hazardous = sim_df.loc[
            (sim_df.y == 1) & (sim_df.excavated_cycle.isna())
        ]

        return known_y_hazardous

    def get_highest_probs(self, sim_df):
        self.check_y_hat_notna(sim_df)

        highest_prob_unknown = sim_df.loc[sim_df.y.isna()].sort_values(
            "y_hat", ascending=False
        )

        return highest_prob_unknown

    def get_median_probs(self, sim_df):
        self.check_y_hat_notna(sim_df)

        sim_df["dist_to_mean"] = np.abs(sim_df.y_hat - sim_df.y_hat.mean())

        return sim_df.sort_values("dist_to_mean")

    def get_known_hazards_then_highest_probs(self, sim_df):

        known_y_hazards = self.get_known_y_hazardous(sim_df)
        highest_probs = self.get_highest_probs(sim_df)

        return known_y_hazards.append(highest_probs)

    def random_guess(self, sim_df):
        needs_replacing = sim_df.loc[(sim_df.y == 1) & (sim_df.excavated_cycle.isna())]

        unknown_y = sim_df.loc[sim_df.y.isna()]

        if len(unknown_y) > self.batch_size:

            unknown_y = unknown_y.sample(
                n=self.batch_size, random_state=self.random_state
            )

        excavation_locs = needs_replacing.append(unknown_y)

        return excavation_locs

    def check_y_hat_notna(self, sim_df):
        if sim_df.y_hat.isna().sum() > 0:
            raise BaseException("y_hat has at least one nan value!")

import numpy as np
import pandas as pd


class inspector_object:
    def __init__(
        self, batch_size=100, random_state=None, start_size=None, policy="default"
    ):

        self.random_state = random_state
        self.policy = policy
        self.batch_size = batch_size
        self.start_size = start_size or 100
        self.inspection_funcs = {
            "default": self.random_guess,
            "random-guess": self.random_guess,
            "no-inspection;-excavate-top-yhat": None,
            "inspect-randomly;-excavate-only-inspected": self.random_guess,
            "inspect-top-yhat;-excavate-only-inspected": self.get_highest_probs,
            "inspect-randomly;-excavate-top-yhat": self.random_guess,
            "inspect-med-yhat;-excavate-top-yhat": self.get_median_probs,
        }

    ### get_inspection_locs() is the primary I/O for this module. ###
    ### Should not need to be changed unless for architectural reasons!
    ##################################################################

    def get_inspection_locs(self, simstate):
        df = simstate

        if len(df) < self.batch_size:
            return_length = len(df)
        else:
            return_length = self.batch_size

        # If zero known y, guess randomly.
        if len(df.loc[df.y.notna()]) == 0:
            # TO-DO: move this to a debugging log statement.
            # print(
            #     """
            # Starting with zero known y; inspection starting with a
            # random guess of length {}.""".format(
            #         self.start_size
            #     )
            # )
            locs = self.random_guess(sim_df=df, batch_size=self.start_size)
            return locs.index[:self.start_size]

        # If nothign to inspect, return empty.
        elif len(df.loc[df.y.isna()]) == 0:
            return np.array([])

        # get inspection locations (this is most common condition).
        else:
            if self.inspection_funcs[self.policy] is None:
                return np.array([])

            else:
                inspection_locs = self.inspection_funcs[self.policy](
                    sim_df=simstate.loc[simstate.y.isna()]
                )

            return inspection_locs.index[:return_length]

    ##################################################################
    #### These, below, are the custom policy functions. ##############

    def get_highest_probs(self, sim_df):

        highest_prob_unknown = sim_df.loc[sim_df.y.isna()].sort_values(
            "y_hat", ascending=False
        )

        return highest_prob_unknown

    def get_median_probs(self, sim_df):
        sim_df = sim_df.copy()
        dist_to_mean = np.abs(sim_df.y_hat - sim_df.y_hat.mean())
        sim_df["dist_to_mean"] = dist_to_mean

        return sim_df.sort_values("dist_to_mean")

    def random_guess(self, sim_df, batch_size=None):

        if not batch_size:
            batch_size = self.batch_size

        unknown_y = sim_df.loc[sim_df.y.isna()]

        if len(unknown_y) > batch_size:

            unknown_y = unknown_y.sample(n=batch_size, random_state=self.random_state)

        locs = unknown_y

        return locs

    def no_guess(self, sim_df):
        return pd.DataFrame({})

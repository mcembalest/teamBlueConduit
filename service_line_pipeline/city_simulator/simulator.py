import pandas as pd
import numpy as np

from service_line_pipeline.city_simulator import (
    inspection,
    excavation,
    live_analysis,
)

#
# def run_n_times(n, **kwargs):
#     output = []
#     for i in n:
#         simstate = run_once(**kwargs)


def run_once(
    model,
    X,
    y,
    train_stride=5,
    excavate_batch_size=100,
    inspect_batch_size=100,
    first_inspect_size=None,
    random_state=None,
    max_cycles=100,
    policy="default",
    sampling_frac=0.8,
    iteration=None,
    print_log=True,
    recording_state=False,
):
    X, y = get_random_sample_X_y(X, y, sampling_frac, random_state)

    simstate = simulation_state(X)
    gtruth = ground_truth(X, y)
    inspector = inspection.inspector_object(
        batch_size=inspect_batch_size, random_state=random_state, policy=policy,
        start_size=first_inspect_size

    )
    excavator = excavation.excavator_object(
        batch_size=excavate_batch_size, random_state=random_state, policy=policy
    )
    recorder = live_analysis.recorder(recording_state, X=X)

    train_cycles = get_train_cycles(
        train_stride=train_stride,
        max_cycles=max_cycles,
        random_state=random_state,
    )
    for cycle in range(max_cycles):
        # Safeguard against infinite loops.
        if simstate.nothing_left_to_explore() or simstate.inactivity(cycles=5):
            break

        # Inspection: choose locations; get results; update simstate.
        inspection_locs = inspector.get_inspection_locs(simstate.df)
        inspection_results_df = gtruth.get_label_results(locs=inspection_locs)
        simstate.update_inspected(inspection_results_df, cycle=cycle)

        # Re-train model on current known data.
        if cycle in train_cycles:
            known_y = simstate.get_known_y()
            model.fit(X.loc[known_y.index], known_y.y)
            y_hat = get_curr_yhat(model, X)
            simstate.update_yhat(y_hat)
            recorder.record_snapshot(y_hat, cycle)
            print_progress(cycle, simstate, iteration, print_at_all=print_log)

        # Excavation: choose locations; get results; update simstate.
        excavation_locs = excavator.get_excavation_locs(simstate.df)
        excavation_results_df = gtruth.get_label_results(locs=excavation_locs)
        simstate.update_excavated(excavation_results_df, cycle=cycle)

        if cycle > max_cycles: print("\nMax cycles exceeded.")

    return simstate.df, recorder.df


def print_progress(cycle, simstate, iteration=None, print_at_all=True):
    if not print_at_all:
        return
    len_unknown_y = simstate.df.y.isna().sum()
    if iteration is None:
        print(
            "",
            end="\r cycle: {}; unknown labels: {}".format(
                str(cycle).zfill(4), str(len_unknown_y).zfill(5)
            ),
            flush=True,
        )
    else:
        print(
            "",
            end="\r iteration: {}; cycle: {}; unknown labels: {}".format(
                str(iteration).zfill(3),
                str(cycle).zfill(4),
                str(len_unknown_y).zfill(5),
            ),
            flush=True,
        )


class simulation_state:
    """
    Represents the knowledge we have as we inspect an unknown
    city in real time. Does not know about the labels!
    (Extra precaution for preventing data leakage.)
    """

    def __init__(self, X):
        self.df = pd.DataFrame(
            {
                "y": np.nan,
                "y_hat": np.nan,
                "excavated_cycle": np.nan,
                "inspected_cycle": np.nan,
            },
            index=X.index,
        )
        self.inspection_inactivity_cycles = 0
        self.excavation_inactivity_cycles = 0

    def update_inspected(self, inspection_results_df, cycle):
        if len(inspection_results_df) == 0:
            self.inspection_inactivity_cycles += 1
            return
        self.inspection_inactivity_cycles = 0

        # update discovered y values; set these two columns as well:
        inspection_results_df[["inspected_cycle"]] = cycle
        self.df.update(inspection_results_df)

    def update_excavated(self, excavation_results_df, cycle):
        df = excavation_results_df.copy()
        if len(df) == 0:
            self.excavation_inactivity_cycles += 1
            return
        self.excavation_inactivity_cycles = 0

        df["excavated_cycle"] = cycle

        self.df.update(df)

    def get_known_y(self):
        return self.df.loc[self.df.y.notna(), ["y"]]

    def update_yhat(self, y_hat):
        self.df["y_hat"] = y_hat

    def nothing_left_to_explore(self):
        nothing_left = self.df.y.isna().sum() < 2
        # if nothing_left:
        #     print("\nnothing left to explore")
        return nothing_left

    def inactivity(self, cycles=5):
        inactivity = (
            # Allowing inspection to return 0-length for now,
            # as it's in policies without inspection.
            self.inspection_inactivity_cycles > cycles and
             self.excavation_inactivity_cycles > cycles
        )
        # if inactivity:
        #     print("\nInactivity!")

        return inactivity


def get_curr_yhat(model, X, **kwargs):
    yhat = model.predict_proba(X, **kwargs)
    if yhat.shape[1] == 1:
        return yhat
    else:
        return yhat[:, 1]


def get_random_sample_X_y(X, y, sampling_frac, random_state):
    y_df = pd.DataFrame({"y": y}, index=X.index)
    X_sample = X.sample(int(sampling_frac * len(X)), random_state=random_state)
    # X_sample = X_sample.sort_index(axis=0)
    y_sample = y_df.loc[X_sample.index].y

    return X_sample, y_sample


class ground_truth:
    """
    Represents the true state of lead label knowledge. This is
    for the sake of this simulation being very careful
    to prevent any data leakage into our simulation state as we
    "discover" the labels anew with each simulation. This is the
    *only* place the y knowledge should live, once everything
    is instanstiated.
    """

    def __init__(self, X, y):
        self.df = pd.DataFrame({"y": y}, index=X.index)

    def get_label_results(self, locs):
        return self.df.loc[locs][["y"]]


def get_train_cycles(train_stride, max_cycles, random_state):
    if not type(train_stride) == int:
        raise TypeError("train_stride must be of type 'int'.")
    if train_stride < 1:
        raise BaseException("Train stride must be 1 or greater.")
    elif train_stride == 1:
        train_cycles = np.arange(max_cycles)
    elif train_stride > 1:
        rng = np.random.default_rng(random_state)
        rand_ints = rng.integers(
            low=1,
            high= (2 * train_stride),
            size=max_cycles
        )
        train_cycles = [0]
        train_cycles += list(rand_ints.cumsum())

    return train_cycles

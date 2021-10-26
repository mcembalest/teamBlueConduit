import pandas as pd
import numpy as np

from service_line_pipeline.city_simulator import (
    simulator,
    inspection,
    excavation,
    analyze_simstate,
    rule_based_models,
)


### generate fake simstate
def make_testing_simstate():
    simstate = pd.DataFrame(
        {
            "y": [],
            "y_hat": [],
            "excavated_cycle": np.nan,
            "inspected_cycle": np.nan,
        }
    )

    rows = [
        # cycle 0
        [0, 0.2, 0, np.nan],
        [0, 0.9, np.nan, 0],
        [1, 0.8, 0, np.nan],
        [0, 0.1, np.nan, 0],
        # cycle 1
        [0, 0.03, 1, np.nan],
        [1, 0.93, 1, 1],
        [1, 0.3, 1, np.nan],
        [0, 0.89, np.nan, 1],
        [1, 0.84, 1, np.nan],
        # cycle 2
        [1, 0.03, 2, 2],
        [0, 0.03, np.nan, 2],
        [1, 0.93, 2, 2],
        [1, 0.3, 2, np.nan],
        [0, 0.89, np.nan, 2],
        [0, 0.84, 2, np.nan],
    ]

    for i, row in enumerate(rows):
        simstate.loc[i] = row

    return simstate


def count_y1s_where_excavated_inspected(simstate):
    return simstate.loc[
        (simstate.inspected_cycle.notna())
        & (simstate.excavated_cycle.notna())
        & (simstate.y.astype(bool))
    ].shape[0]


def count_y0s_where_only_inspected(simstate):
    return simstate.loc[
        (simstate.inspected_cycle.notna())
        & (simstate.excavated_cycle.notna())
        & (~simstate.y.astype(bool))
    ].shape[0]


### def test simstate validity
def test_simstate_validity():
    simstate = make_testing_simstate()

    assert count_y1s_where_excavated_inspected(simstate) != 0
    assert count_y0s_where_only_inspected(simstate) == 0


def test_simstate_metrics():
    simstate = make_testing_simstate()

    df = analyze_simstate.get_metrics_by_cycle_df(simstate)

    assert list(df["excavation_hit_rate"]) == [
        0.5,
        0.75,
        0.75,
    ]

    assert list(df["inspection_hit_rate"]) == [
        0,
        0.5,
        0.5,
    ]

    assert list(df["total_replaced"]) == [
        1,
        3,
        3,
    ]

    # 5000 for each excavation with replacement;
    # 3000 for each excavation without.
    assert list(df["total_excavation_cost"]) == [
        5000 + 3000,
        15000 + 3000,
        15000 + 3000,
    ]

    # 300 for each inspection;
    assert list(df["inspect_cost"]) == [
        300 * 2,
        300 * 2,
        300 * 4,
    ]


### test simulation metrics

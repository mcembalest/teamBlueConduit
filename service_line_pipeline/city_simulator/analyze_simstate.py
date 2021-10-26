import pandas as pd
import numpy as np


def get_packaged_analysis(simstate):
    df = get_metrics_by_cycle_df(simstate)

    keep_features = [
        "total_replaced",
        "total_replaced_cumulative",
        "total_excavation_cost",
        "total_cost",
        "excavation_hit_rate",
        "inspection_hit_rate",
        "cumulative_hitrate",
        "total_cost_cumulative",
        "mean_cost_replacement",
        "total_cost_replacement_cumulative",
    ]

    output = {}
    for feature in keep_features:
        output[feature] = df[feature]

    return output


# NOTES ON SIMULATION PRICING
# inspection = $300
# excavate + replace = $5000
# excavate + don't replace = $3000
# See video of mcdaniel on PBS watching an actual replacement.

# one policy would be to only excavate where you've inspected + found a lead pipe.
# hazard_rate; hazardous.

# also look at average cost of replacement.


def get_df_indexed_by_cycle(simstate):
    excavate_cycles = set(simstate.excavated_cycle.value_counts().index)
    inspect_cycles = set(simstate.inspected_cycle.value_counts().index)

    total = excavate_cycles.union(inspect_cycles)
    total = np.array(list(total))

    return pd.DataFrame([], index=total)


def get_inspect_excavate_curves(simstate):

    # Make an empty dataframe with an index of unique cycles.
    excavate_cycles = set(simstate.excavated_cycle.value_counts().index)
    inspect_cycles = set(simstate.inspected_cycle.value_counts().index)
    unique_cycles = excavate_cycles.union(inspect_cycles)
    unique_cycles = np.array(list(unique_cycles))
    df = pd.DataFrame([], index=unique_cycles)
    df = df.rename_axis("cycle")

    # Add the inspections.
    df = df.merge(
        pd.DataFrame(simstate.groupby("inspected_cycle").count().y),
        left_index=True,
        right_index=True,
        how="left",
    )

    # Add the excavations.
    df = df.merge(
        pd.DataFrame(simstate.groupby("excavated_cycle").count().y),
        left_index=True,
        right_index=True,
        how="left",
    )

    df = df.rename(columns={"y_x": "inspections", "y_y": "excavations"})

    df = df.fillna(0)

    return df


def get_metrics_by_cycle_df(
    simstate,
    inspection_cost=300,
    excavation_without_replacement_cost=3000,
    excavation_with_replacement_cost=5000,
):
    df = get_inspect_excavate_curves(simstate)

    df["inspect_cost"] = df["inspections"].fillna(0) * inspection_cost

    # Add the excavations that AREN'T replaced.
    excavated_not_replaced = pd.DataFrame(
        simstate.loc[(simstate.y == 0)].groupby("excavated_cycle").y.count()
    )
    excavated_not_replaced = excavated_not_replaced.rename(
        columns={"y": "exca_no_replaced"}
    )
    df = df.merge(excavated_not_replaced, how="left", left_index=True, right_index=True)
    df["exca_no_replaced"] = df["exca_no_replaced"].fillna(0)
    df["exca_no_replaced cost"] = (
        df["exca_no_replaced"].fillna(0) * excavation_without_replacement_cost
    )

    # Add the excavations that ARE replaced.
    excavated_and_replaced = pd.DataFrame(
        simstate.loc[(simstate.y == 1)].groupby("excavated_cycle").y.count()
    )
    excavated_and_replaced = excavated_and_replaced.rename(
        columns={"y": "exca_yes_replaced"}
    )
    df = df.merge(excavated_and_replaced, how="left", left_index=True, right_index=True)
    df["exca_yes_replaced"] = df["exca_yes_replaced"].fillna(0)
    df["exca_yes_replaced cost"] = (
        df["exca_yes_replaced"].fillna(0) * excavation_with_replacement_cost
    )

    # compute total_costs
    df["total_excavation_cost"] = (
        df["exca_yes_replaced cost"] + df["exca_no_replaced cost"]
    )
    df["total_cost"] = df["total_excavation_cost"] + df["inspect_cost"]
    df["total_cost_cumulative"] = df["total_cost"].cumsum()

    # compute excavation_hit_rate
    df["excavation_hit_rate"] = df["exca_yes_replaced"].fillna(0) / (
        df["exca_no_replaced"].fillna(0) + df["exca_yes_replaced"].fillna(0)
    )
    df["excavation_hit_rate"] = df["excavation_hit_rate"].fillna(0)

    # compute inspection_hit_rate.
    inspected_no_hazard = pd.DataFrame(
        simstate.loc[(simstate.y == 0)].groupby("inspected_cycle").y.count()
    ).rename(columns={"y": "inspected_no_hazard"})
    inspected_yes_hazard = pd.DataFrame(
        simstate.loc[(simstate.y == 1)].groupby("inspected_cycle").y.count()
    ).rename(columns={"y": "inspected_yes_hazard"})

    df = df.merge(inspected_no_hazard, how="left", left_index=True, right_index=True)
    df = df.merge(inspected_yes_hazard, how="left", left_index=True, right_index=True)
    df["inspected_no_hazard"] = df["inspected_no_hazard"].fillna(0)
    df["inspected_yes_hazard"] = df["inspected_yes_hazard"].fillna(0)

    df["inspection_hit_rate"] = df["inspected_yes_hazard"].fillna(0) / (
        df["inspected_no_hazard"].fillna(0) + df["inspected_yes_hazard"].fillna(0)
    )

    # compute total_replaced.
    df["total_replaced"] = df["exca_yes_replaced"]
    df["total_replaced_cumulative"] = df["total_replaced"].cumsum()
    df["mean_cost_replacement"] = (
        df["total_cost"].cumsum() / df["total_replaced_cumulative"]
    )
    df["total_cost_replacement_cumulative"] = df["mean_cost_replacement"].cumsum()

    # compute cumulative excavation hit-rate
    cumulative_hitrate = simstate.groupby("excavated_cycle").y.sum().cumsum()
    cumulative_hitrate /= simstate.groupby("excavated_cycle").y.count().cumsum()
    cumulative_hitrate = cumulative_hitrate.rename("cumulative_hitrate")
    df = df.merge(cumulative_hitrate, how="left", left_index=True, right_index=True)

    return df

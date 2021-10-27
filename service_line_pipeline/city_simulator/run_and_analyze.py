# import pandas as pd
# import numpy as np
#
# from service_line_pipeline.city_simulator import simulator
#
#
# def run_policies_models_n_times_each(
#     models,
#     X,
#     y,
#     policies=["default", "random guess"],
#     **kwargs,
# ):
#     full_df = pd.DataFrame([])
#     for model_name, model in models.items():
#         for policy in policies:
#             print("Model: '{}'; policy: '{}'".format(model_name, policy))
#             model_policy_n_metrics_df = run_n_times(
#                 model, X, y, policy=policy, **kwargs
#             )
#             model_policy_n_metrics_df[["model name", "policy"]] = model_name, policy
#             full_df = full_df.append(model_policy_n_metrics_df)
#
#     return full_df
#
#
# def run_n_times(
#     model,
#     X,
#     y,
#     n_simulations=1,
#     excavate_batch_size=100,
#     inspect_batch_size=100,
#     train_stride=5,
#     max_cycles=500,
#     random_state=None,
#     sampling_frac=0.8,
#     policy="default",
# ):
#     analysis_data = {}
#
#     # Run the simulation
#     for i in range(n_simulations):
#         simstate = simulator.run_once(
#             model=model,
#             X=X,
#             y=y,
#             excavate_batch_size=excavate_batch_size,
#             inspect_batch_size=inspect_batch_size,
#             train_stride=train_stride,
#             max_cycles=max_cycles,
#             random_state=random_state,
#             sampling_frac=sampling_frac,
#             policy=policy,
#             iteration=i,
#         )
#
#         # Store the analysis of each simulation's output.
#         analysis_dict = get_packaged_analysis(simstate)
#         analysis_data[i] = analysis_dict
#
#     # Sort the analysis data into a big tidy-format dataframe.
#     data_names = analysis_data[0].keys()
#     big_df = pd.DataFrame(columns=["cycle", "metric", "value", "policy"])
#     for metric in data_names:
#         for i, data_dict in analysis_data.items():
#             little_df = pd.DataFrame(
#                 {
#                     "value": data_dict[metric],
#                     "metric": metric,
#                     "cycle": data_dict[metric].index,
#                     "sim_iteration": i,
#                     "policy": policy,
#                 }
#             )
#             big_df = big_df.append(little_df, ignore_index=True)
#
#     return big_df
#
#
# def get_packaged_analysis(simstate):
#     output = {
#         "Excavation hit-rate curve": get_excavation_hitrate_curve(simstate),
#         "Inspection hit-rate curve": get_inspection_hitrate_curve(simstate),
#         "Replacement curve": get_replacement_curve(simstate),
#         "Total cost curve": get_inspection_cost_curve(simstate)
#         + get_excavation_cost_curve(simstate),
#         #'Total cost': compute_total_cost(simstate),
#         #'Excavation cost curve': get_excavation_cost_curve(simstate),
#         #'Inspection cost curve': get_inspection_cost_curve(simstate),
#     }
#
#     return output
#
#
# # NOTES ON SIMULATION PRICING
# # inspection = $300
# # excavate + replace = $5000
# # excavate + don't replace = $3000
# # See video of mcdaniel on PBS watching an actual replacement.
#
# # one policy would be to only excavate where you've inspected + found a lead pipe.
# # hazard_rate; hazardous.
#
# # also look at average cost of replacement.
#
#
# def get_df_indexed_by_cycle(simstate):
#     excavate_cycles = set(simstate.excavated_cycle.value_counts().index)
#     inspect_cycles = set(simstate.inspected_cycle.value_counts().index)
#
#     total = excavate_cycles.union(inspect_cycles)
#     total = np.array(list(total))
#
#     return pd.DataFrame([], index=total)
#
#
# def get_inspect_excavate_curves(simstate):
#
#     # Make an empty dataframe with an index of unique cycles.
#     excavate_cycles = set(simstate.excavated_cycle.value_counts().index)
#     inspect_cycles = set(simstate.inspected_cycle.value_counts().index)
#     unique_cycles = excavate_cycles.union(inspect_cycles)
#     unique_cycles = np.array(list(unique_cycles))
#     df = pd.DataFrame([], index=unique_cycles)
#
#     # Add the inspections.
#     df = df.merge(
#         pd.DataFrame(simstate.groupby("inspected_cycle").count().y),
#         left_index=True,
#         right_index=True,
#         how="left",
#     )
#
#     # Add the excavations.
#     df = df.merge(
#         pd.DataFrame(simstate.groupby("excavated_cycle").count().y),
#         left_index=True,
#         right_index=True,
#         how="left",
#     )
#
#     df = df.rename(columns={"y_x": "inspections", "y_y": "excavations"})
#
#     df = df.fillna(0)
#
#     return df
#
#
# def get_metrics_by_cycle_df(
#     simstate,
#     inspection_cost=300,
#     excavation_without_replacement_cost=3000,
#     excavation_with_replacement_cost=5000,
# ):
#     df = get_inspect_excavate_curves(simstate)
#
#     df["inspect cost"] = df["inspections"].fillna(0).cumsum() * inspection_cost
#
#     # Add the excavations that AREN'T replaced.
#     excavated_not_replaced = pd.DataFrame(
#         simstate.loc[(simstate.y == 0)].groupby("excavated_cycle").y.count()
#     )
#     excavated_not_replaced = excavated_not_replaced.rename(
#         columns={"y": "exca no replaced"}
#     )
#     df = df.merge(excavated_not_replaced, how="left", left_index=True, right_index=True)
#     df["exca no replaced"] = df["exca no replaced"].fillna(0)
#     df["exca no replaced cost"] = (
#         df["exca no replaced"].fillna(0).cumsum() * excavation_without_replacement_cost
#     )
#
#     # Add the excavations that ARE replaced.
#     excavated_and_replaced = pd.DataFrame(
#         simstate.loc[(simstate.y == 1)].groupby("excavated_cycle").y.count()
#     )
#     excavated_and_replaced = excavated_and_replaced.rename(
#         columns={"y": "exca yes replaced"}
#     )
#     df = df.merge(excavated_and_replaced, how="left", left_index=True, right_index=True)
#     df["exca yes replaced"] = df["exca yes replaced"].fillna(0)
#     df["exca yes replaced cost"] = (
#         df["exca yes replaced"].fillna(0).cumsum() * excavation_with_replacement_cost
#     )
#
#     # compute total costs
#     df["total excavation cost"] = (
#         df["exca yes replaced cost"] + df["exca no replaced cost"]
#     )
#     df["total cost"] = df["total excavation cost"] + df["inspect cost"]
#
#     # compute excavation hit-rate
#     df["excavation hit-rate"] = df["exca yes replaced"].fillna(0) / (
#         df["exca no replaced"].fillna(0) + df["exca yes replaced"].fillna(0)
#     )
#
#     # compute inspection hit-rate.
#     inspected_no_hazard = pd.DataFrame(
#         simstate.loc[(simstate.y == 0)].groupby("inspected_cycle").y.count()
#     ).rename(columns={"y": "inspected_no_hazard"})
#     inspected_yes_hazard = pd.DataFrame(
#         simstate.loc[(simstate.y == 1)].groupby("inspected_cycle").y.count()
#     ).rename(columns={"y": "inspected_yes_hazard"})
#
#     df = df.merge(inspected_no_hazard, how="left", left_index=True, right_index=True)
#     df = df.merge(inspected_yes_hazard, how="left", left_index=True, right_index=True)
#     df["inspected_no_hazard"] = df["inspected_no_hazard"].fillna(0)
#     df["inspected_yes_hazard"] = df["inspected_yes_hazard"].fillna(0)
#
#     df["inspection hit-rate"] = df["inspected_yes_hazard"].fillna(0) / (
#         df["inspected_no_hazard"].fillna(0) + df["inspected_yes_hazard"].fillna(0)
#     )
#
#     return df
#
#
# def compute_total_cost(
#     simstate,
#     inspection_cost=300,
#     excavation_with_replacement_cost=3000,
#     excavation_without_replacement_cost=5000,
# ):
#     inspection_cost = simstate.inspected.sum() * inspection_cost
#     excavation_cost_no_replacement = (
#         simstate.loc[
#             (simstate_random.y == 0) & (simstate.excavated == True)
#         ].excavated.sum()
#         * excavation_without_replacement_cost
#     )
#
#     excavation_cost_with_replacement = (
#         simstate.loc[
#             (simstate.replaced_cycle.notna()) & (simstate.excavated_cycle.notna())
#         ].replaced_cycle.count()
#         * excavation_with_replacement_cost
#     )
#
#     return (
#         inspection_cost
#         + excavation_cost_no_replacement
#         + excavation_cost_with_replacement
#     )
#
#
# def get_inspection_cost_curve(
#     simstate,
#     inspection_cost=1000,
#     excavation_cost=5000,
# ):
#     inspection_cost_vec = (
#         simstate.groupby("inspected_cycle").y.count() * inspection_cost
#     )
#
#     return inspection_cost_vec.cumsum()
#
#
# def get_excavation_cost_curve(
#     simstate,
#     inspection_cost=1000,
#     excavation_cost=5000,
# ):
#     excavation_cost_vec = (
#         simstate.groupby("excavated_cycle").y.count() * excavation_cost
#     )
#
#     return excavation_cost_vec.cumsum()
#
#
# def get_excavation_hitrate_curve(simstate):
#     """
#     The ratio, at each cycle, of number of
#     lead-positives discovered to the number of pipes excavated.
#     """
#
#     df = simstate.groupby("excavated_cycle").y.agg(["count", "sum"])
#     df["excavation hit-rate"] = df["sum"] / df["count"]
#
#     return df["excavation hit-rate"]
#
#
# def get_inspection_hitrate_curve(simstate):
#     """
#     The ratio, at each cycle, of number of
#     lead-positives discovered to the number of pipes checked.
#     """
#     df = simstate.groupby("inspected_cycle").y.agg(["count", "sum"])
#     df["inspection hit-rate"] = df["sum"] / df["count"]
#
#     return df["inspection hit-rate"]
#
#
# def get_replacement_curve(simstate):
#     """
#     The number of replacements at each cycle
#     """
#     # return simstate.groupby('excavated_cycle').y.sum().cumsum()
#     return simstate.groupby("excavated_cycle").agg("sum").y.cumsum()
#
#
# def get_inspection_rate(simstate):
#
#     return simstate.groupby("inspected_cycle").agg("sum").y.cumsum()
#
#
# def get_discovery_curve(simstate):
#     inspected_hazardous_curve = (
#         simstate.groupby("inspected_cycle").agg("sum").y.cumsum()
#     )
#     excavated_hazardous_curve = (
#         simstate.groupby("excavated_cycle").agg("sum").y.cumsum()
#     )
#
#     total_discovery_curve = inspected_hazardous_curve + excavated_hazardous_curve
#
#     return total_discovery_curve

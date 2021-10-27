import pandas as pd
import numpy as np

from service_line_pipeline.city_simulator import simulator
from service_line_pipeline.city_simulator.analyze_simstate import get_packaged_analysis


def run_policies_models_n_times_each(
    models,
    X,
    y,
    policies=["default", "random guess"],
    print_log=False,
    **kwargs,
):
    full_df = pd.DataFrame([])
    counter = 0
    for model_name, model in models.items():
        for policy in policies:
            print(
                "",
                end="\r Progress: {}/{}".format(
                    str(counter).zfill(3),
                    str(len(policies) * len(models.keys())).zfill(3)
                )
            )
            if print_log:
                print("\nModel: '{}'; policy: '{}'".format(model_name, policy))
            model_policy_n_metrics_df = run_n_times(
                model, X, y, policy=policy, print_log=print_log, **kwargs
            )
            model_policy_n_metrics_df[["model name", "policy"]] = model_name, policy
            full_df = full_df.append(model_policy_n_metrics_df)
            counter += 1

    return full_df


def run_n_times(
    model,
    X,
    y,
    n_simulations=1,
    excavate_batch_size=100,
    inspect_batch_size=100,
    first_inspect_size=None,
    train_stride=5,
    max_cycles=500,
    random_state=None,
    sampling_frac=0.8,
    policy="default",
    print_log=True,
):
    analysis_data = {}

    # Run the simulation
    for i in range(n_simulations):
        simstate = simulator.run_once(
            model=model,
            X=X,
            y=y,
            excavate_batch_size=excavate_batch_size,
            inspect_batch_size=inspect_batch_size,
            first_inspect_size=first_inspect_size,
            train_stride=train_stride,
            max_cycles=max_cycles,
            random_state=random_state,
            sampling_frac=sampling_frac,
            policy=policy,
            iteration=i,
            print_log=print_log,
        )[0]

        # Store the analysis of each simulation's output.
        analysis_dict = get_packaged_analysis(simstate)
        analysis_data[i] = analysis_dict

    # Sort the analysis data into a big tidy-format dataframe.
    data_names = analysis_data[0].keys()
    big_df = pd.DataFrame(columns=["cycle", "metric", "value", "policy"])
    for metric in data_names:
        for i, data_dict in analysis_data.items():
            little_df = pd.DataFrame(
                {
                    "value": data_dict[metric],
                    "metric": metric,
                    "cycle": data_dict[metric].index,
                    "sim_iteration": i,
                    "policy": policy,
                }
            )
            big_df = big_df.append(little_df, ignore_index=True)

    return big_df

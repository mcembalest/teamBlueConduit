import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def plot_analysis(metrics_df, hue="model name", style=None):

    metrics_df = metrics_df.copy()
    metrics_df = metrics_df.reset_index(drop=True)

    metrics_df = metrics_df.loc[
        (metrics_df.metric == "total_replaced_cumulative")
        | (metrics_df.metric == "excavation_hit_rate")
        | (metrics_df.metric == "cumulative_hitrate")
        | (metrics_df.metric == "mean_cost_replacement")
        | (metrics_df.metric == "total_cost_replacement_cumulative")
    ]

    fig1, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=False)

    metrics = [
        "excavation_hit_rate",
        "cumulative_hitrate",
        "total_replaced_cumulative",
        "total_cost_replacement_cumulative",
    ]
    ylabels = ["", "", "Replacements", "Cost"]
    titles = [
        "Excavation hit-rate",
        "Cumulative excavation hit-rate\n(Total-replacements / total-excavations)",
        "Total lines replaced",
        "Cost-per-replacement",
    ]

    index = 0
    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            ax.set_xlabel("Excavation batch (100 each)")
            ax.set_ylabel(ylabels[index])
            ax.set_title(titles[index])
            metric = metrics[index]
            sns.lineplot(
                data=metrics_df.loc[metrics_df.metric == metric],
                x="cycle",
                ax=ax,
                y="value",
                hue=hue,
                style=None,
            )
            index += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=metrics_df.loc[metrics_df.metric == "mean_cost_replacement"],
        x="cycle",
        ax=ax,
        y="value",
        hue=hue,
    )
    ax.set_xlabel("Excavation batch (100 each)")
    ax.set_ylabel("Cost")
    ax.set_title("Mean cost of pipe replacement")

    return fig1


def plot_facetgrid(metrics_df):
    metrics_df = full_df.loc[
        (full_df.metric == "total_replaced_cumulative")
        | (full_df.metric == "excavation_hit_rate")
        | (full_df.metric == "cumulative_hitrate")
        | (full_df.metric == "mean_cost_replacement")
        | (full_df.metric == "total_cost_replacement_cumulative")
    ]

    g = sns.FacetGrid(
        data=metrics_df, row="metric", sharex=True, sharey=False, height=2, aspect=2
    )
    g.map_dataframe(sns.lineplot, x="cycle", y="value", hue="model name")
    g.add_legend()

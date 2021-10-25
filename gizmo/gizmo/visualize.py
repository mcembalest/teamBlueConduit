#!/usr/bin/env python3
"""
Functions in this module should take dataframes of metrics collected from
multiple runs under some sampling or resampling scheme, such as k-fold
cross-validation, learning curves, bootstrapped datasets, etc,.
"""

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from gizmo.bc_logger import get_simple_logger
from gizmo.utils import make_uri_friendly_string
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


log = get_simple_logger(__name__)

BC_COLORS = {
    "primary-blue": "#184E84",
    "secondary-blue": "#5CA2E6",
    "primary-gold": "#F4D396",
}


def save_figure(fig, figure_dir, filename):

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(filename + ".png")
    figure_filepath = (figure_dir / filename).resolve()

    log.info(
        "Saving figure",
        figure_dir=figure_dir.name,
        figure_filename=figure_filepath.name,
    )
    fig.savefig(figure_filepath)

    plt.close(fig)

    return figure_filepath


def plot_roc_curves(
    roc_curve_info_array,
    ax_roc_curve,
    line_color="cyan",
    mean_line_color="b",
    ci_color="blue",
    line_style=None,
    model_name=None,
):
    auc_fpr_means = np.linspace(0, 1, 100)
    auc_tpr_means = []
    aucs = []
    for curve_info_one_run in roc_curve_info_array:
        fpr, tpr = curve_info_one_run.get("fpr"), curve_info_one_run.get("tpr")
        ax_roc_curve.plot(
            fpr, tpr, color=line_color, lw=1, alpha=0.5, label="_nolegend_"
        )

        # Save these for mean curve and std plot
        auc_tpr_means.append(np.interp(auc_fpr_means, fpr, tpr))
        auc_tpr_means[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(auc_tpr_means, axis=0)
    mean_tpr[-1] = 1.0

    # Plot mean of ROC curve
    mean_auc = auc(auc_fpr_means, mean_tpr)
    std_auc = np.std(aucs)

    ax_roc_curve.set_title("Mean ROC Curve with Std. Dev.")

    ax_roc_curve.plot(
        auc_fpr_means,
        mean_tpr,
        color=mean_line_color,
        label=r"%s (AUC = %0.2f $\pm$ %0.2f)" % (model_name, mean_auc, std_auc),
        lw=5,
        linestyle=line_style,
    )

    # Draw and fill in lines +/- std from mean roc_curve
    std_tpr = np.std(auc_tpr_means, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc_curve.fill_between(
        auc_fpr_means,
        tprs_lower,
        tprs_upper,
        color=ci_color,
        alpha=0.35,
        # label=r"$\pm$ 1 std. dev.",
    )

    # Plot housekeeping.
    ax_roc_curve.set_xlim([-0.05, 1.05])
    ax_roc_curve.set_ylim([-0.05, 1.05])
    ax_roc_curve.set_xlabel("False Positive Rate")
    ax_roc_curve.set_ylabel("True Positive Rate")
    ax_roc_curve.legend(loc="lower right")

    return ax_roc_curve


def plot_precision_recall_curves(
    precision_recall_info_array,
    ax_precision_recall,
    line_color="blue",
    mean_line_color="red",
    line_style=None,
    model_name=None,
):
    pr_fpr_means = np.linspace(0, 1, 100)
    pr_means = []
    precision_arrays = []
    recall_arrays = []
    for curve_info_one_run in precision_recall_info_array:
        prc, rcl = curve_info_one_run.get("prc"), curve_info_one_run.get("rcl")
        precision_arrays.append(prc)
        recall_arrays.append(rcl)
        ax_precision_recall.plot(
            rcl, prc, color=line_color, lw=1, alpha=0.5, label="_nolegend_"
        )

        # roc_auc = auc(fpr, tpr)
        # aucs.append(roc_auc)

    # Calculate and plot the mean PR curve.
    x = max([len(p) for p in precision_arrays])
    for p in range(0, len(precision_arrays)):
        while len(precision_arrays[p]) < x:
            precision_arrays[p] = np.append(precision_arrays[p], [1])
    mean_pr = np.mean(precision_arrays, axis=0)

    x = max([len(p) for p in recall_arrays])
    for p in range(0, len(recall_arrays)):
        while len(recall_arrays[p]) < x:
            recall_arrays[p] = np.append(recall_arrays[p], [0])
    mean_rc = np.mean(recall_arrays, axis=0)

    # mean_apc = pd.Series(apcs).mean()
    # std_apc = pd.Series(apcs).std()
    ax_precision_recall.set_title("Mean PR Curve")
    # Plot Mean PR Curve
    ax_precision_recall.plot(
        mean_rc,
        mean_pr,
        color=mean_line_color,
        label=model_name,  # ; AP: %0.2f $\pm$ %0.2f)  % (mean_apc, std_apc),
        lw=5,
        linestyle=line_style,
    )
    pr_means = np.interp(pr_fpr_means, mean_rc, mean_pr)
    # Draw and fill in lines +/- std from mean roc_curve
    std_pr = np.std(pr_means, axis=0)
    pr_upper = np.minimum(pr_means + std_pr, 1)
    pr_lower = np.maximum(pr_means - std_pr, 0)

    # TODO Does it make sense to have a CI for PR?
    # ax_precision_recall.fill_between(mean_pr, pr_lower,
    #                           pr_upper, color='blue', alpha=.35,
    #                           label=r'$\pm$ 1 std. dev.')

    ax_precision_recall.set_xlabel("Recall")
    ax_precision_recall.set_ylabel("Precision")
    # ax_precision_recall.set_title('Mean Precision Recall Curve')
    ax_precision_recall.legend(loc="lower left")

    return ax_precision_recall


def plot_hit_rate_curves(
    hit_rate_info_array,
    ax_hit_rate,
    mean_line_color="blue",
    line_color="blue",
    line_style=None,
    model_name=None,
):
    # so each sample stays the same size
    n = min(
        [
            len(curve_info_one_run.get("hit_rate"))
            for curve_info_one_run in hit_rate_info_array
        ]
    )
    hr_curve_arrays = []
    for curve_info_one_run in hit_rate_info_array:
        hit_rate_data = curve_info_one_run.get("hit_rate")[:n]

        hr_curve_arrays.append(hit_rate_data)

        ax_hit_rate.plot(
            np.arange(0, n, 1),
            hit_rate_data,
            color=line_color,
            lw=1,
            linestyle=line_style,
        )
        # plt.xlim(1,1450);
        ax_hit_rate.set_ylim(0, 1.1)
        ax_hit_rate.set_title(f"Average Hit Rate Curve")
        ax_hit_rate.set_xlabel("Number of digs")
        ax_hit_rate.set_ylabel("Fraction  of lead")

    mean_hr_curve = np.mean(hr_curve_arrays, axis=0)
    ax_hit_rate.plot(
        np.arange(0, len(mean_hr_curve), 1),
        mean_hr_curve,
        color=mean_line_color,
        linestyle=line_style,
        linewidth=6,
        label=model_name,
    )

    # Draw and fill in lines +/- std from mean roc_curve
    std_hr = np.std(hr_curve_arrays, axis=0)
    hrs_upper = np.minimum(mean_hr_curve + std_hr, 1)
    hrs_lower = np.maximum(mean_hr_curve - std_hr, 0)
    ax_hit_rate.fill_between(
        np.arange(0, len(mean_hr_curve), 1),
        hrs_lower,
        hrs_upper,
        color=mean_line_color,
        alpha=0.15,
        # label=r"$\pm$ 1 std. dev.",
    )

    ax_hit_rate.legend()
    return ax_hit_rate


def plot_feature_importance(feature_data, ax_feature_importance):
    feature_data_zipped = [
        dict(zip(feature_data.feature_names[i], feature_data.importance_score[i]))
        for i in feature_data.index
    ]
    features_dataframe = pd.DataFrame(feature_data_zipped)
    ax_feature_importance.set_title("Feature Importance")
    sns.boxplot(data=features_dataframe, ax=ax_feature_importance)
    plt.setp(
        ax_feature_importance.get_xticklabels(),
        rotation=-30,
        ha="left",
        rotation_mode="anchor",
    )
    for i, artist in enumerate(ax_feature_importance.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        artist.set_color(BC_COLORS["primary-blue"])
        artist.set_edgecolor(BC_COLORS["secondary-blue"])
        for j in range(i * 6, i * 6 + 6):
            line = ax_feature_importance.lines[j]
            line.set_color(BC_COLORS["secondary-blue"])  # use #ffdc7d for baseline
            line.set_mfc(BC_COLORS["secondary-blue"])
            line.set_mec(BC_COLORS["secondary-blue"])


def plot_evaluation_results(
    evaluation_results,
    plot_title,
    selected_metrics=["auc", "precision", "recall", "accuracy"],
):
    fig = plt.figure(figsize=(28, 12), constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.5)

    fig.suptitle(plot_title, fontsize=16)

    fig.suptitle(plot_title, fontsize=16)

    # top row left to right:
    # feature importance, roc curve, learning curve, precision recall

    # bottom row (one long boxplot)
    # summary metrics
    (
        ax_roc_curve,
        ax_precision_recall,
        ax_hit_rate,
        ax_summary_metrics,
        ax_feature_importance,
    ) = (
        fig.add_subplot(gs1[0, 0]),
        fig.add_subplot(gs1[0, 1]),
        fig.add_subplot(gs1[0, 2]),
        fig.add_subplot(gs1[1, :]),
        fig.add_subplot(gs1[2, :]),
    )

    # TODO replace with plot_metrics_boxplot function
    cv_model_metrics = evaluation_results["cv_model_metrics"]
    # Plot box plot metrics
    metrics_dataframe = pd.DataFrame(cv_model_metrics)
    sns.boxplot(data=metrics_dataframe[selected_metrics], ax=ax_summary_metrics)
    for i, artist in enumerate(ax_summary_metrics.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        artist.set_color(BC_COLORS["primary-blue"])
        artist.set_edgecolor(BC_COLORS["secondary-blue"])
        for j in range(i * 6, i * 6 + 6):
            line = ax_summary_metrics.lines[j]
            line.set_color(BC_COLORS["secondary-blue"])  # use #ffdc7d for baseline
            line.set_mfc(BC_COLORS["secondary-blue"])
            line.set_mec(BC_COLORS["secondary-blue"])

    # PLot Feature Importances
    feature_data = pd.DataFrame(evaluation_results["feature_data"])
    if feature_data.importance_score[0] is not None:
        plot_feature_importance(feature_data, ax_feature_importance)

    # Plot curve data
    curve_data = evaluation_results["curve_data"]
    # Plot ROC curves
    roc_curve_info_array = [
        cv_fold_metrics.get("roc_curve") for cv_fold_metrics in curve_data
    ]
    plot_roc_curves(
        roc_curve_info_array,
        ax_roc_curve,
        ci_color=BC_COLORS["secondary-blue"],
        mean_line_color=BC_COLORS["secondary-blue"],
        line_color=BC_COLORS["secondary-blue"],
        model_name=evaluation_results.get("model_name"),
    )

    # Plot precision recall curves
    precision_recall_info_array = [
        cv_fold_metrics.get("precision_recall") for cv_fold_metrics in curve_data
    ]
    plot_precision_recall_curves(
        precision_recall_info_array,
        ax_precision_recall,
        mean_line_color=BC_COLORS["secondary-blue"],
        line_color=BC_COLORS["secondary-blue"],
        model_name=evaluation_results.get("model_name"),
    )

    # Plot Hit Rate curve
    hit_rate_info_array = [
        cv_fold_metrics.get("hit_rate") for cv_fold_metrics in curve_data
    ]
    plot_hit_rate_curves(
        hit_rate_info_array,
        ax_hit_rate,
        mean_line_color=BC_COLORS["secondary-blue"],
        line_color=BC_COLORS["secondary-blue"],
        model_name=evaluation_results.get("model_name"),
    )

    return fig


def plot_model_comparison(
    a_model_results,
    b_model_results,
    plot_title=None,
    selected_metrics=["auc", "precision", "recall", "accuracy"],
):

    plot_title = (
        plot_title
        or f"{a_model_results.get('model_name')}-vs-{b_model_results.get('model_name')}"
    )

    fig = plt.figure(figsize=(28, 12), constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.5)

    fig.suptitle(plot_title, fontsize=16)

    # top row left to right:
    # feature importance, roc curve, learning curve, precision recall

    # bottom row (one long boxplot)
    # summary metrics
    (
        ax_roc_curve,
        ax_precision_recall,
        ax_hit_rate,
        ax_summary_metrics,
        ax_feature_importance,
    ) = (
        fig.add_subplot(gs1[0, 0]),
        fig.add_subplot(gs1[0, 1]),
        fig.add_subplot(gs1[0, 2]),
        fig.add_subplot(gs1[1, :]),
        fig.add_subplot(gs1[2, :]),
    )

    # Plot box plot metrics
    a_model_cv_model_metrics = a_model_results["cv_model_metrics"]
    a_model_metrics_dataframe = pd.DataFrame(a_model_cv_model_metrics)[
        selected_metrics
    ].assign(model_name=a_model_results.get("model_name"))
    b_model_cv_model_metrics = b_model_results["cv_model_metrics"]
    b_model_metrics_dataframe = pd.DataFrame(b_model_cv_model_metrics)[
        selected_metrics
    ].assign(model_name=b_model_results.get("model_name"))

    model_metrics_dataframe = pd.concat(
        [a_model_metrics_dataframe, b_model_metrics_dataframe],
        axis=0,
    )

    plot_metrics_comparison_boxplot(
        model_metrics_dataframe,
        boxplot_ax=ax_summary_metrics,
    )

    # Plot curve data
    a_model_curve_data = a_model_results["curve_data"]
    # Plot ROC curves
    a_model_roc_curve_info_array = [
        cv_fold_metrics.get("roc_curve") for cv_fold_metrics in a_model_curve_data
    ]
    plot_roc_curves(
        a_model_roc_curve_info_array,
        ax_roc_curve,
        line_color=BC_COLORS["secondary-blue"],
        ci_color=BC_COLORS["secondary-blue"],
        mean_line_color=BC_COLORS["secondary-blue"],
        line_style="-",
        model_name=a_model_results.get("model_name"),
    )

    # Plot curve data
    b_model_curve_data = b_model_results["curve_data"]
    # Plot ROC curves
    b_model_roc_curve_info_array = [
        cv_fold_metrics.get("roc_curve") for cv_fold_metrics in b_model_curve_data
    ]
    plot_roc_curves(
        b_model_roc_curve_info_array,
        ax_roc_curve,
        line_color=BC_COLORS["primary-gold"],
        ci_color=BC_COLORS["primary-gold"],
        mean_line_color=BC_COLORS["primary-gold"],
        line_style="--",
        model_name=b_model_results.get("model_name"),
    )

    # Plot precision recall curves
    a_model_precision_recall_info_array = [
        cv_fold_metrics.get("precision_recall")
        for cv_fold_metrics in a_model_curve_data
    ]
    plot_precision_recall_curves(
        a_model_precision_recall_info_array,
        ax_precision_recall,
        line_color=BC_COLORS["secondary-blue"],
        mean_line_color=BC_COLORS["secondary-blue"],
        line_style="-",
        model_name=a_model_results.get("model_name"),
    )

    # Plot precision recall curves
    b_model_precision_recall_info_array = [
        cv_fold_metrics.get("precision_recall")
        for cv_fold_metrics in b_model_curve_data
    ]
    plot_precision_recall_curves(
        b_model_precision_recall_info_array,
        ax_precision_recall,
        line_color=BC_COLORS["primary-gold"],
        mean_line_color=BC_COLORS["primary-gold"],
        line_style="--",
        model_name=b_model_results.get("model_name"),
    )

    # Plot Hit rate curves
    a_hit_rate_info_array = [
        cv_fold_metrics.get("hit_rate") for cv_fold_metrics in a_model_curve_data
    ]
    plot_hit_rate_curves(
        a_hit_rate_info_array,
        ax_hit_rate,
        mean_line_color=BC_COLORS["secondary-blue"],
        line_color=BC_COLORS["secondary-blue"],
        model_name=a_model_results.get("model_name"),
    )

    b_hit_rate_info_array = [
        cv_fold_metrics.get("hit_rate") for cv_fold_metrics in b_model_curve_data
    ]
    plot_hit_rate_curves(
        b_hit_rate_info_array,
        ax_hit_rate,
        mean_line_color=BC_COLORS["primary-gold"],
        line_color=BC_COLORS["primary-gold"],
        line_style="--",
        model_name=a_model_results.get("model_name"),
    )

    return fig


def plot_metrics_comparison_boxplot(model_metrics_dataframe, boxplot_ax):
    colors = [BC_COLORS["secondary-blue"], BC_COLORS["primary-gold"]]
    model_names = model_metrics_dataframe.model_name.unique()
    p = dict(zip(model_names, colors))

    sns.boxplot(
        data=model_metrics_dataframe.melt(id_vars=["model_name"]),
        x="variable",
        y="value",
        hue="model_name",
        ax=boxplot_ax,
        palette=p,
    )

    # for i, artist in enumerate(boxplot_ax.artists):
    #     # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    #     artist.set_color("blue")
    #     artist.set_edgecolor("cyan")
    #     for j in range(i * 6, i * 6 + 6):
    #         line = boxplot_ax.lines[j]
    #         line.set_color("cyan")  # use #ffdc7d for baseline
    #         line.set_mfc("cyan")
    #         line.set_mec("cyan")

    return boxplot_ax


def plot_metrics_boxplot(
    model_metrics_dataframe, boxplot_ax, primary_color="blue", secondary_color="cyan"
):

    # Plot box plot metrics
    sns.boxplot(data=model_metrics_dataframe, ax=boxplot_ax)
    for i, artist in enumerate(boxplot_ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        artist.set_color(primary_color)
        artist.set_edgecolor(secondary_color)
        for j in range(i * 6, i * 6 + 6):
            line = boxplot_ax.lines[j]
            line.set_color(secondary_color)
            line.set_mfc(secondary_color)
            line.set_mec(secondary_color)

    return boxplot_ax


def gen_metrics_figure(metrics_df, plot_title):
    plt.figure()
    plot = sns.boxplot(data=metrics_df.drop(["n"], axis="columns"))
    plot = sns.stripplot(data=metrics_df.drop(["n"], axis="columns"), color=".2")
    plt.xticks(rotation=75)
    plot.set_title(plot_title)
    plt.tight_layout()
    return plot.get_figure()


# TODO Double check this -- curves look weird
def plot_learning_curve(lc_data, plot_title):
    plt.figure()

    # TODO This is brittle and subject to dictionary ordering
    train_set_sizes, train_scores, test_scores = lc_data.values()

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_set_sizes, train_mean, "--", color="#111111", label="Training score")
    plt.plot(
        train_set_sizes, test_mean, color="#111111", label="Cross-validation score"
    )

    for i in range(test_scores.shape[1]):
        plt.plot(train_set_sizes, test_scores[:,i], linestyle="dotted", alpha=0.3)

    # Draw bands
    plt.fill_between(
        train_set_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD"
    )
    plt.fill_between(
        train_set_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD"
    )

    # Create plot
    plt.title(plot_title)
    plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
    plt.tight_layout()
    return plt.gcf()


def plot_calibration_curves(cv_model_metrics, plot_title):
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    calib_curves_x, calib_curves_y, hists, bins = [], [], [], []

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for index, mod in enumerate(cv_model_metrics):
        fraction_of_positives, mean_predicted_value = mod["calibration"]
        clf_score, prob_pos = mod["brier_score"], mod["prob_pos"]

        calib_curves_y.append(fraction_of_positives)
        calib_curves_x.append(mean_predicted_value)

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            alpha=0.15,
            label="Fold %s Brier score: (%1.3f)" % (index, clf_score),
        )
        h = ax2.hist(
            prob_pos,
            range=(0, 1),
            bins=10,
            label="Fold %s" % (index),
            histtype="step",
            alpha=0.2,
            lw=2,
        )
        hists.append(h[0])
        bins = h[1]

    ax1.errorbar(
        x=np.array(calib_curves_x).mean(axis=0),
        y=np.array(calib_curves_y).mean(axis=0),
        yerr=np.array(calib_curves_y).std(axis=0),
        fmt="o-",
        label="%s fold average performance" % (len(cv_model_metrics)),
        c="k",
    )

    hists = np.array(hists)
    cols = [hists[:, i] for i in range(hists.shape[1])]

    ax2.boxplot(
        cols,
        # getting middle of bins for boxplot position
        positions=(bins[1:] + bins[:-1]) / 2,
        widths=0.025,
    )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(plot_title)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", ncol=2)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))

    plt.tight_layout()
    return plt.gcf()


def plot_figure_suite(
    client_name, model_base_name, dataset_hash, cv_constructor, metrics_payload
):
    cv_model_metrics = metrics_payload["cv_model_metrics"]
    lc_data = metrics_payload["lc_data"]

    # CV metrics boxplots
    plot_title = (
        f"{model_base_name} with {cv_constructor} on {dataset_hash} for {client_name}"
    )
    fig = gen_metrics_figure(
        pd.DataFrame.from_dict(cv_model_metrics),
        plot_title,
    )
    save_figure(fig, "figures", make_uri_friendly_string(plot_title))

    # Learning curve
    plot_title = f"{model_base_name} with {cv_constructor} on {dataset_hash} learning curve for {client_name}"
    fig = plot_learning_curve(lc_data, plot_title)
    save_figure(fig, "figures", make_uri_friendly_string(plot_title))

    plot_title = f"report for {model_base_name} with {cv_constructor} on {dataset_hash} for {client_name}"
    fig = plot_evaluation_results(metrics_payload, plot_title)
    save_figure(fig, "figures", make_uri_friendly_string(plot_title))

    # Calibration curves
    plot_title = f"{model_base_name} with {cv_constructor} on {dataset_hash} calibration curve for {client_name}"
    try:
        fig = plot_calibration_curves(cv_model_metrics, plot_title)
        save_figure(fig, "figures", make_uri_friendly_string(plot_title))
    except ValueError as e:
        log.error(
            "Error plotting calibration curves."
            " This is probably related to SLIRP issue #122."
            " Proceeding without calibration curve."
        )

    return plt.gcf()


def plot_predictions_data(geodataframe, model_base_name, dataset_hash):

    geodataframe["Sample Site"] = (geodataframe["y_score"] >= 0.8).astype(float)
    for i in geodataframe[geodataframe.y_score.isna()].index:
        geodataframe.at[i, "Sample Site"] = np.nan
    for i in geodataframe.dropna(subset=["has_lead"]).index:
        if geodataframe["has_lead"][i] == True:
            geodataframe.at[i, "Material"] = "Known Lead"
        else:
            geodataframe.at[i, "Material"] = "Known Non-Lead"
    for i in geodataframe.dropna(subset=["Sample Site"]).index:
        if geodataframe["Sample Site"][i] == True:
            geodataframe.at[i, "Material"] = "Likely Lead"
        else:
            if geodataframe["y_hat"][i] == 1:
                geodataframe.at[i, "Material"] = "Less Likely Lead"
            else:
                geodataframe.at[i, "Material"] = "Likely Non-Lead"
    geodataframe["Material"] = geodataframe.Material.fillna("Unknown")

    ax = geodataframe.plot(
        column="Material",
        figsize=(16, 9),
        categorical=True,
        legend=True,
        legend_kwds={"loc": "lower right"},
    )
    fig = ax.get_figure()

    plot_title = model_base_name + " predictions on " + dataset_hash
    ax.set_title(plot_title)

    save_figure(fig, "figures", make_uri_friendly_string(plot_title))

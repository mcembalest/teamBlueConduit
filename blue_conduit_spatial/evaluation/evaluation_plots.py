import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from blue_conduit_spatial.evaluation import generate_calibration_curve, generate_hit_rate_curve, generate_hit_rate_curve_by_partition

#### HELPER FUNCTIONS #####
def sample_num_to_prob(hit_rates, pred_probs, n=500):
    """Converts an ordered hit rate to a prediction probabilities
    
    Args:
        hit_rates (array-like): numpy array of hit rates, ordered by prediction probability
        pred_probs (array-like): array of prediction probabilities, corresponding
                            to the hit rates
        n (int): number of points to put into line; optional, default is 500
    
    Returns:
        hit_rate_list: list of hit rates corresponding to each binary class threshold
        thresholds: ordered thresholds for plotting
    """
    # Create thresholds & reverse to get [1., 0.9, 0.8, ...] like pattern
    thresholds = np.linspace(0, 1, n)[::-1]

    hit_rate_output = []
    for i in range(1, len(thresholds)):
        # Find first index of prediction array where threshold is met
        # Keeps only the prediction probabilities w/prob. > threshold
        try:
            idx = np.where(pred_probs > thresholds[i])[0][-1]
        except:
            idx = 0
        hit_rate_output.append(hit_rates[idx])
    
    return np.array(hit_rate_output), thresholds[1:]

##### PLOTTING FUNCTIONS

def plot_hit_rate_curve(y_true, 
                        y_pred, 
                        plot_probs=True, 
                        labels=None, 
                        max_perf=False, 
                        order_by_prob=False, 
                        figsize=(10,6), 
                        savefig=False, 
                        figname=None, 
                        figdir=None, 
                        mode='all',
                        parcel_df=None,
                        pid_list=None,
                        threshold_init=None, 
                        title_suffix=None,
                        min_hit_rate=0.0,
                        custom_cmap=None,
                        show_as_pct=False,
                        return_obj=False,
                        **kwargs
                        ):
    """Generates plot of hit rate curve with highly flexible set of options.

    Important Design Choices:
        1. # of models: Can pass multiple models as a list to `y_pred` and 
                        concurrent number of labels as list to `labels`.

        2. Investigation 'mode': Can investigate plots with mode = 'all' which
                        will dig parcels with highest P(lead). If `mode` == 'partition'
                        curve will be done by partition & sweeps. Controlled via
                        `threshold_init` parameter.

        3. Plot prediction probability?: Can plot only hit rate curve or also prediction
                        probability alongside for each model as dotted line.

        4. Draw a line with maximum possible performance: this mode (controlled via) `max_perf`
                        allows user to locate a line where the 'kink' would occur with
                        perfect (i.e. all lead then all non-lead) behavior.

        5. Order by probability: Rather than draw in space, `order_by_prob` can be used to draw
                        the hit rate curves in prob. space rather than sample space.

        6. Show x-axis as the sample # or the % of test set.
        

    Args:
        y_true: Ground truth outcomes for dangerous/not dangerous
        y_pred: Either a list of model prediction probabilities or 
                a single model outcomes
        plot_probs: Boolean for whether to include prediction
                    probabilities in model
        labels: Labels to include if y_pred is a list
        max_perf (bool): indicates whether to plot the 'maximum performance'
                        or the kink in the curve where a perfect model would
                        decrease performance
        figsize: Follows matplotlib fig size convention of (h, w)
        savefig: Boolean indicating whether to save figure
        figname: Figure title
        figdir: Directory to save figure.
        mode: One of "all" or "partition". Controls whether parcels are 
              investigated in unrestricted way or partition-by-partition, 
              ordering by the highest priority partitions.
        parcel_df: Required in "partition" mode. Will guide the partitions that 
              each parcel belongs to for aggregating partition investigation decisions.
        pid_list : Required in "partition" mode. Guides list of which PIDs in `parcel_df` 
              are actually in the test set.
        threshold_init: Required in "partition" mode. Sets the initial threshold for 
             digging. Typically considered to be 0.9 in baseline of 
             `generate_hit_rate_curve_by_partition`.
        title_suffix: Suffix to be included in plot title.
        min_hit_rate: Min hit rate to show, translates to lower x-limit if not 
              showing entire x-axis. 
        custom_cmap: Customizable matplotlib cmap; must be a list and will 
              otherwise select the 'Dark2' palette.
        show_as_pct: Option to show all plots as a percent of the test set.
        return_obj: Will return matplotlib objects to be manipulated for production

    Returns:
        fig: If return_obj, a matplotlib figure; otherwise returns nothing
    """
    if mode not in ['all', 'partition']:
        raise ValueError(f"Mode must be one of 'all' or 'partition'.")

    fig = plt.figure(figsize=figsize)

    # Handle non-list instances of the predictions
    if not isinstance(y_pred, list):
        y_pred = [y_pred]

    hit_rate_list = []
    pred_prob_list = []
    for mod in y_pred:
        if mode == 'all':
            hit_rates, pred_probs = generate_hit_rate_curve(y_true, mod)
        elif mode == 'partition':
            if sum([x is None for x in [parcel_df, pid_list, threshold_init]]) > 0:
                raise ValueError(
                    "In partition mode, none of parcel_df, index_list, or threshold_init can  be none")
            hit_rates, pred_probs, _ = generate_hit_rate_curve_by_partition(parcel_df, 
                pid_list, 
                y_true, 
                mod, 
                threshold_init, 
                **kwargs)

        # If ordering by probability; set up thresholds
        if order_by_prob == True:
            title = "Cumulative Hit Rate Curve by Classification Threshold"
            if not title_suffix is None:
                 title = title + f'\n{title_suffix}'
            hit_rates, xs = sample_num_to_prob(hit_rates, pred_probs, n=500)
            plt.xlabel(f"Classification threshold")
            plt.title(title)
            plt.xlim(1, 0)
        else:
            test_N = len(hit_rates)
            xs = np.arange(test_N)

            # if show as percent, can divide through & mult. by 100
            pos_str = 'Position'
            if show_as_pct:
                xs = (xs / test_N) * 100
                pos_str = 'Percentile'

            if mode == 'partition':
                plt.xlabel(f'{pos_str} in sample, ordered by partition')
            else:
                plt.xlabel(f'{pos_str} in sample, order by pred. prob.')
            title = "Cumulative Hit Rate Curve by Prediction Probability"
            if not title_suffix is None:
                 title = title + f'\n{title_suffix}'
            plt.title(title)
        hit_rate_list.append(hit_rates)
        pred_prob_list.append(pred_probs)

    if labels == None:
        labels = ['Hit rate curve']
    
    # Only set cmap when a custom cmap is not passed
    # useful when fixing colors for clarity between models
    if custom_cmap is None:
        cmap = cm.get_cmap('Dark2').colors
    else:
        cmap = custom_cmap
    
    for i, hr in enumerate(hit_rate_list):
        plt.plot(xs, hr, label=labels[i], color=cmap[i])
    
    if plot_probs == True:
        for i, pp in enumerate(pred_prob_list):
            plt.plot(pp, ls='--', label=f'Pred. probs. ({labels[i]})', color=cmap[i])
        plt.ylabel(f"Cumulative Hit Rate / Prediction Probability")
    else:
        plt.ylabel(f"Cumulative Hit Rate")
    
    if max_perf == True:
        # Maximum performance will be the total number of parcels with lead in the
        # test set.
        tot_w_lead = y_true.sum()
        plt.axvline(tot_w_lead, ls='-.', label=f"Total w/Lead; Maximum Performance", color='k')
    
    plt.ylim(min_hit_rate,1)
    plt.legend()

    if savefig == True:
        plt.savefig(figdir + figname)
    elif return_obj == True:
        return fig 
    else:
        plt.show()

def plot_calibration_curve(y_true, y_pred, n_bins=10, labels=None, figsize=(10,6), savefig=False, figname=None, figdir=None, **kwargs):
    """Plots probability calibration curve for various number of model results

    Args:
        y_true: Ground truth outcomes for dangerous/not dangerous
        y_pred: Either a list of model prediction probabilities or 
                a single model outcomes
        n_bins: Number of bins to discretize for each model
        labels: Labels to include if y_pred is a list
        figsize: Follows `matplotlib` fig size convention of (h, w)
        savefig: Boolean indicating whether to save figure
        figname: Figure title
        figdir: Directory to save figure.
    """
    fig = plt.figure(figsize=figsize)
    if labels == None:
        labels = ['Model']
    elif type(labels) == str:
        labels = [labels]

    if not isinstance(y_pred, list):
        pred_list = [y_pred]
    else:
        pred_list = y_pred
    
    cmap = cm.get_cmap('Dark2').colors
    for i, mod in enumerate(pred_list):
        true_curve, pred_curve = generate_calibration_curve(y_true, mod, n_bins=10, **kwargs)
        plt.plot(pred_curve, true_curve, color=cmap[i], label=labels[i])
    plt.axline([0,0], slope=1, ls='--', color='k', label='Perfect calibration')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Mean predicted probability success')
    plt.ylabel('Fraction digs successful')
    plt.legend()

    if savefig == True:
        plt.savefig(figdir + figname)
    else:
        plt.show()
        
def plot_pr_curve(y_true, y_pred, labels=None,
                  figsize=(10,6), dpi=90, savefig=False, figname=None, figdir=None):
    """Generates precisio-recall curve plot for single or multiple models

    Args:
        y_true: Ground truth outcomes for dangerous/not dangerous
        y_pred: Either a list of model prediction probabilities or 
                a single model outcomes
        labels: Labels to include if y_pred is a list
        dpi: matplotlib dpi
        figsize: Follows matplotlib fig size convention of (h, w)
        savefig: Boolean indicating whether to save figure
        figname: Figure filepath / file title (not nec. title of plot)
        figdir: Directory to save figure.

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Handle non-list instances of the predictions
    if not isinstance(y_pred, list):
        y_pred = [y_pred]

    prec_list = []
    recall_list = []
    auc_list = []
    
    for mod in y_pred:
        mod = np.array(mod).reshape(-1)
        prec_i, recall_i, thres_i = precision_recall_curve(y_true, mod)
        prec_list.append(prec_i)
        recall_list.append(recall_i)
        
        auc_i = auc(recall_i, prec_i)
        auc_list.append(auc_i)

    if labels == None:
        labels = [f'Precision-recall {i}' for i in range(len(prec_list))]
    cmap = cm.get_cmap('Dark2').colors
    
    for prec_, recall_, auc_, label_, color_ in zip(prec_list, recall_list, auc_list, labels, cmap):
        label_ = label_ + f' AUC:{auc_:.2f}'
        plt.plot(recall_, prec_, label=label_, color=color_)
    
    plt.title('Precision-Recall curve for lead predictions')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    if savefig == True:
        plt.savefig(figdir + figname)
    else:
        plt.show()
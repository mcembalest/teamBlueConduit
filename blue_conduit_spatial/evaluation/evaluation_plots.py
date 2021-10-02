import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np
from blue_conduit_spatial.evaluation import generate_calibration_curve, generate_hit_rate_curve

#### HELPER FUNCTIONS #####
def order_by_prob(hit_rates, pred_probs, n=500):
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
        idx = np.where(pred_probs > thresholds[i])[0][-1]
        hit_rate_output.append(hit_rates[idx])
    
    return np.array(hit_rate_output), thresholds[1:]


def plot_hit_rate_curve(y_true, y_pred, plot_probs=True, labels=None, max_perf=False, 
                        order_by_prob=False, figsize=(10,6), savefig=False, figname=None, figdir=None):
    """Generates plot of hit rate curve with three potential modes:
        (1) Single model, no prediction probabilities;
        (2) Multiple models, no prediction probabilities;
        (3) Single model, prediction probabilities;
        (4) Multiple models, prediction probabilities

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

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize)

    if isinstance(y_pred, list):
        hit_rate_list = []
        pred_prob_list = []
        for mod in y_pred:
            hit_rates, pred_probs = generate_hit_rate_curve(y_true, mod)
            hit_rate_list.append(hit_rates)
            pred_prob_list.append(pred_probs)
    else:
        hit_rates, pred_probs = generate_hit_rate_curve(y_true, y_pred)
        hit_rate_list = [hit_rates]
        pred_prob_list = [pred_probs]
    
    if labels == None:
        labels = ['Hit rate curve']
    cmap = cm.get_cmap('Dark2').colors
    
    for i, hr in enumerate(hit_rate_list):
        plt.plot(hr, label=labels[i], color=cmap[i])
    
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
    

    plt.ylim(0,1)
    plt.xlabel('Position in sample, order by pred. prob.')
    plt.title("Cumulative Hit Rate Curve by Prediction Probability")
    plt.legend()

    if savefig == True:
        plt.savefig(figdir + figname)
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
        figsize: Follows matplotlib figsize directions
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
import numpy as np
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import precision_score, classification_report
from sklearn.calibration import calibration_curve

def pred2d_to_1d(arr):
    """Helper method for creating 1D array to enforce
    same array dimensions for methods"""

    if len(arr.shape) == 1:
        return arr
    else:
        return arr[:,1]


def roc_auc_score(y_true, y_pred):
    """Returns the ROC-AUC Score for some y_true and y_test. Can pass predicted
    labels as probabilities. Will make determination based on number of columns in
    y_pred dataset.

    Args:
        y_true: Ground truth labels. Should be binary.
        y_pred: Prediction probabilities or class labels
    
    Returns:
        ROC-AUC score for the metrics
    """
    yhat = pred2d_to_1d(y_pred)
    return ras(y_true, y_pred[:,1])

def hit_rate(y_true, y_pred, threshold=0.5):
    """Returns the 'hit rate'. This is the total predicted positive that actually
    were (i.e. precision for the positive class)
    
    Args:
        y_true: Ground truth labels. Should be binary.
        y_pred: Prediction probabilities or class labels
        threshold: Float threshold for considering a prediction
              to be 'dangerous' 
    
    Returns
        hit_rate: The total number of successful digs (i.e. precision over the
                  positive class"""
    try:
        if y_pred.shape[1] > 2:
            raise ValueError('Only two classes can be passed')
        else:
            y_hat = y_pred[:,1] > threshold

    except:
        y_hat = y_pred > threshold
    
    output_dict = classification_report(y_true, y_hat, output_dict=True)
    hit_rate = output_dict['1']['precision']
    return hit_rate

def generate_hit_rate_curve(y_true, y_pred):
    """Generates cumulative hit rate curve on a sample by sample basis for entire set.
    Note: this assumes that data passed is ordered correctly, as w/sk-learn

    Args:
        y_true: Ground truth labels. Should be binary.
        y_pred: Prediction probabilities or class labels

    Returns:
        hit_rates: Cumulative hit rates for ordered test set
        pred_probs: Predicted probabilities for ordered test set
    """
    # Handle passing of 1-D and 2-D arrays
    yhat = pred2d_to_1d(y_pred)

    y_comb = np.stack([y_true, yhat], axis=1)

    # Sort data to produce cumulative curves
    y_comb_sorted = y_comb[np.argsort(y_comb, axis=0)[:,1][::-1]]

    # Find hit rates by calc cumulative sum of successes and 
    # dividing by position in array
    hit_rates = np.cumsum(y_comb_sorted[:,0])/(np.arange(y_comb_sorted.shape[0]) + 1)
    return hit_rates, y_comb_sorted[:,1]

def generate_calibration_curve(y_true, y_pred, n_bins=10, **kwargs):
    """Mask (with error handling) for sk-learn calibration curve

    Args:
        y_true: Ground truth labels. Should be binary.
        y_pred: Prediction probabilities or class labels
        n_bins: number of bins to discretize
    
    Returns:
        true_curve: Bucket mean true positives
        pred_curve: Bucket mean predicted positives
    """
    yhat = pred2d_to_1d(y_pred)

    return calibration_curve(y_true, yhat, n_bins=n_bins, **kwargs)


if __name__ == '__main__':
    raw_data = np.load('../../data/predictions/baseline_preds.npz', allow_pickle=True)
    y_true = raw_data['ytrue'].astype(int)
    y_hat = raw_data['yhat'].astype(float)

    ra_sk = generate_calibration_curve(y_true, y_hat)
    print(ra_sk)

    


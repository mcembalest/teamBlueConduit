import numpy as np
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import precision_score, classification_report
from sklearn.calibration import calibration_curve
import geopandas as gpd

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

def generate_hit_rate_curve_by_partition(parcel_df, 
                                        index_list, 
                                        y_true, 
                                        y_pred, 
                                        threshold_init, 
                                        threshold_increment=0.1, 
                                        min_digs=5, 
                                        min_digs_increment=5
                                        ):
    """Generates a hit rate curve where parcels are investigated partition. 
    
    Made to be inter-operable with the `gizmo` partitioning. Agnostic to shape 
    of the partition, but must pass a parcel df which contains a parcel ID. Can 
    replicate initial analyses by passing an ordered DF with 
    'partition_ID' = 'PRECINCT'.
    
    Args:
        parcel_df: Ordered dataframe passing reference to pid and partition_ID
        index_list: Array-like list of indices in the i.e. test set (as used 
          to divide train/test data)
        y_true: True values (should be in order of `index_list`). Sk-learn API.
        y_pred: Prediction probabilities at the parcel-level; SK-Learn API. 
        threshold_init: Initial threshold for decision-making.
        threshold_increment: Float to decrease threshold by at each iteration.
        min_digs: Minimum number of digs required in a partition to visit.

    Returns:
        hit_rates: Cumulative hit rates for ordered test set
        pred_probs: Predicted probabilities for ordered test set
    """
    # Create temporary dataframe containing only necessary features for filtering
    # process (initial parcel_df has many features; done to save space)
    df = parcel_df.iloc[index_list]
    df['pred_prob'] = y_pred.copy()
    df['true_val'] = y_true.copy()
    df = df[['partition_ID', 'pred_prob', 'true_val']]
    threshold = threshold_init

    # Create running list of true values and prediction probs
    true_label_list = []
    pred_prob_list = []

    # Also track indices that have been dug
    part_dug_idx_list = []

    # Continue to move through DF while some parcels have not been added
    while len(df) > 0:
        # Find total parcels in each partition which have > threshold pred prob
        df['dig'] = (df['pred_prob'] > threshold).astype(int)
        part_sort = df.groupby(
            'partition_ID').sum()['dig'].sort_values(ascending=False)
        
        for i, x in enumerate(part_sort):
            # If fewer than `min_digs` expected hits in a given partition, 
            # break out of loop (sorting enforces that partition n+1 will
            # have <= x expected hits)
            if x < min_digs:
                break
            else:
                part_idx = part_sort.index[i]
                # Within a partition, order by the prediction probability
                # Simulates prioritizing specific parcels within a partition, but could
                # also choose some sort of random / noisy optimization as well
                to_dig = df[(df['partition_ID']==part_idx) & (df['dig']==1)].sort_values(
                    'pred_prob', ascending=False)

                true_label_list.extend(to_dig['true_val'].values)
                part_dug_idx_list.extend(to_dig.index.values)
                pred_prob_list.extend(to_dig['pred_prob'].values)

        # Remove all parcels which have been excavated from the
        df = df.iloc[~df.index.isin(part_dug_idx_list)]

        # If possible, decrease threshold
        if threshold - threshold_increment > 0:
            threshold -= threshold_increment
        else:
            min_digs -= min_digs_increment
            threshold = threshold_init

    return np.cumsum(true_label_list)/(np.arange(len(true_label_list))+1), pred_prob_list

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

    



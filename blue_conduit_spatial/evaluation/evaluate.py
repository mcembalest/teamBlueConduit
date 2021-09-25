import numpy as np
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import precision_score, classification_report

def roc_auc_score(y_true, y_pred):
    """Returns the ROC-AUC Score for some y_true and y_test. Can pass predicted
    labels as probabilities. Will make determination based on number of columns in
    y_pred dataset.

    y_true: Ground truth labels. Should be binary.
    y_pred: Prediction probabilities or class labels
    """
    try:
        if y_pred.shape[1] > 2:
            raise ValueError('Only two classes can be passed')
        else:
            return ras(y_true, y_pred[:,1])
    except:
        return ras(y_true, y_pred)


def hit_rate(y_true, y_pred, threshold=0.5):
    """Returns the 'hit rate'. This is the total predicted positive that actually
    were (i.e. precision for the positive class)
    
    y_true: Ground truth labels. Should be binary.
    y_pred: Prediction probabilities or class labels
    threshold: Float threshold for considering a prediction
              to be 'dangerous' """
    try:
        if y_pred.shape[1] > 2:
            raise ValueError('Only two classes can be passed')
        else:
            y_hat = y_pred[:,1] > threshold

    except:
        y_hat = y_pred
    
    output_dict = classification_report(y_true, y_hat, output_dict=True)
    return output_dict['1']['precision']

def generate_hit_rate_curve(y_true, y_pred):
    """Generates cumulative hit rate curve on a sample by sample basis for entire set.
    Note: this assumes that data passed is ordered correctly, as w/sk-learn

    y_true: Ground truth labels. Should be binary.
    y_pred: Prediction probabilities or class labels
    """
    y_comb = 0



if __name__ == '__main__':
    raw_data = np.load('../../data/predictions/baseline_preds.npz', allow_pickle=True)
    y_true = raw_data['ytrue'].astype(int)
    y_hat = raw_data['yhat'].astype(float)

    ra_sk = hit_rate(y_true, y_hat.argmax(axis=1))
    print(ra_sk)

    



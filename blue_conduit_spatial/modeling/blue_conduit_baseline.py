import numpy as np 
import pandas as pd 
import numpy as np
import geopandas as gpd
from blue_conduit_spatial.utilities import load_datasets
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import sys

def fit_baseline_model(x_train, y_train):
    mod = xgb.XGBClassifier()
    mod.fit(x_train, y_train, eval_metric='logloss')
    return mod

def create_data_from_idx(xdata, ydata, train_idx_list, test_idx_list):
    xtrain = xdata.iloc[train_idx_list]
    ytrain = ydata.iloc[train_idx_list]['dangerous']

    xtest = xdata.iloc[test_idx_list]
    ytest = ydata.iloc[test_idx_list]['dangerous']

    return xtrain, xtest, ytrain, ytest

def run_baselines(load_dir='../../data/processed', save_dir='../../data/predictions'):
    Xdata, Ydata, pid, train_idx, test_idx = load_datasets(load_dir)

    # Find list of all train percentages available
    train_pcts = list(train_idx.keys())
    # Calculate number of splits per percentage
    n_split = len(train_idx[train_pcts[0]])

    pred_probs_train_dict = {}
    pred_probs_test_dict = {}
    for i, t in enumerate(train_pcts):
        train_probs = []
        test_probs = []
        for s in range(n_split):
            # Create data specific to the model being fit
            selected_train_idx = train_idx[t][s]
            selected_test_idx = test_idx[t][s]
            xtrain, xtest, ytrain, ytest = create_data_from_idx(Xdata, Ydata, selected_train_idx, selected_test_idx)

            mod = fit_baseline_model(xtrain, ytrain)
            
            train_preds = mod.predict_proba(xtrain)
            train_probs.append(train_preds[:,1])
            
            test_preds = mod.predict_proba(xtest)
            test_probs.append(test_preds[:,1])

        pred_probs_train_dict[t] = np.array(train_probs, dtype='object')
        pred_probs_test_dict[t] = np.array(test_probs, dtype='object')

    np.savez(f'{save_dir}/pred_probs_train.npz', **pred_probs_train_dict)
    np.savez(f'{save_dir}/pred_probs_test.npz', **pred_probs_test_dict)

if __name__ == '__main__':
    run_baselines()

    

    



import numpy as np 
import pandas as pd 
import numpy as np
import geopandas as gpd
from blue_conduit_spatial.utilities import load_datasets
import xgboost as xgb
import sys

def fit_baseline_model(x_train, y_train):
    mod = xgb.XGBClassifier()
    mod.fit(x_train, y_train, eval_metric='logloss')
    return mod

def create_data_from_idx(xdata, ydata, train_pid_list, test_pid_list):
    xtrain = xdata.loc[train_pid_list]
    ytrain = ydata.loc[train_pid_list]['dangerous']

    xtest = xdata.loc[test_pid_list]
    ytest = ydata.loc[test_pid_list]['dangerous']

    return xtrain, xtest, ytrain, ytest

def run_baselines(load_dir='../../data/processed', save_dir='../../data/predictions', verbose=True):
    Xdata, Ydata, location, train_pid_all, test_pid_all, partitions_builder  = load_datasets(load_dir)

    # Find list of all train percentages available
    train_pcts = list(train_pid_all.keys())

    # Find resolutions
    train_resolutions = list(train_pid_all[train_pcts[0]].keys())

    # Calculate number of splits per percentage
    n_split = len(train_pid_all[train_pcts[0]][train_resolutions[0]])

    pred_probs_train_dict = {}
    pred_probs_test_dict = {}
    for i, t in enumerate(train_pcts):
        if verbose:
            print(f"Working on train percentage {t}")
        train_probs_pct = {}
        test_probs_pct = {}
        for j, res in enumerate(train_resolutions):
            train_probs_res = []
            test_probs_res = []
            for s in range(n_split):
                # Create data specific to the model being fit
                selected_train_idx = train_pid_all[t][res][s]
                selected_test_idx = test_pid_all[t][res][s]
                xtrain, xtest, ytrain, ytest = create_data_from_idx(Xdata, Ydata, selected_train_idx, selected_test_idx)

                mod = fit_baseline_model(xtrain, ytrain)
                
                train_preds = mod.predict_proba(xtrain)
                train_probs_res.append(train_preds[:,1])
                
                test_preds = mod.predict_proba(xtest)
                test_probs_res.append(test_preds[:,1])
            if verbose:
                print(f"Finished with resolution {res}")
            
            train_probs_pct[res] = np.array(train_probs_res, dtype='object')
            test_probs_pct[res] = np.array(test_probs_res, dtype='object')

        pred_probs_train_dict[t] = train_probs_pct
        pred_probs_test_dict[t] = test_probs_pct

    np.savez(f'{save_dir}/pred_probs_train.npz', **pred_probs_train_dict)
    np.savez(f'{save_dir}/pred_probs_test.npz', **pred_probs_test_dict)

if __name__ == '__main__':
    run_baselines(verbose=True)

    

    



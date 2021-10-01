from datetime import datetime
import json
import geopandas as gpd
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from .metadata import cols_metadata_dict

def jared_preprocessing(sl_df, cols_metadata):
    drop_cols = cols_metadata['drop_cols']
    dummy_cols = cols_metadata['dummy_cols']
    target_cols = cols_metadata['target_cols']

    data = sl_df.drop(drop_cols, axis = 1)

    # Only keep labelled data
    data = data[~pd.isnull(data.dangerous)].reset_index(drop=True)

    # Drop everything except target from training data
    Xdata = data.drop(['pid', 'sl_private_type', 'sl_public_type', 'dangerous'], axis = 1)

    # Build target.  Each 'dangerous' is True when sl_private_type OR sl_public_type contain lead.
    Ydata = data[['sl_private_type', 'sl_public_type', 'dangerous']]

    # Fill missing data
    Xdata = Xdata.fillna(-1)

    # Create dummies from categorical columns
    Xdata = pd.get_dummies(Xdata, columns=dummy_cols)

    # Groups for spatial cross validation
    groups = Xdata['PRECINCT']
    Xdata = Xdata.drop('PRECINCT', axis=1)
    
    return Xdata, Ydata, groups

def split_data(Xdata, Ydata, groups, n_splits=3, train_size=.75, random_state=42):
    gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)

    for train_idx, test_idx in gss.split(Xdata, Ydata, groups):
        train_index = train_idx
        test_index = test_idx
        break

    Xtrain = Xdata.loc[train_index]
    Xtest = Xdata.loc[test_index]
    Ytrain = Ydata.loc[train_index.tolist()]
    Ytest = Ydata.loc[test_index.tolist()]
    
    return Xtrain, Xtest, Ytrain, Ytest

def build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size=.75, random_state=42):
    cols_metadata = cols_metadata_dict(save_dir)
    col_name_dictionary = cols_metadata['col_name_dictionary']
    
    sl_df = gpd.read_file(data_raw_path)
    sl_df = sl_df.rename(col_name_dictionary, axis=1)
    Xdata, Ydata, groups = jared_preprocessing(sl_df, cols_metadata)
    Xtrain, Xtest, Ytrain, Ytest = split_data(Xdata, Ydata, groups, n_splits, train_size, random_state)
    
    if not save_dir is None:
        os.makedirs(save_dir, exist_ok=True) 
        Xtrain_path = f'{save_dir}/Xtrain.csv'
        Xtest_path = f'{save_dir}/Xtest.csv'
        Ytrain_path = f'{save_dir}/Ytrain.csv'
        Ytest_path = f'{save_dir}/Ytest.csv'    
    
        for df, path in zip([Xtrain, Xtest, Ytrain, Ytest],[Xtrain_path, Xtest_path, Ytrain_path, Ytest_path]):
            df.to_csv(path, index=False)
            
    return Xtrain, Xtest, Ytrain, Ytest

def load_datasets(load_dir):
    Xtrain_path = f'{load_dir}/Xtrain.csv'
    Xtest_path = f'{load_dir}/Xtest.csv'
    Ytrain_path = f'{load_dir}/Ytrain.csv'
    Ytest_path = f'{load_dir}/Ytest.csv'    
    
    Xtrain = pd.read_csv(Xtrain_path)
    Xtest = pd.read_csv(Xtest_path)
    Ytrain = pd.read_csv(Ytrain_path)
    Ytest = pd.read_csv(Ytest_path)
    
    return Xtrain, Xtest, Ytrain, Ytest
    
if __name__=='__main__':
    Xtrain, Xtest, Ytrain, Ytest = build_datasets(data_raw_path, save_dir=save_dir)
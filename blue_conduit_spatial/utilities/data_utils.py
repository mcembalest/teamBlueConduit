from datetime import datetime
import json
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from metadata import cols_metadata_dict

def blue_conduit_preprocessing(sl_df, cols_metadata):
    drop_cols = cols_metadata['drop_cols']
    dummy_cols = cols_metadata['dummy_cols']
    target_cols = cols_metadata['target_cols']

    data = sl_df.drop(drop_cols, axis = 1)

    # Only keep labelled data
    data = data[~pd.isnull(sl_df['Longitude'])]
    data = data[~pd.isnull(data.dangerous)].reset_index(drop=True)
    
    # Keep track of pid
    pid = data[['pid', 'Latitude', 'Longitude', 'geometry']]
    
    # Drop everything except target from training data
    Xdata = data.drop(['pid', 'sl_private_type', 'sl_public_type', 'dangerous', 'Latitude', 'Longitude', 'geometry'], axis = 1)

    # Build target.  Each 'dangerous' is True when sl_private_type OR sl_public_type contain lead.
    Ydata =  data[target_cols]#data[['sl_private_type', 'sl_public_type', 'dangerous']]

    # Fill missing data
    Xdata = Xdata.fillna(-1)

    # Create dummies from categorical columns
    Xdata = pd.get_dummies(Xdata, columns=dummy_cols)

    # Groups for spatial cross validation
    groups = Xdata['PRECINCT']
    Xdata = Xdata.drop('PRECINCT', axis=1)
    
    return Xdata, Ydata, groups, pid
    
def split_index(Xdata, Ydata, groups, n_splits=5, train_size_list=None, random_state=42):
    '''
    Split in `n_splits` train and test indices based on `PRECINT` groups, 
    for multiple `train_size` in `train_size_list`.
    '''
    
    assert isinstance(train_size_list, (list, tuple, np.ndarray))
    assert isinstance(train_size_list[0], float)
    
    train_idx_data = {}
    test_idx_data = {}

    for train_size in train_size_list:
        gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
        train_idx, test_idx = list(zip(*gss.split(Xdata, Ydata, groups)))
        train_idx = np.array(train_idx, dtype=object)
        test_idx = np.array(test_idx, dtype=object)

        train_idx_data[f'{train_size}'] = train_idx
        test_idx_data[f'{train_size}'] = test_idx
        
    return train_idx_data, test_idx_data

def build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, random_state=42):
    cols_metadata = cols_metadata_dict(save_dir)
    col_name_dictionary = cols_metadata['col_name_dictionary']
    
    # Default list of `train_size`.
    if train_size_list is None:
        train_size_list = np.round(np.linspace(0,1, 11)[1:-1], 1)
    
    sl_df = gpd.read_file(data_raw_path)
    sl_df = sl_df.rename(col_name_dictionary, axis=1)
    Xdata, Ydata, groups, pid = blue_conduit_preprocessing(sl_df, cols_metadata)
    train_idx, test_idx = split_index(Xdata, Ydata, groups, n_splits, train_size_list, random_state)
    
    if not save_dir is None:
        os.makedirs(save_dir, exist_ok=True) 
        
        Xdata_path = f'{save_dir}/Xdata.csv'
        Ydata_path = f'{save_dir}/Ydata.csv'
        train_idx_path = f'{save_dir}/train_index.npz'
        test_idx_path = f'{save_dir}/test_index.npz'
        pid_path = f'{save_dir}/pid.csv'
    
        for df, path in zip([Xdata, Ydata, pid],[Xdata_path, Ydata_path, pid_path]):
            df.to_csv(path, index=False)
            
        np.savez(train_idx_path, **train_idx)
        np.savez(test_idx_path, **test_idx)
            
    return Xdata, Ydata, pid, train_idx, test_idx

def load_datasets(load_dir):
    Xdata_path = f'{load_dir}/Xdata.csv'
    Ydata_path = f'{load_dir}/Ydata.csv'
    train_idx_path = f'{load_dir}/train_index.npz'
    test_idx_path = f'{load_dir}/test_index.npz'
    pid_path = f'{load_dir}/pid.csv'
    
    Xdata = pd.read_csv(Xdata_path)
    Ydata = pd.read_csv(Ydata_path)
    pid = pd.read_csv(pid_path)
    train_idx = np.load(train_idx_path, allow_pickle=True)
    test_idx = np.load(test_idx_path, allow_pickle=True)
    
    return Xdata, Ydata, pid, train_idx, test_idx

if __name__ == '__main__':
    data_dir = '../../data'
    data_raw_path = f'{data_dir}/raw/flint_sl_materials/'
    save_dir = f'{data_dir}/processed'
    build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, random_state=42)
    
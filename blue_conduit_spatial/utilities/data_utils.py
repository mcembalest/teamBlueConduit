from datetime import datetime
import json
import geopandas as gpd
#import pickle
import pickle5 as pickle
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from gizmo import spatial_partitions
from gizmo.spatial_partitions import partitions
from .metadata import cols_metadata_dict
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def get_partitions_builder(data):
    data['parcel_id'] = data.pid
    data['has_lead'] = data.dangerous
    data['year_built'] = data['Year Built']
    data['parcel_acres'] = data['Parcel Acres']

    partitions_builder = partitions.PartitionsBuilder(
        parcel_gdf=data,
        parcel_id_col='parcel_id',
        target_col='has_lead',
        copy_all_cols=True
    )
    
    data.drop(['parcel_id','has_lead','year_built','parcel_acres'], axis=1, inplace=True)
    
    return partitions_builder

def blue_conduit_preprocessing(sl_df, cols_metadata):
    drop_cols = cols_metadata['drop_cols']
    dummy_cols = cols_metadata['dummy_cols']
    target_cols = cols_metadata['target_cols']

    data = sl_df.drop(drop_cols, axis = 1)

    # Only keep labelled data
    data = data[~pd.isnull(sl_df['Longitude'])]
    data = data[~pd.isnull(data.dangerous)].reset_index(drop=True)
    
    # Build partitions
    partitions_builder = get_partitions_builder(data)
    
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
    
    # Formerly groups for spatial cross validation, now done through `partitions_builder`
    Xdata = Xdata.drop('PRECINCT', axis=1)
    
    return Xdata, Ydata, partitions_builder, pid

def get_partition(partitions_builder, num_cells_across, n_splits, random_state, plot_splits=False, test_size=0.2):
    hexagons = partitions_builder.Partition(partition_type='hexagon', num_cells_across=num_cells_across)
    hex_splitter = hexagons.cv_splitter(n_splits=n_splits, 
                                        strategy='ShuffleSplit', 
                                        plot=plot_splits, 
                                        random_state=random_state,
                                        test_size=test_size
                                       )
    train_idx, test_idx = list(zip(*hex_splitter.split(output_level='parcel')))
    train_idx = np.array(train_idx, dtype=object)
    test_idx = np.array(test_idx, dtype=object)
    
    return train_idx, test_idx
    
def split_index(Xdata, 
                Ydata, 
                partitions_builder, 
                pid, 
                n_splits=5, 
                train_size_list=None, 
                cells_across_list=None, 
                random_state=42,
                plot_splits=False
               ):
    '''
    Split in `n_splits` train and test indices for multiple `train_size` in `train_size_list`
    and `num_cells_across` in `cells_across_list`.
    '''
    
    # Default list of `train_size`.
    if train_size_list is None:
        train_size_list = np.round(np.linspace(0,1, 8)[1:-1], 1)
    
    # Default list of `num_cells_across`
    # This indicates the resolution of the hexagons
    # A big num_cells_across will divide the map in more hexagons
    if cells_across_list is None:
        cells_across_list = np.logspace(1, 2.86, base=5, num=5).astype(int)
        
    print(cells_across_list)
    assert isinstance(train_size_list, (list, tuple, np.ndarray))
    assert isinstance(train_size_list[0], float)
    assert max(train_size_list)<=1
    assert min(train_size_list)>=0
    
    assert isinstance(cells_across_list, (list, tuple, np.ndarray))
    assert isinstance(cells_across_list[0], (np.int64, int))
    
    train_idx_data = dict([(f'ts_{train_size_}',{}) for train_size_ in train_size_list])
    test_idx_data = dict([(f'ts_{train_size_}',{}) for train_size_ in train_size_list])

    for train_size in tqdm(train_size_list):
        for num_cells_across in tqdm(cells_across_list):
            test_size = 1-train_size
            train_idx, test_idx = get_partition(partitions_builder, num_cells_across, n_splits, random_state, plot_splits, test_size)
            train_idx_data[f'ts_{train_size}'][f'res_{num_cells_across}'] = train_idx
            test_idx_data[f'ts_{train_size}'][f'res_{num_cells_across}'] = test_idx
        
    return train_idx_data, test_idx_data

def build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, cells_across_list=None, 
                   random_state=42, plot_splits=False):
    cols_metadata = cols_metadata_dict(save_dir)
    col_name_dictionary = cols_metadata['col_name_dictionary']
    
    sl_df = gpd.read_file(data_raw_path)
    sl_df = sl_df.rename(col_name_dictionary, axis=1)
    Xdata, Ydata, partitions_builder, pid = blue_conduit_preprocessing(sl_df, cols_metadata)
    train_idx, test_idx = split_index(Xdata, 
                                      Ydata, 
                                      partitions_builder, 
                                      pid, 
                                      n_splits, 
                                      train_size_list, 
                                      cells_across_list,
                                      random_state,
                                      plot_splits
                                     )
    
    if not save_dir is None:
        os.makedirs(save_dir, exist_ok=True) 
        
        Xdata_path = f'{save_dir}/Xdata.csv'
        Ydata_path = f'{save_dir}/Ydata.csv'
        train_idx_path = f'{save_dir}/train_index.npz'
        test_idx_path = f'{save_dir}/test_index.npz'
        pid_path = f'{save_dir}/pid.gpkg'
        builder_path = f'{save_dir}/partitions_builder.pk'
        
        # Save Xdata and Ydata as regular csv
        for df, path in zip([Xdata, Ydata],[Xdata_path, Ydata_path]):
            df.to_csv(path, index=False)
        
        # Save train and test index as npz
        np.savez(train_idx_path, **train_idx)
        np.savez(test_idx_path, **test_idx)
        
        # Save pid with geometry as 
        pid.to_file(pid_path, driver="GPKG")
        
        # Save the partition builder to recreate partitions
        with open(builder_path, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(partitions_builder, outp, pickle.HIGHEST_PROTOCOL)
            
    return Xdata, Ydata, pid, train_idx, test_idx, partitions_builder

def load_datasets(load_dir):
    Xdata_path = f'{load_dir}/Xdata.csv'
    Ydata_path = f'{load_dir}/Ydata.csv'
    train_idx_path = f'{load_dir}/train_index.npz'
    test_idx_path = f'{load_dir}/test_index.npz'
    pid_path = f'{load_dir}/pid.gpkg'
    builder_path = f'{load_dir}/partitions_builder.pk'
    
    def format_npz_dict(dict_):
        dict_ = dict(dict_)
        dict_ = dict([(k,v.item()) for k,v in dict_.items()])
        return dict_
    
    Xdata = pd.read_csv(Xdata_path)
    Ydata = pd.read_csv(Ydata_path)
    pid = gpd.read_file(pid_path) 
    train_idx = format_npz_dict(np.load(train_idx_path, allow_pickle=True))
    test_idx = format_npz_dict(np.load(test_idx_path, allow_pickle=True))
    
    with open(builder_path, 'rb') as f:  # Overwrites any existing file.
        partitions_builder = pickle.load(f)
    
    return Xdata, Ydata, pid, train_idx, test_idx, partitions_builder

if __name__ == '__main__':
    data_dir = '../../data'
    data_raw_path = f'{data_dir}/raw/flint_sl_materials/'
    save_dir = f'{data_dir}/processed'
    Xdata, Ydata, pid, train_idx, test_idx, partitions_builder = build_datasets(data_raw_path, save_dir=save_dir)
    
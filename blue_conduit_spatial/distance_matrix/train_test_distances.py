import numpy as np
import pickle



def create_train_test_dist_matrices(load_dir):
    
    _, _, Ytrain, Ytest = load_datasets(load_dir)
    
    # load distance data
    dis_path = '../data/road_distances.npz'
    distances = np.load(dis_path, allow_pickle=True)
    dis_data = distances['road_distances']
    dis_data[dis_data==(1.00e+05)]=0.0 #assign invalid 1e5 values to be 0
    dis_key_map = distances['idx2pid'].item()
    dis_key_map_inv = dict(zip(dis_key_map.values(), dis_key_map.keys()))
    
    # local function to use
    def get_parcel_index(_parcel):
        try:
            return dis_key_map_inv[_parcel]
        except KeyError:
            pass
    
    # build distance matrices
    train_indices = [get_parcel_index(_parcel) for _parcel in Ytrain.pid.values if get_parcel_index(_parcel) is not None]
    test_indices = [get_parcel_index(_parcel) for _parcel in Ytest.pid.values if get_parcel_index(_parcel) is not None]
    train_distances = dis_data[train_indices,:][:,train_indices]
    test_distances = dis_data[test_indices,:][:,test_indices]
    
    # save arrays
    with open('../data/processed/train_indices.npy', 'wb') as f1:
        np.save(f1, train_indices)   
    with open('../data/processed/test_indices.npy', 'wb') as f2:
        np.save(f2, test_indices)
    with open('../data/processed/distances/train_distances.npy', 'wb') as f3:
        np.save(f3, train_distances)
    with open('../data/processed/distances/test_distances.npy', 'wb') as f4:
        np.save(f4, test_distances)
        
def load_train_test_dist_matrices():
    train_i = np.load('../data/processed/train_indices.npy')
    test_i = np.load('../data/processed/test_indices.npy')
    train_d = np.load('../data/processed/distances/train_distances.npy')
    test_d = np.load('../data/processed/distances/test_distances.npy')
    return train_i, test_i, train_d, test_d
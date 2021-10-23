"""Module that wraps the various spatial diffusion models tested. Relies on graph-based
methods using NetworkX. Model is graph agnostic. That is, diffusion will work for parcel or
partition-based methods"""

import matplotlib.pyplot as plt 
import matplotlib as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

import numpy as np 
import pandas as pd 

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix

import networkx as nx 

# Set random seed
np.random.seed(297)

class ServiceLineDiffusion:
    """Note: graph should be a (N, N) numpy array
    
    Args:
        graph (nd.array): (N, N) array
        train_indices (array-like): sequence of indices in graph corresponding
            to the training set
        test_indices (array-like): sequence of indices in graph corresponding
            to the test set
        lat_long_df (pd.DataFrame): if plotting, this will be used to localize
            points on the graph"""
    def __init__(self, graph, train_indices, test_indices, Ytrain, Ytest, Ytest_pred, lam=0.5, lat_long_df=None):
        self.graph = graph
        self.iter_ct = 1
        self.lam = lam

        # Assign overall training and test set indices
        self.train_indices = train_indices
        self.test_indices = test_indices

        # Set up training predictions and baseline prediction probabilities
        self.Ytrain = Ytrain
        self.Ytest = Ytest
        self.Ytest_pred = Ytest_pred
        self.curr_test_pred = self.Ytest_pred.copy() # Initialize 'current' prediction as baseline test
        self.curr_train_pred = self.Ytrain.copy() # Initialize 'current' prediction as baseline test

        self.lat_long_df = lat_long_df

    def fit(self, n_iter=1, neighbor_fn=None, neighbor_params=None, distance_function=None, verbose=False):

        if (neighbor_params is None) or (neighbor_fn is None):
            self.neighbor_params = {'graph': self.graph, 'K': 5}
            self.neighbor_fn = ServiceLineDiffusion.graph_Kneighbors
        else:
            self.neighbor_params = neighbor_params
            self.neighbor_fn = neighbor_fn

        if distance_function is None:
            self.distance_function = ServiceLineDiffusion.diffusion_distance_weights
        else:
            self.distance_function = distance_function
        
        # To reduce compute time, calculate & store nearest neighbors
        self.neighbor_distances, self.neighbor_idx = self.neighbor_fn(**self.neighbor_params)
        self.neighbor_weights = self.distance_function(self.neighbor_distances)

        # Initialize the lead values & find weighted average of neighbor lead values
        lead_vals = np.array([self._get_lead_value(idx) for idx in range(self.graph.shape[0])]).flatten()
        if verbose:
          print(f"{log_loss(self.Ytest, lead_vals[self.test_indices]):0.2f}")

        for i in range(n_iter):
            lead_vals = self.diffusion_step(lead_vals)

            if verbose:
              print(f"{log_loss(self.Ytest, lead_vals[self.test_indices]):0.2f}")

            # Set current predictions
            self.curr_test_pred = lead_vals[self.test_indices]
            self.curr_train_pred = lead_vals[self.train_indices]
        
        return lead_vals
    
    def predict_proba(self, X, mode=None):
        """Returns probabilities after lead values. Because this is a non-standard "model" and does not
        truly take a new X value, it uses the shape of X to determine whether to return the training
        or test predictions
        
        Args:
            - X (pd.DataFrame, np.array): Array of training or test points, used only for its shape
            - mode (str): One of None, 'train', 'test'. If none, will make decision based on X shape
        
        Returns
            - probs (array): Array of probability of (no lead, lead) in a (N, 2) array to be compliant
        """
        if mode == 'train':
            return np.stack([1-self.curr_train_pred, self.curr_train_pred], axis=1)
        elif mode == 'test':
            return np.stack([1-self.curr_test_pred, self.curr_test_pred], axis=1)
        
        # Utilize shape if not
        if X.shape[0] == len(self.curr_test_pred):
            return self.predict_proba(X, mode='test')
        elif X.shape[0] == len(self.curr_train_pred):
            return self.predict_proba(X, mode='train')
        else:
            raise AttributeError(f'X passed is not of same shape as either training or test predictions.')

    def diffusion_step(self, lead_vals):
        """Takes a single diffusion step. Separated for clean code practices"""
        weighted_avg_neighbor_lead = np.average(lead_vals[self.neighbor_idx], axis=1, weights = self.neighbor_weights)
        lead_vals = np.array([self._update_node(node_val, node_idx, weighted_avg_neighbor_lead, self.lam) for node_idx, node_val in enumerate(lead_vals)])
        lead_vals = lead_vals.flatten()
        return lead_vals
    
    def reset_graph(self):
        """Helper method for removing graph when dealing with memory-intensive methods"""
        self.graph = []

    def _update_node(self, node_val, node_idx, weighted_avg_neighbor_lead, lam=0.5):
        """Defines update step for a single node. Can be overwritten, must return between [0,1]"""
        #if node_val in [0., 1.]:
        #    return node_val
        #else:
        #    return lam * node_val + (1 - lam) * weighted_avg_neighbor_lead[node_idx]
        return lam * node_val + (1 - lam) * weighted_avg_neighbor_lead[node_idx]
        
    @staticmethod
    def graph_Kneighbors(graph, K):
        """Gets the nearest neighbors for each node in the graph.
        
        Note: This replaces 0 with 1e7 because sklearn NearestNeighbors treats 0 as
        a valid (close) distance. Since this is a non-connection in our graph, then 
        want it to be a very small weight
        
        Args:
            K (int): Number of neighbors
        Returns:
            neighbor_dist, neighbor_ind : arrays indicating the nearest neighbor and """
        nn = NearestNeighbors(n_neighbors=K, metric='precomputed')
        neighbors_graph = graph.copy()
        neighbors_graph[neighbors_graph==0] = 1e7
        nn.fit(neighbors_graph)
        del neighbors_graph
        return nn.kneighbors()
    
    @staticmethod
    def diffusion_distance_weights(distances):
        """Returns an arbitrary float distance as the inverse for weighting.
        Also adjusts 0 to be 1 such that a distance of zero maps to a weight of 1"""
        return 1/(1 + distances)

    def _get_lat_long(self, idx):
        """Returns latitude and longitude for a given index"""
        return np.array(self.lat_long_df.iloc[idx])

    def _idx2trainidx(self, idx):
        """Returns the specific index within the train data for idx"""
        return np.where(self.train_indices == idx)[0]
    
    def _idx2testidx(self, idx):
        """Returns the specific index within the test data for idx"""
        return np.where(self.test_indices == idx)[0]
    
    def _get_lead_value(self, idx):
        """If the parcel is in the training data, then lead value will be set to
        ground truth. Else, will return the prediction probability"""
        if idx in self.train_indices:
            return self.Ytrain[self._idx2trainidx(idx)]
        else:
            return self.curr_test_pred[self._idx2testidx(idx)]


    

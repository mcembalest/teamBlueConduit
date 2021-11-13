"""Module that wraps the GraphSAGE model. Relies on graph-based
library Stellargraph (which itself depends on NetworkX)."""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import scipy
from scipy.sparse import csr_matrix

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

########### set random state
np.random.seed(297)

def get_xgboost_prob(train_pred, test_pred, train_mask, test_mask, train_index, test_index, idx):
  '''
  Args:
    train_pred: 
    test_pred:
    train_mask:
    test_mask:
    train_index:
    test_index:
    idx: the index according to the original 0,...,26856 list

  Returns:
    res: float, the XGBoost probability of lead at this home (either from train_pred or test_pred)
  '''

  try:
    res = None
    if train_mask[idx]:
      res =  train_pred[np.where(train_index==idx)][0]
    elif test_mask[idx]:
      res = test_pred[np.where(test_index==idx)][0]
    return res
  except BaseException as err:
    print(train_pred.shape, test_pred.shape, train_mask.shape, test_mask_shape)
    raise

def get_label(Ydata, idx):
  '''
  Args:
    Ydata:
    idx: the index according to the original 0,...,26856 list

  Returns:
    the label (0 or 1) representing whether there is dangerous lead in the home
  '''

  return Ydata.dangerous.values[idx]
    
def build_feature_matrix(Xdata_scaled, subgraph_idx, train_pred, test_pred, train_mask, test_mask, train_index, test_index, features = None):
  #'features' is a list of named features for use in feature matrix. Example: ['Year Built', 'Land Value', 'Lot Size']
  #Set 'features' to 'All' to get full feature set
  #Leave features as 'None' to just use XGBoost probabilities

  '''
  Args:
    Xdata_scaled:
    subgraph_idx:
    train_pred:
    test_pred:
    train_mask:
    test_mask:
    train_index:
    test_index:
    features:

  Returns:
    flint_graphSAGE:
    flint_graphSAGE_train_gen:
    flint_graphSAGE_test_gen:
  '''

  #If no additional features, just use XGBoost probabilities
  if features is None:
    return np.array([[get_xgboost_prob(train_pred, test_pred, train_mask, test_mask, train_index, test_index, idx)] for idx in subgraph_idx])

  #Append all features
  elif features == 'All':
    return np.hstack([np.array([get_xgboost_prob(train_pred, test_pred, train_mask, test_mask, train_index, test_index, idx) for idx in subgraph_idx]).reshape(-1,1), Xdata_scaled.iloc[subgraph_idx].values])

  #Append selected features
  else:
    return np.hstack([np.array([get_xgboost_prob(train_pred, test_pred, train_mask, test_mask, train_index, test_index, idx) for idx in subgraph_idx]).reshape(-1,1), Xdata_scaled[features].iloc[subgraph_idx].values])


def get_trained_graphSAGE_for_experiment(train_size, hex_size, s, Xdata,Ydata, pid, train_idx, test_idx, train_pred_all, test_pred_all, partitions_builder, verbose=False):
  '''
  Args:
    train_size:
    hex_size:
    s:
    Xdata:
    Ydata:
    pid:
    train_idx:
    test_idx:
    train_pred_all:
    test_pred_all:
    partitions_builder:
    verbose:

  Returns:
    flint_graphSAGE:
    flint_graphSAGE_train_gen:
    flint_graphSAGE_test_gen:
  '''

  data = select_data(Xdata,Ydata, pid, train_idx, test_idx, train_pred_all, test_pred_all, partitions_builder, train_size=train_size, n_hexagons=hex_size, split=s)
  train_index = data['train_index']
  test_index = data['test_index']
  Xtrain = data['Xtrain']
  Xtest = data['Xtest']
  Ytrain = data['Ytrain']
  Ytest = data['Ytest']
  train_pred = data['train_pred']
  test_pred = data['test_pred']
  hexagons = data['hexagons']

  ## build subgraph
  subgraph_idx = np.array([i for i in range(len(Ydata)) if (i in train_index or i in test_index)])
  n_subgraph = len(subgraph_idx)
  subgraph = graph[:,subgraph_idx][subgraph_idx,:]

  return get_trained_graphsage(Xdata, Ydata, subgraph, subgraph_idx, train_index, test_index, train_pred, test_pred, verbose=verbose)
  
def get_trained_graphsage(Xdata, Ydata, subgraph, subgraph_idx, train_index, test_index, train_pred, test_pred,verbose=False):
  '''
  Args:
  	Xdata:
  	Ydata:
  	subgraph:
    subgraph_idx:
    train_index:
    test_index:
    train_pred:
    test_pred:
    verbose:

  Returns:
    flint_graphSAGE:
    flint_graphSAGE_train_gen:
    flint_graphSAGE_test_gen:
  '''
  
  scaler = preprocessing.StandardScaler()
  Xdata_scaled_train = scaler.fit_transform(Xdata.iloc[train_index])
  Xdata_scaled_test = scaler.transform(Xdata.iloc[test_index])

  Xdata_scaled = Xdata.copy()
  Xdata_scaled.iloc[train_index] = Xdata_scaled_train
  Xdata_scaled.iloc[test_index] = Xdata_scaled_test

  train_mask = np.array([1 if i in train_index else 0 for i in range(len(Ydata))])
  test_mask = np.array([1 if i in test_index else 0 for i in range(len(Ydata))])

  if verbose:
    print('built scaled data and masks')
  train_indices = np.array([idx for idx in subgraph_idx if train_mask[idx]]).astype('int32')
  train_labels = np.array([get_label(Ydata, idx) for idx in subgraph_idx if train_mask[idx]])
  test_indices = np.array([idx for idx in subgraph_idx if test_mask[idx]]).astype('int32')
  test_labels = np.array([get_label(Ydata, idx) for idx in subgraph_idx if test_mask[idx]])

  if verbose:
    print('built labels')
  #chosen_features = ['Year Built', 'Land Value', 'Parcel Acres', 'Residential Building Value', 'Hydrant Type_A.D.', 'Hydrant Type_Dar', 'Hydrant Type_Mueller', 'Hydrant Type_Other','Hydrant Type_T.C.']
  chosen_features = Xdata.columns

  # fts = build_feature_matrix(Xdata_scaled, subgraph_idx, train_pred, test_pred, train_mask, test_mask, features='All').astype('float32')
  fts = build_feature_matrix(Xdata_scaled, subgraph_idx, train_pred, test_pred, train_mask, test_mask, train_index, test_index, features='All').astype('float32')

  if verbose:
    print('built features')
  sparse_subgraph = csr_matrix(subgraph)
  row_idx, col_idx, vals = scipy.sparse.find(sparse_subgraph)

  # Define graph, namely an edge tensor and a node feature tensor
  edges = np.vstack([subgraph_idx[col_idx], subgraph_idx[row_idx]]).T.astype('float32')
  if verbose:
    print('built edges')

  ## stellargraph object
  flint_train_labels_series = pd.Series(data=train_labels,index=train_indices)
  flint_test_labels_series = pd.Series(data=test_labels,index=test_indices)

  flint_features = np.append(['XGBoost Prob'], chosen_features)
  flint_node_data = pd.DataFrame(
      {f : fts[:,i] for i, f in enumerate(flint_features)}, index=subgraph_idx.astype('int64')
  )
  flint_edge_data = pd.DataFrame({"source": edges[:,0].astype('int64'), "target": edges[:,1].astype('int64')})
  flint_G = sg.StellarGraph(flint_node_data, flint_edge_data)

  if verbose:
    print('built graph')

  ## GraphSAGE

  batch_size = 256
  num_samples = [15, 10]
  generator = GraphSAGENodeGenerator(flint_G, batch_size, num_samples)

  target_encoding = preprocessing.LabelBinarizer()
  train_targets = target_encoding.fit_transform(flint_train_labels_series)
  test_targets = target_encoding.transform(flint_test_labels_series)
  flint_graphSAGE_train_gen = generator.flow(flint_train_labels_series.index, train_targets)
  flint_graphSAGE_test_gen = generator.flow(flint_test_labels_series.index, test_targets)

  graphsage_model = GraphSAGE(
    layer_sizes=[64, 64], generator=generator, bias=True, dropout=0.5,
  )

  x_inp, x_out = graphsage_model.in_out_tensors()
  prediction = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)

  flint_graphSAGE = Model(inputs=x_inp, outputs=prediction)

  if verbose:
    print('built model')

  ## train

  flint_graphSAGE.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'), tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec'), tf.keras.metrics.TruePositives(name='tp'), tf.keras.metrics.TrueNegatives(name='tn'), tf.keras.metrics.FalsePositives(name='fp'), tf.keras.metrics.FalseNegatives(name='fn') ]
  )
  if verbose:
    print('starting training...')
    flint_graphSAGE.fit(
        flint_graphSAGE_train_gen,
        epochs=20,
        verbose=2,
        shuffle=True,
    )
    print('...done training')
  else:
    flint_graphSAGE.fit(
      flint_graphSAGE_train_gen,
      epochs=20,
      verbose=0,
      shuffle=True,
  )
    
  return flint_graphSAGE, flint_graphSAGE_train_gen, flint_graphSAGE_test_gen

def train_models_on_data_splits(GraphSAGE_train_preds, GraphSAGE_test_preds, train_sizes, resolutions, splits, Xdata,Ydata, pid, train_idx, test_idx, train_pred_all, test_pred_all, partitions_builder):
  '''
  Args:
    GraphSAGE_train_preds:  
    GraphSAGE_test_preds:

    train_sizes:
    splits:
    resolutions:

  Returns:
    GraphSAGE_train_preds
    GraphSAGE_test_preds
  '''
  for train_size in train_sizes:

    for hex_size in resolutions:
      train_preds= []
      test_preds =[]
      for s in splits:
        flint_graphSAGE = get_trained_graphSAGE_for_experiment(train_size, hex_size, s, Xdata,Ydata, pid, train_idx, test_idx, train_pred_all, test_pred_all, partitions_builder)

        y_pred_train_graphSAGE = flint_graphSAGE.predict(flint_graphSAGE_train_gen).flatten()
        y_pred_test_graphSAGE = flint_graphSAGE.predict(flint_graphSAGE_test_gen).flatten()

        train_preds.append(y_pred_train_graphSAGE)
        test_preds.append(y_pred_test_graphSAGE)
      
      GraphSAGE_train_preds[f'ts_{train_size}'][f'res_{hex_size}'] = np.array(train_preds)
      GraphSAGE_test_preds[f'ts_{train_size}'][f'res_{hex_size}'] = np.array(test_preds)

  return GraphSAGE_train_preds, GraphSAGE_test_preds


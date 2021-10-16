import numpy as np 
import pandas as pd 
import geopandas as gpd
from blue_conduit_spatial.utilities import load_datasets
from sklearn.model_selection import GroupShuffleSplit
import xgboost
import sys


load_dir = '../../data'
Xdata, Ydata = load_datasets(load_dir)
groups = Xdata['PRECINCT']

# Create splits
splitter = GroupShuffleSplit(n_splits=10, train_size=0.1, random_state=42)

# For each split, save out predictions on train and test sets
for train_idx, test_idx in splitter.split(Xdata, Ydata, groups):

    # Subset data
    train_data = Xdata.iloc[]

    # Fit model
    mod = xgboost.XGBClassifier()
    mod.fit()


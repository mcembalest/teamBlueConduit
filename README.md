# teamBlueConduit
capstone project for Harvard IACS AC297r

Team Members: Javiera Astudillo, Max Cembalest, Kevin Hare, and Dashiell Young-Saver

# Libraries setup

Start at root folder

## Install local libraries

```
pip install .
```

## Install requirements

```
pip install -r requirements.txt
```

## Install partner libraries

```
cd gizmo
pip install -e .
```

# Data

## Data folder structure

To reduce space locally, we have utilized a consistent structure of the data folders. Below is a brief tutorial on how to replicate the data directories. First, the following directory structure must be created:

```
...
├── data
│   ├── README.md
│   ├── predictions
│   │   ├── pred_probs_test.npz
│   │   └── pred_probs_train.npz
│   ├── processed
│   │   ├── Xdata.csv
│   │   ├── Ydata.csv
│   │   ├── cols_metadata.json
│   │   ├── haversine_distances.npz
│   │   ├── partitions_builder.pk
│   │   ├── pid.gpkg
│   │   ├── road_distances.npz
│   │   ├── test_index.npz
│   │   └── train_index.npz
│   └── raw
│       └── flint_sl_materials
│           ├── flint_sl_materials.cpg
│           ├── flint_sl_materials.dbf
│           ├── flint_sl_materials.prj
│           ├── flint_sl_materials.shp
│           └── flint_sl_materials.shx
...
```
All files can be replicated locally, though the distance matrices are > 5GB and thus were handled via Google Colab.


## Build datasets
```build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, cells_across_list=None, 
                   random_state=42, plot_splits=True)

Returns:
* Xdata: pd.DataFrame
* Ydata: pd.DataFrame
* pid: gpd.GeoDataFrame
* train_idx: dict
* test_idx: dict
* partitions_builder: gizmo.spatial_partitions.partitions.PartitionsBuilder
```              

**Example**

Note: the following code can be executed in a Jupyter Notebook, or alternatively by navigating to `blue_conduit_spatial/utilities` and running `python data_utils.py`, where the default behavior will mirror the directories below.

```
from blue_conduit_spatial.utilities import build_datasets

data_dir = '../data'
data_raw_path = f'{data_dir}/raw/flint_sl_materials/'
save_dir = f'{data_dir}/processed'

Xdata, Ydata, pid, train_idx, test_idx, partitions_builder = build_datasets(data_raw_path, save_dir=save_dir)
```
## Load datasets

```load_datasets(load_dir)```

**Example**

```
from blue_conduit_spatial.utilities import load_datasets

data_dir = '../data'
load_dir = f'{data_dir}/processed'

Xdata, Ydata, pid, train_idx, test_idx, partitions_builder = load_datasets(load_dir)

train_idx.keys()
>>> dict_keys(['ts_0.1', 'ts_0.3', 'ts_0.4', 'ts_0.6', 'ts_0.7', 'ts_0.9'])

train_idx['ts_0.1'].keys()
>>> dict_keys(['res_5', 'res_10', 'res_22', 'res_47', 'res_99'])

train_idx_['ts_0.1']['res_5']
>>> array([Int64Index([1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872,
                   ...
                   4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230],
                   dtype='int64', length=2368)                                  ,
           Int64Index([6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785, 6786, 6787,
                   ...
                   9154, 9155, 9156, 9157, 9158, 9159, 9160, 9161, 9162, 9163],
                   dtype='int64', length=2386)                                  ,
           Int64Index([20632, 20633, 20634, 20635, 20636, 20637, 20638, 20639, 20640,
                   20641,
                   ...
                   23567, 23568, 23569, 23570, 23571, 23572, 23573, 23574, 23575,
                   23576],
                   dtype='int64', length=2945)                                    ],
      dtype=object)
```

# Modeling

## Blue Conduit Baseline
To fit the Blue Conduit baseline XGBoost models, we can run the following command. This requires the directory structure above, in particular having run or downloaded the `predictions` directory.

- Navigate to `blue_conduit_spatial/modeling`
- Execute `python blue_conduit_baseline.py`

Taken together, these commands will generate the `pred_probs_train.npz` and `pred_probs_test.npz` files. These correspond exactly to the indices described in `train_index.npz` and `test_index.npz`.

# Plots

## Plot precision-recall curve

```
import numpy as np
import pandas as pd

from blue_conduit_spatial.evaluation import plot_pr_curve

data_dir = '../data'
y_train_path = f'{data_dir}/processed/Ytrain.csv'
bc_yhat_train_path = f'{data_dir}/processed/predictions/jared_train_yhat.csv'

y_train = pd.read_csv(y_train_path)['dangerous'].values
y_hat_train = pd.read_csv(bc_yhat_train_path).values[:,1]
y_hat_random = [np.random.rand(len(y_train))]

mod_list = [y_hat_train, y_hat_random]
labels = ['BC Baseline', 'Random']

plot_pr_curve(y_train, mod_list, labels=labels)
```

![pr-sample](plots/pr_sample.png)


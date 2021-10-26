# teamBlueConduit
capstone project for Harvard IACS AC297r

Team Members: Javiera Astudillo, Max Cembalest, Kevin Hare, and Dashiell Young-Saver

# Libraries setup

```pip install .```

# Data

## Data folder structure

To reduce space locally, we have utilized a consistent structure of the data folders. Below is a brief tutorial on how to replicate the data directories. First, the following directory structure must be created:

```
...
├── data
│   ├── README.md
│   ├── processed
│   │   ├── Xdata.csv
│   │   ├── Ydata.csv
│   │   ├── cols_metadata.json
│   │   ├── pid.csv
│   │   ├── test_index.npz
│   │   └── train_index.npz
|   |   └── road_distances.npz
|   |   └── haversine_distances.npz
│   ├── raw
│   │   └── flint_sl_materials
│   │       ├── flint_sl_materials.cpg
│   │       ├── flint_sl_materials.dbf
│   │       ├── flint_sl_materials.prj
│   │       ├── flint_sl_materials.shp
│   │       └── flint_sl_materials.shx
│   ├── predictions
│   │   ├── pred_probs_train.npz
│   │   ├── pred_probs_test.npz

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
```build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, random_state=42)```

**Example**

Note: the following code can be executed in a Jupyter Notebook, or alternatively by navigating to `blue_conduit_spatial/utilities` and running `python data_utils.py`, where the default behavior will mirror the directories below.

```
from blue_conduit_spatial.utilities import build_datasets, load_datasets

data_dir = '../data'
data_raw_path = f'{data_dir}/raw/flint_sl_materials/'
save_dir = f'{data_dir}/processed'

Xdata, Ydata, pid, train_idx, test_idx = build_datasets(data_raw_path, save_dir=save_dir)
```
## Load datasets

```load_datasets(load_dir)```

**Example**

```
from blue_conduit_spatial.utilities import build_datasets, load_datasets

data_dir = '../data'
load_dir = f'{data_dir}/processed'

Xdata, Ydata, pid, train_idx, test_idx = load_datasets(load_dir)

train_idx.files
>>> ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

train_idx['0.1']
>>> array([array([  537,   539,   540, ..., 24621, 24622, 24623]),
           array([   48,    51,    59, ..., 26859, 26860, 26861]),
           array([ 2893,  2912,  2919, ..., 26852, 26857, 26862])],
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


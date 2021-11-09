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

Note that in order to install the partner libraries you need a version of Python >=3.7. If you are in a virtual env with conda, you can do it like this:

```
conda update python
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
**```build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, cells_across_list=None, 
                    random_state=42, plot_splits=False)```**

```
Returns
---------------------
  Xdata: pd.DataFrame
  Ydata: pd.DataFrame
  pid: gpd.GeoDataFrame
  train_idx: dict
  test_idx: dict
  partitions_builder: gizmo.spatial_partitions.partitions.PartitionsBuilder
```

If no `train_size_list` is provided it will be replaced with:
```array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])```

Similarly, `cells_across_list` will be replaced with:
```array([ 5, 10, 22, 47, 99])```

Also, note that with these default parameters, the total time to build the datasets will be around ~386.67 s (measured on two CPU, 16Gb RAM).

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

**```load_datasets(load_dir)```**

```
Returns
---------------------
  Xdata: pd.DataFrame
  Ydata: pd.DataFrame
  pid: gpd.GeoDataFrame
  train_idx: dict
  test_idx: dict
  partitions_builder: gizmo.spatial_partitions.partitions.PartitionsBuilder
```

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

**```load_predictions(pred_dir)```**

```
Returns
---------------------
  train_preds: dict
  test_preds: dict

```

**Example**

```
from blue_conduit_spatial.utilities import load_predictions

data_dir = '../data'
pred_dir = f'{data_dir}/processed'

train_preds, test_preds = load_datasets(pred_dir)
```

# Modeling

## Blue Conduit Baseline
To fit the Blue Conduit baseline XGBoost models, we can run the following command. This requires the directory structure above, in particular having run or downloaded the `predictions` directory.

- Navigate to `blue_conduit_spatial/modeling`
- Execute `python blue_conduit_baseline.py`

Taken together, these commands will generate the `pred_probs_train.npz` and `pred_probs_test.npz` files. These correspond exactly to the indices described in `train_index.npz` and `test_index.npz`.

# Evaluation

## Generate hit rate curve
There are two primary methods for generating the hit-rate curve, and a single plotting utility for doing so. The functions live in [blue_conduit_spatial.evaluation](blue_conduit_spatial/evaluation). First, `generate_hit_rate_curve` will create a hit rate curve ordering by the prediction probability. If generating the hit rate curve by partition (i.e. group by the partition and then investigate partitions with more expected lead first), this can be done by `generate_hit_rate_curve_by_partition`. These are equivalent when `threshold_init=1.0`, `threshold_increment=1e-6` (or some other arb. small number), and `min_digs=0`. That will visit partitions exactly in the order of the highest probability (though will be substantially slower).

### Sample usage:
```python3
from blue_conduit_spatial.utilities import load_datasets, load_predictions, select_data
from blue_conduit_spatial.evaluation import generate_hit_rate_curve, generate_hit_rate_curve_by_partition, plot_hit_rate_curve, dig_stats, dig_savings

data_dir = '../data'
load_dir = f'{data_dir}/Processed'
pred_dir = f'{data_dir}/Predictions'
pid_lat_lon_path = f'{load_dir}/pid.gpkg'

# Load data for all hexagons resolutions, train sizes and splits
Xdata, Ydata, pid, train_idx, test_idx, partitions_builder = load_datasets(load_dir)
train_pred_all_bl, test_pred_all_bl = load_predictions(pred_dir, probs_prefix='baseline')
train_pred_all_diff, test_pred_all_diff = load_predictions(pred_dir, probs_prefix='diffusion')

# Filter the data for one selection of hyperparameters (hexagons resolutions, train sizes and splits)
n_hexagons = 47
train_size = 0.1
split = 0

args = {    
    'Xdata': Xdata,
    'Ydata': Ydata,
    'pid': pid,
    'train_idx': train_idx,
    'test_idx': test_idx,
    'train_size': train_size,
    'split': split,
    'return_pid': False,
    'generate_hexagons': False
}

# Data selection arguments
args_bl, args_diff = args.copy(), args.copy()

# Set data selection arguments for Baseline model
args_bl['train_pred_all'] = train_pred_all_bl
args_bl['test_pred_all'] = test_pred_all_bl

# Set data selection arguments for Diffusion model
args_diff['train_pred_all'] = train_pred_all_diff
args_diff['test_pred_all'] = test_pred_all_diff

# Select data per model
data_bl = select_data(**args_bl)
data_diff = select_data(**args_diff)

# Get city map hexagons
hexagons = partitions_builder.Partition(partition_type='hexagon', num_cells_across=n_hexagons)
parcel_gdf = hexagons.parcel_gdf

test_index = data_bl['test_index']
y_test = data_bl['Ytest']

plot_args = {
    'plot_probs': False,
    'labels':['BlueConduit Baseline', 'Random Beta(1,1)'],
    'mode':'all',
    'y_true': y_test,
    'y_pred': [data_bl['test_pred'], np.random.beta(1, 1, size=len(y_test))],
    'title_suffix': 'Test set'
}

plot_hit_rate_curve(**plot_args)
```

![hrc-baseline](plots/plot_hrc_baseline.png)


### Comparison between HRC methods
In the plot below, we demonstrate the differences in performance for the Blue Conduit baseline, for a single split / resolution / set of hyperparameters. Note that most splits / resolutions show qualitatively similar results. Increasing initial threshold improves performance over initial homes. Decreasing increment has similar parameter. Decreasing minimum num. homes improves relative performance over second half of homes.

![hrc-comparison](plots/hit_rate_curve_comparison.png)

## Generate digging statistics table

**```dig_stats(parcel_gdf, index_list, y_true, y_pred, strat_names=None, bins=15, mode='digs_lead_number', hr_args=None)```**

```
Returns
---------------------
  dig_stats_df: pd.DataFrame
```

Calculate digging statistics for each digging strategy within `y_pred` based on `mode` criteria.
Bins the data for improving the insights, following the digging order imposed by the hit rate curve
ordered by partition.

Modes:
* `digs_number`: parcels are binned in batches with equal number of diggings.
* `digs_lead_number`: parcels are binned in batches with equal number of lead diggings.

**```dig_savings(dig_stats_df, model1_str, model2_str)```**

Meant to be used when `dig_stats` is in `mode=digs_lead_number`. Comparison of lead pipe replacement cost between
`model1_str` and `model2_str` which should correspond to names in the `strat_names` passed to `dig_stats`.
Generates a new column in the `dig_stats_df` with the cost savings segmented every `N` replaced pipes.

```
Returns
---------------------
  dig_stats_df: pd.DataFrame
```

### Sample usage

```
# Set `dig_stats` arguments
parcel_gdf = parcel_gdf
index_list = data_bl['test_index']
y_true = data_bl['Ytest']
y_pred = [data_bl['test_pred'], data_diff['test_pred']]
strat_names = ['Baseline', 'Diffusion']
bins = 15

# Get digging statistics
mode = 'digs_number'
dig_stats_df = dig_stats(parcel_gdf, index_list, y_true, y_pred, strat_names=strat_names, bins=bins, mode=mode)
dig_stats_df
```
![hrc-comparison](plots/table_digs.png)

```
mode = 'digs_lead_number'
dig_stats_df = dig_stats(parcel_gdf, index_list, y_true, y_pred, strat_names=strat_names, bins=bins, mode=mode)
dig_stats_df = dig_savings(dig_stats_df, 'Baseline', 'Diffusion')
dig_stats_df.head()
```
![hrc-comparison](plots/table_digs_lead.png)

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


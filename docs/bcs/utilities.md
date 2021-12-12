# Utilities

### Table of Contents

- [Examples](#build-datasets)
- [API Reference](#API-Reference)

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

Xdata, Ydata, location, train_pid, test_pid, partitions_builder = build_datasets(data_raw_path, save_dir=save_dir)
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

Xdata, Ydata, location, train_pid, test_pid, partitions_builder = load_datasets(load_dir)

train_pid.keys()
>>> dict_keys(['ts_0.1', 'ts_0.3', 'ts_0.4', 'ts_0.6', 'ts_0.7', 'ts_0.9'])

train_pid['ts_0.1'].keys()
>>> dict_keys(['res_5', 'res_10', 'res_22', 'res_47', 'res_99'])

train_pid['ts_0.1']['res_5']
>>> array([array([4012476011, 4012476025, 4012476026, ..., 4002459029, 4002378013,
              4002459031])                                                    ,
       array([4011380043, 4011456005, 4011456012, ..., 4011131001, 4011129001,
              4011133029])                                                    ,
       array([4118455026, 4119132003, 4119132004, ..., 4130129025, 4130131006,
              4130104034])                                                    ],
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



## API Reference

- `get_partitions_builder(data)`
  Get partitions builder from `gizmo` library to build the train test splits according to the spatial cross-validation framework.

  | **Argument**         | **Type**                                   | **Status**       | **Description**                                                  |
  | -------------------- | ------------------------------------------ | ---------------- | ---------------------------------------------------------------- |
  | `data`               | `geopandas.geodataframe.GeoDataFrame`      | required         | Geopandas dataframe with parcels location andd lead information  |

  | **Return**           | **Type**                                                   | **Description**                                              |
  | -------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
  | `partitions_builder` | `gizmo.spatial_partitions.partitions.partitions_builder`   | Partitions builder object from gizmo library that creates spatial cross-validation splits for different hexgaons resolutions. |

- `blue_conduit_preprocessing(sl_df, cols_metadata)`

- `get_partition(partitions_builder, num_cells_across, n_splits, random_state, plot_splits, test_size=0.2)`

- `split_index(Xdata, Ydata, partitions_builder, n_splits=5, train_size_list=None, cells_across_list=None, random_state=42, plot_splits=False)`

- `build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, cells_across_list=None, random_state=42, plot_splits=False)`

- `load_datasets(load_dir)`

- `load_predictions(pred_dir, probs_prefix='baseline')`

  Loads specific types of predictions with `.npz` endings.

  Note that files should be saved as `'{prefix}_pred_probs_train.npz` and `'{prefix}pred_probs_test.npz'`. For the baseline models this will be simply `pred_probs_train.npz`, but for e.g. diffusion it will be `diffusion_pred_probs_train.npz`.

  | **Argument**   | **Type** | **Status**           | **Description**                                              |
  | -------------- | -------- | -------------------- | ------------------------------------------------------------ |
  | `pred_dir`     | `str`    | required             | Relative path to directory where predictions are stored.     |
  | `probs_prefix` | `str`    | optional; 'baseline' | Prefix for probabilities to load, e.g. 'diffusion', 'baseline'. |

  

- `select_data(Xdata, Ydata, location, test_pid, train_pred_all, test_pred_all, partitions_builder, train_size=0.1, n_hexagons=47, split=0, return_location=False, generate_hexagons=False)`

  Selects data for a single split, hex size, train size combination. Returns all information in a dictionary with keys reported below.

  | **Argument**         | **Type**            | **Status**       | **Description**                                              |
  | -------------------- | ------------------- | ---------------- | ------------------------------------------------------------ |
  | `Xdata`              | `pd.DataFrame`      | required         | The dataframe containing all features for every parcel including both the train and test set. Will be internally determined which to select. |
  | `Ydata`              | array-like          | required         | Array-like containing all outcome information for all parcels |
  | `train_pid`          | `dict`              | required         | Dictionary containing all training PIDs by train / hex / split ordering |
  | `test_pid`           | `dict`              | required         | Dictionary containing all test PIDs by train / hex / split ordering |
  | `train_pred_all`     | `dict`              | required         | Dictionary containing all training predictions for the model |
  | `test_pred_all`      | `dict`              | required         | Dictionary containing all test predictions for the model     |
  | `partitions_builder` | `PartitionsBuilder` | optional; None   | If generating hexagons, must pass a `PartitionsBuilder` object from the `gizmo` package. |
  | `train_size`         | `float`             | optional; 0.1    | Share of data in train set                                   |
  | `n_hexagons`         | `int`               | optional; 47     | Number of hexagons across grid                               |
  | `split`              | `int`               | optional; 0      | Split number                                                 |
  | `return_location`    | `bool`              | optional;  False | If `True`, this will return the latitude / longitude of the training and test parcels. |
  | `generate_hexagons`  | `bool`              | optional; False  | If `True` will generate hexagons object necessary for mapping and aggregation. |

  | **Return**    | **Type** | **Description**                                              |
  | ------------- | -------- | ------------------------------------------------------------ |
  | `result_dict` | `dict`   | Dictionary containing all pertinent information; keys listed below as separate section. |

  | `result_dict` key   | **Type**                  | **Description**                                              |
  | ------------------- | ------------------------- | ------------------------------------------------------------ |
  | `train_pid`         | `np.array`                | Array of training set PIDs                                   |
  | `test_pid`          | `np.array`                | Array of test set PIDs                                       |
  | `Xtrain`            | `pd.DataFrame`            | `Xdata` subset to only the parcels in the train set          |
  | `Xtest`             | `pd.DataFrame`            | `Xdata` subset to only the parcels in the test set           |
  | `Ytrain`            | `np.array`                | `Ydata` subset to only the parcels in the train set          |
  | `Ytest`             | `np.array`                | `Ydata` subset to only the parcels in the test set           |
  | `train_pred`        | `np.array`                | Model predictions for the training set, indexed in same way as the `train_pid` and `Ytrain`. |
  | `test_pred`         | `np.array`                | Model predictions for the test set, indexed in same way as the `test_pid` and `Ytest`. |
  | `train_graph_index` | `np.array`                | Because the graph (i.e. distance matrix) was created using only the indexes rather than the PIDs, this provides a way to access the correct entries in the distance matrix, once again ordered in the same way as the `Ytrain` and `train_pred`. |
  | `test_graph_index`  | `np.array`                | See `train_graph_index` but for the test set.                |
  | `location_train`    | `pd.DataFrame`            | DataFrame with latitude / longitude info for the training set. Only appears when `return_location == True`. |
  | `location_test`     | `pd.DataFrame`            | DataFrame with latitude / longitude info for the test set. Only appears when `return_location == True`. |
  | `hexagons`          | `gizmo.PartitionsBuilder` | Hexagons object (containing partition and parcel data frames). Only generated when `generate_hexagons == True`. Not commonly used when hyperparameter tuning because can be computationally costly with no advantages in terms of differences in the underlying data. |

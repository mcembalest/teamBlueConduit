


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


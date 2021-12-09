# Evaluation

### Table of Contents
- [Hit Rate Curve](##Generating-Hit-Rate-Curves)
- [Dig Table Statistics](#Generating-digging-statistics-table)
- [API Reference](#API-Reference)


## Generating Hit Rate Curves

There are two primary methods for generating the hit-rate curve, and a single plotting utility for doing so. The functions live in [blue_conduit_spatial.evaluation](../../blue_conduit_spatial/evaluation). First, `generate_hit_rate_curve` will create a hit rate curve ordering by the prediction probability. If generating the hit rate curve by partition (i.e. group by the partition and then investigate partitions with more expected lead first), this can be done by `generate_hit_rate_curve_by_partition`. These are equivalent when `threshold_init=1.0`, `threshold_increment=1e-6` (or some other arb. small number), and `min_digs=0`. That will visit partitions exactly in the order of the highest probability (though will be substantially slower).

### Sample usage:

```python3
from blue_conduit_spatial.utilities import load_datasets, load_predictions, select_data
from blue_conduit_spatial.evaluation import generate_hit_rate_curve, generate_hit_rate_curve_by_partition, plot_hit_rate_curve, dig_stats, dig_savings

data_dir = '../data'
load_dir = f'{data_dir}/Processed'
pred_dir = f'{data_dir}/Predictions'
pid_lat_lon_path = f'{load_dir}/pid.gpkg'

# Load data for all hexagons resolutions, train sizes and splits
Xdata, Ydata, location, train_pid, test_pid, partitions_builder = load_datasets(load_dir)
train_pred_all_bl, test_pred_all_bl = load_predictions(pred_dir, probs_prefix='baseline')
train_pred_all_diff, test_pred_all_diff = load_predictions(pred_dir, probs_prefix='diffusion')

# Filter the data for one selection of hyperparameters (hexagons resolutions, train sizes and splits)
n_hexagons = 47
train_size = 0.1
split = 0

args = {    
    'Xdata': Xdata,
    'Ydata': Ydata,
    'location': location,
    'train_pid': train_idx,
    'test_pid': test_idx,
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

![hrc-baseline](../../plots/plot_hrc_baseline.png)


### Comparison between HRC methods

In the plot below, we demonstrate the differences in performance for the Blue Conduit baseline, for a single split / resolution / set of hyperparameters. Note that most splits / resolutions show qualitatively similar results. Increasing initial threshold improves performance over initial homes. Decreasing increment has similar parameter. Decreasing minimum num. homes improves relative performance over second half of homes.

![hrc-comparison](../../plots/hit_rate_curve_comparison.png)

## Generating digging statistics table

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

![hrc-comparison](../../plots/table_digs.png)

```
mode = 'digs_lead_number'
dig_stats_df = dig_stats(parcel_gdf, index_list, y_true, y_pred, strat_names=strat_names, bins=bins, mode=mode)
dig_stats_df = dig_savings(dig_stats_df, 'Baseline', 'Diffusion')
dig_stats_df.head()
```

![hrc-comparison](../../plots/table_digs_lead.png)

## API Reference

Note: Most evaluation functions are written to approximate the `sklearn` API for metrics. That is, the general formula will be `metric(y_true, y_pred)` though some metrics require additional parameters to be passed.

###### Metrics

- `hit_rate(y_true, y_pred, threshold=0.5)`

  Returns the hit rate if using a particular probabilistic threshold. Equivalent to the precision on the positive (read: lead) class.

  | **Argument** | **Type**   | **Status**              | **Description**                                              |
  | ------------ | ---------- | ----------------------- | ------------------------------------------------------------ |
  | `y_true`     | `np.array` | required                | Ground truth labels. Should be binary.                       |
  | `y_pred`     | `np.array` | required                | Prediction probabilities or class labels                     |
  | `threshold`  | `float`    | optional; default = 0.5 | Float threshold for considering a prediction to be 'dangerous' |

  

- `generate_hit_rate_curve(y_true, y_pred)`

  Generates cumulative hit rate curve on a sample by sample basis for entire set. Note: this assumes that data passed is ordered consistently between `y_true` and `y_pred`. That is, for a given index `i`, the parcel ID for the parcel at `y_true[i]` should be the same as `y_pred[i]`.  

  | **Argument** | **Type**   | **Status** | **Description**                          |
  | ------------ | ---------- | ---------- | ---------------------------------------- |
  | `y_true`     | `np.array` | required   | Ground truth labels. Should be binary.   |
  | `y_pred`     | `np.array` | required   | Prediction probabilities or class labels |

  | **Return**                | **Type**   | **Description**                              |
  | ------------------------- | ---------- | -------------------------------------------- |
  | `hit_rates`               | `np.array` | Cumulative hit rates for ordered test set    |
  | `predicted_probabilities` | `np.array` | Predicted probabilities for ordered test set |

  

- `generate_hit_rate_curve_by_partition(parcel_df, pid_list, y_true, y_pred, threshold_init, threshold_increment=0.1, min_digs=1, min_digs_increment=1, gen_dig_metadata=False)`

  Generates a hit rate curve where parcels are investigated partition. 

  

  Made to be inter-operable with the `gizmo` partitioning. Agnostic to shape of the partition, but must pass a parcel DataFrame which contains a parcel ID. Can replicate initial analyses by passing an ordered DataFrame with `partition_ID = 'PRECINCT'`.

  | **Argument**          | **Type**        | **Status**                  | **Description**                                              |
  | --------------------- | --------------- | --------------------------- | ------------------------------------------------------------ |
  | `parcel_df`           | `gpd.DataFrame` | required                    | Ordered DataFrame passing reference to `pid` and `partition_ID` |
  | `pid_list`            | `np.array`-like | required                    | Array-like list of indices in the i.e. test set (as used to divide train/test data). |
  | `y_true`              | `np.array`      | required                    | Ground truth labels. Should be binary.                       |
  | `y_pred`              | `np.array`      | required                    | Prediction probabilities or class labels                     |
  | `threshold_init`      | `float`         | required                    | Initial threshold for decision-making.                       |
  | `threshold_increment` | `float`         | optional; default = 0.1     | Float to decrease threshold by at each iteration / sweep     |
  | `min_digs`            | `int`           | optional; default = 1       | Minimum number of digs required in a partition to visit.     |
  | `min_digs_increment`  | `int`           | optional; default = 1       | Minimum digs to increment at each sweep                      |
  | `gen_dig_metadata`    | `bool`          | optional; default = `False` | Returns the IDs of the digging order as well as the partition IDs when `True` |

  | **Return**                | **Type**       | **Description**                                              |
  | ------------------------- | -------------- | ------------------------------------------------------------ |
  | `hit_rates`               | `np.array`     | Cumulative hit rates for ordered test set                    |
  | `predicted_probabilities` | `np.array`     | Predicted probabilities for ordered test set                 |
  | `dig_metadata`            | `pd.DataFrame` | If `gen_dig_metadata` is False, None. Else a DataFrame containing dig order, ID, and partition ID. |



- `dig_stats_base(dig_data, criteria, include_cost=False)`

  - *Javiera*

- `dig_stats(parcel_gdf, index_list, y_true, y_pred, strat_names=None, bins=15, mode='dig_lead_number', hr_args=None)`

  - *Javiera*

- `dig_savings(dig_stats, model1_str, model2_str)`

  - *Javiera*

- `roc_auc_score(y_true, y_pred)`

  Returns the ROC-AUC Score for some y_true and y_test. 

  | **Argument** | **Type**   | **Status** | **Description**                          |
  | ------------ | ---------- | ---------- | ---------------------------------------- |
  | `y_true`     | `np.array` | required   | Ground truth labels. Should be binary.   |
  | `y_pred`     | `np.array` | required   | Prediction probabilities or class labels |

  | **Return**      | **Type** | **Description**               |
  | --------------- | -------- | ----------------------------- |
  | `roc_auc_score` | `float`  | ROC-AUC score for the dataset |

- `generate_calibration_curve(y_true, y_pred, n_bins, **kwargs)`

  Mask (with error handling) for sk-learn calibration curve.

  | **Argument** | **Type**   | **Status**             | **Description**                                     |
  | ------------ | ---------- | ---------------------- | --------------------------------------------------- |
  | `y_true`     | `np.array` | required               | Ground truth labels. Should be binary.              |
  | `y_pred`     | `np.array` | required               | Prediction probabilities or class labels            |
  | `n_bins`     | `int`      | optional; default = 10 | Number of bins to discretize for calibration curve. |
  

  | **Return**   | **Type**   | **Description**                 |
  | ------------ | ---------- | ------------------------------- |
  | `true_curve` | `np.array` | Bucket mean true positives      |
  | `pred_curve` | `np.array` | Bucket mean predicted positives |

###### Plots

- `plot_hit_rate_curve(y_true, y_pred, plot_probs=False, labels=None, max_perf=False, order_by_prob=False, figsize=(10,6), savefig=False, figname=None, figdir=None, mode='all', parcel_df=None, pid_list=None, threshold_init=None, title_suffix=None, min_hit_rate=0.0, custom_cmap=None, **kwargs)`

  Generates plot of hit rate curve with highly flexible set of options.

  Important Design Choices:

  1. *`#` of models*: Can pass multiple models as a list to `y_pred` and concurrent number of labels as list to `labels`.
  2. *Investigation 'mode'*: Can investigate plots with `mode == 'all'` which will dig parcels with highest P(lead). If `mode == 'partition'` curve will be done by partition & sweeps. Controlled via `threshold_init` parameter.
  3. *Plot prediction probability?*: Can plot only hit rate curve or also prediction probability alongside for each model as dotted line.
  4. *Draw a line with maximum possible performance*: this mode (controlled via) `max_perf` allows user to locate a line where the 'kink' would occur with perfect (i.e. all lead then all non-lead) behavior.
  5. *Order by probability*: Rather than draw in space, `order_by_prob` can be used to draw the hit rate curves in prob. space rather than sample space.
  6. *Show x-axis as the sample # or the % of test set*.

  | **Argument**     | **Type**        | **Status**       | **Description**                                              |
  | ---------------- | --------------- | ---------------- | ------------------------------------------------------------ |
  | `y_true`         | `np.array`      | required         | Ground truth outcomes for dangerous/not dangerous            |
  | `y_pred`         | `np.array`      | required         | Either a list of model prediction probabilities or a single model outcomes |
  | `plot_probs`     | `bool`          | optional; True   | Boolean for whether to include prediction probabilities in model |
  | `labels`         | `list`          | optional; None   | Labels to include if `y_pred` is a list                      |
  | `max_perf`       | `bool`          | optional; False  | Indicates whether to plot the 'maximum performance' or the kink in the curve where a perfect model would decrease performance. |
  | `figsize`        | `tuple`         | optional; (10,6) | Follows `matplotlib` fig size convention of (h, w)           |
  | `savefig`        | `bool`          | optional; False  | Boolean indicating whether to save figure                    |
  | `figname`        | `str`           | optional; None   | Figure filepath / file title (not nec. title of plot)        |
  | `figdir`         | `str`           | optional; None   | Directory to save figure.                                    |
  | `mode`           | `str`           | optional; "all"  | One of "all" or "partition". Controls whether parcels are investigated in unrestricted way or partition-by-partition, ordering by the highest priority partitions. |
  | `parcel_df`      | `gpd.DataFrame` | optional; None   | Required in "partition" mode. Will guide the partitions that each parcel belongs to for aggregating partition investigation decisions. |
  | `pid_list`       | array-like      | optional; None   | Required in "partition" mode. Guides list of which PIDs in `parcel_df` are actually in the test set. |
  | `threshold_init` | `float`         | optional; None   | Required in "partition" mode. Sets the initial threshold for digging. Typically considered to be 0.9 in baseline of `generate_hit_rate_curve_by_partition`. |
  | `title_suffix`   | `str`           | optional; None   | Suffix to be included in plot title.                         |
  | `min_hit_rate`   | `float`         | optional; 0.0    | Min hit rate to show, translates to lower x-limit if not showing entire x-axis |
  | `custom_cmap`    | `list`          | optional; None   | Customizable `matplotlib` `cmap`; must be a list and will otherwise select the 'Dark2' palette. |
  | `show_as_pct`    | `bool`          | optional; False  | Option to show all plots as a percent of the test set.       |



- `plot_calibration_curve(y_true, y_pred, n_bins, labels=None, figsize=(10,6), savefig=False, figname=None, figdir=None, **kwargs)`

  Plots probability calibration curve for various number of model results

  | **Argument** | **Type**   | **Status**       | **Description**                                              |
  | ------------ | ---------- | ---------------- | ------------------------------------------------------------ |
  | `y_true`     | `np.array` | required         | Ground truth outcomes for dangerous/not dangerous            |
  | `y_pred`     | `np.array` | required         | Either a list of model prediction probabilities or a single model outcomes |
  | `n_bins`     | `int`      | optional; 10     | Number of bins to discretize for each model                  |
  | `labels`     | `list`     | optional; None   | Labels to include if `y_pred` is a list                      |
  | `figsize`    | `tuple`    | optional; (10,6) | Follows `matplotlib` fig size convention of (h, w)           |
  | `savefig`    | `bool`     | optional; False  | Boolean indicating whether to save figure                    |
  | `figname`    | `str`      | optional; None   | Figure filepath / file title (not nec. title of plot)        |
  | `figdir`     | `str`      | optional; None   | Directory to save figure.                                    |

  

- ```plot_pr_curve(y_ture, y_pred, labels=None, figsize=(10,6), dpi=90, savefig=False, figname=None, figdir=None)```

  Generates precisio-recall curve plot for single or multiple models

  | **Argument** | **Type**   | **Status**       | **Description**                                              |
  | ------------ | ---------- | ---------------- | ------------------------------------------------------------ |
  | `y_true`     | `np.array` | required         | Ground truth outcomes for dangerous/not dangerous            |
  | `y_pred`     | `np.array` | required         | Either a list of model prediction probabilities or a single model outcomes |
  | `labels`     | `list`     | optional; None   | Labels to include if `y_pred` is a list                      |
  | `dpi`        | `int`      | optional; 90     | `matplotlib` dpi                                             |
  | `figsize`    | `tuple`    | optional; (10,6) | Follows `matplotlib` fig size convention of (h, w)           |
  | `savefig`    | `bool`     | optional; False  | Boolean indicating whether to save figure                    |
  | `figname`    | `str`      | optional; None   | Figure filepath / file title (not nec. title of plot)        |
  | `figdir`     | `str`      | optional; None   | Directory to save figure.                                    |

  
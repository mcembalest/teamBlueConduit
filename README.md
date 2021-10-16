# teamBlueConduit
capstone project for Harvard IACS AC297r

# Libraries setup

```pip install .```

## Build datasets
```build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size_list=None, random_state=42)```

**Example**

```
from blue_conduit_spatial.utilities import build_datasets, load_datasets

data_dir = '../data'
data_raw_path = f'{data_dir}/raw/flint_sl_materials/'
save_dir = f'{data_dir}/test_dir'

Xdata, Ydata, pid, train_idx, test_idx = build_datasets(data_raw_path, save_dir=save_dir)
```
## Load datasets

```load_datasets(load_dir)```

**Example**

```
from blue_conduit_spatial.utilities import build_datasets, load_datasets

data_dir = '../data'
load_dir = f'{data_dir}/test_dir'

Xdata, Ydata, pid, train_idx, test_idx = load_datasets(load_dir)

train_idx.files
>>> ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

train_idx['0.1']
>>> array([array([  537,   539,   540, ..., 24621, 24622, 24623]),
           array([   48,    51,    59, ..., 26859, 26860, 26861]),
           array([ 2893,  2912,  2919, ..., 26852, 26857, 26862])],
           dtype=object)
```

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


# teamBlueConduit
capstone project for Harvard IACS AC297r

# Libraries setup

```pip install .```

## Build datasets
```build_datasets(data_raw_path, save_dir=None, n_splits=3, train_size=.75, random_state=42)```

**Example**

```
from blue_conduit_spatial.utilities import build_datasets
save_dir = f'data/test_dir'  
Xtrain, Xtest, Ytrain, Ytest = build_datasets(data_raw_path, save_dir=save_dir)
```
## Load datasets

**Example**

```
from blue_conduit_spatial.utilities import load_datasets
load_dir = f'data/test_dir'  
Xtrain, Xtest, Ytrain, Ytest = load_datasets(load_dir)
```

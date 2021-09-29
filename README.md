# teamBlueConduit
capstone project for Harvard IACS AC297r

# Libraries setup

```pip install .```

## Build datasets

```
from blue_conduit_spatial.data import build_datasets
save_dir = f'data/test_dir'  
Xtrain, Xtest, Ytrain, Ytest = build_datasets(data_raw_path, save_dir=save_dir)
```
## Load datasets

```
from blue_conduit_spatial.data import load_datasets
load_dir = f'data/test_dir'  
Xtrain, Xtest, Ytrain, Ytest = load_datasets(load_dir)
```

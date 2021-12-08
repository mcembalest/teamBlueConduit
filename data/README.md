# Data Directory

## Directory structure

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
...
```

All files can be replicated locally, though the distance matrices are > 5GB and thus were handled via Google Colab.

## Data Dictionary

The raw data (stored as `flint_sl_materials/`) can be connected to the following data dictionary:

- [Data Dictionary](./flint_raw_sl_materials_dictionary.xlsx)
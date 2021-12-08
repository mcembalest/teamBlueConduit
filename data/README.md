# Data Directory

## Directory structure

To reduce space locally, we have utilized a consistent structure of the data folders. Below is a brief tutorial on how to replicate the data directories. First, the following directory structure must be created:

```
...
├── data
│   ├── README.md
│   ├── predictions
│   │   ├── GP_pred_probs_test.npz
│   │   ├── GP_pred_probs_train.npz
│   │   ├── GP_spatiotemporal_pred_probs_test.npz
│   │   ├── GP_spatiotemporal_pred_probs_train.npz
│   │   ├── GP_spatiotemporal_var_pred_probs_test.npz
│   │   ├── GP_spatiotemporal_var_pred_probs_train.npz
│   │   ├── GPvar_pred_probs_test.npz
│   │   ├── GPvar_pred_probs_train.npz
│   │   ├── GraphSAGE_pred_probs_test.npz
│   │   ├── GraphSAGE_pred_probs_train.npz
│   │   ├── baseline_pred_probs_test.npz
│   │   ├── baseline_pred_probs_train.npz
│   │   ├── diffusion_pred_probs_test.npz
│   │   ├── diffusion_pred_probs_train.npz
│   │   ├── stacking_pred_probs_test.npz
│   │   └── stacking_pred_probs_train.npz
│   ├── processed
│   │   ├── Xdata.csv
│   │   ├── Ydata.csv
│   │   ├── cols_metadata.json
│   │   ├── haversine_distances.npz
│   │   ├── location.gpkg
│   │   ├── partitions_builder.pk
│   │   ├── road_distances.npz
│   │   ├── test_pid.npz
│   │   └── train_pid.npz
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

## Data Dictionary

The raw data (stored as `flint_sl_materials/`) can be connected to the following data dictionary:

- [Data Dictionary](./flint_raw_sl_materials_dictionary.xlsx)
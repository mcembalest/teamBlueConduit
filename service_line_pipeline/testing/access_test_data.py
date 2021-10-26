from service_line_pipeline.data import (
    get_benton_harbor_data,
    get_toledo_data,
    get_flint_data,
    get_trenton_data,
)
from gizmo import evaluation
import geopandas, geoplot
import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import service_line_pipeline.models
from service_line_pipeline.utils import (
    build_cmd_parser,
    get_csv_files_from_dir,
    load_client_configs,
)
from service_line_pipeline.models import build_models, get_models
from service_line_pipeline.data import get_data

from matplotlib import pyplot as plt
from pathlib import Path

from gizmo.spatial_partitions import partitions

import xgboost as xgb
from sklearn.metrics import roc_auc_score as roc
import pickle
from gizmo.bc_logger import get_simple_logger

log = get_simple_logger(__name__)


CLIENT_SAMPLE_DATA_DIR = Path("data/client_data_snippets")
client_get_data_dict = {
    "benton_harbor": get_benton_harbor_data,
    "flint": get_flint_data,
    "toledo": get_toledo_data,
    "trenton": get_trenton_data,
}


def write_sample_data(
    client_name,
    X,
    num_rows=1000,
    sample_data_dir=CLIENT_SAMPLE_DATA_DIR,
    random_state=42,
):
    """
    Function for creating a short (~1k row random sample) version of
    a SLIRP client GeoDataFrame, saved as a pickle file, and generated
    using preexisting get_data functions. This 'client data snippet' is
    stored in its own pickle file in the sample_data_dir.

    This function is only intended to be run once, to generate and write this file
    locally (takes around 1 minute / flint-sized-city). Thereafter, pickle file
    should be accessed using the following function, get_sample_data().
    """
    # Make sure the directory exists. If not, create it.
    CLIENT_SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_file_name = CLIENT_SAMPLE_DATA_DIR / "{}.pkl".format(client_name)

    # only include rows where label is not a nan.
    target_col, id_col = get_target_and_id(client_name=client_name)
    X = X.loc[~X[target_col].isna(), :].reset_index(drop=True)

    # Write a subsample of the gdf; or write the full file if it's short.
    with open(sample_file_name, "wb") as out_file:
        if num_rows > len(X):
            pickle.dump(X, out_file)
        else:
            pickle.dump(X.sample(num_rows, random_state=random_state), out_file)

    log.info(
        "Wrote {} rows of gdf in pickle form at {}".format(num_rows, sample_file_name)
    )


def get_sample_data(client_name, sample_data_dir=CLIENT_SAMPLE_DATA_DIR):
    sample_file_name = CLIENT_SAMPLE_DATA_DIR / "{}.pkl".format(client_name)

    if not sample_file_name.exists():
        print(
            """
        Generating sample file for the first time --
        downloading the whole thing and then saving 1k rows.
        """
        )
        X = client_get_data_dict[client_name]()

        write_sample_data(client_name=client_name, X=X, num_rows=1000)

    with open(sample_file_name, "rb") as in_file:
        X = pickle.load(in_file)

    return X


def get_target_and_id(client_name, client_config_path="configs/clients.yaml"):
    client_configs = load_client_configs(Path(client_config_path))
    client_config = client_configs[client_name]
    split_config = client_config.get("splits") or {}
    target_column, id_column = [*split_config.values()]

    return target_column, id_column


# Packages all three of the above functions into one generator.
def get_models_and_data(client="flint"):
    X = get_sample_data(client_name=client)

    models = get_models(client_name=client)

    target_col, id_col = get_target_and_id(client_name=client)

    target_columns = [target_col, id_col]

    labeled_data = X.loc[~X[target_col].isna(), :].reset_index(drop=True)

    X, y = (
        labeled_data.drop(target_columns, axis="columns"),
        # TODO take care of bad y in data fetching (before here)
        labeled_data[target_col].astype(int).to_numpy(),
    )

    for model_name, model in models.items():
        yield X, y, model, model_name

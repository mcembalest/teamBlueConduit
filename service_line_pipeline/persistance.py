#!/usr/bin/env python3
"""
A module containing the save/load functions for persisting sources like data,
models, figures, etc,. to sinks like disk, s3, a database, etc,.

TODO Pull out into gizmo -- probably no reason to have this in the main pipeline.
"""

import joblib as jl
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from hashlib import md5
from tempfile import TemporaryFile
from datetime import datetime as dt
import os
from pathlib import Path
import pandas as pd

from typing import Union, Dict, Tuple

from gizmo.bc_logger import get_simple_logger

log = get_simple_logger(__name__)


def load_model(model_fp: Union[Path, str]):
    """Take a filepath to a serialized model and deserialize into memory."""
    # Defensively coerce to `Path`
    model_fp = Path(model_fp).resolve()
    deserialized_model = jl.load(model_fp)
    log.debug("Loaded Model", model_filename=model_fp.name)
    return deserialized_model


def hash_object(obj, hash_name="md5", abbreviate_hash=True):
    """The joblib hash function follows references to numpy arrays and actually
    hashes that data, where python's normal `hash` will ignore them.

    NOTE The implementer is not totally sure of all the edge cases, but tests in
    a repl yeild exected results, so we should be fine for now."""
    result = jl.hash(obj, hash_name=hash_name)
    return result[:8] if abbreviate_hash else result


def make_model_name(model, prefix: str = "") -> str:
    model_hash = hash_object(model)
    date_string = dt.now().date().isoformat()
    computed_name = f"{date_string}-{model_hash}"
    return f"{prefix}-{computed_name}" if prefix else computed_name


def save_model(
    model,
    client_name: str = "",
    model_basename: str = "",
    model_dir: str = "./models",
    sl_segment: str = "",
) -> Path:
    """Save a model to disk. If a model name is not provided, the default will
    be a function of today's date and the bytes of the joblib file."""

    # TODO really? It's not fitted?
    # check_is_fitted(model)
    model_name = make_model_name(model, prefix=model_basename)
    model_fp = Path(f"{model_dir}/{client_name}.{sl_segment}.{model_name}.joblib").resolve()

    # Make parents if they don't exist
    model_fp.parents[0].mkdir(parents=True, exist_ok=True)

    jl.dump(model, model_fp)
    log.debug("Saved Model", model_filename=model_fp.name)
    return model_fp


def save_metrics_csv(
    model_base_name: str,
    dataset_hash: str,
    cv_constructor: str,
    cv_model_metrics: Dict,
    metrics_directory: str = "./metrics",
) -> Path:
    metrics_df = pd.DataFrame.from_dict(cv_model_metrics)

    saved_model_metrics_filename = (
        f"{model_base_name}.{dataset_hash}.{cv_constructor}.metrics"
    )
    saved_model_metrics_csv_filepath = (
        Path(metrics_directory) / Path(saved_model_metrics_filename + ".csv")
    ).resolve()

    Path(metrics_directory).resolve().mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(saved_model_metrics_csv_filepath, index=False)

    return saved_model_metrics_csv_filepath


def save_metrics_dict(
    model_base_name: str,
    dataset_hash: str,
    cv_strategy: str,
    metrics_payload: Dict,
    metrics_directory: str = "./metrics",
) -> Path:
    saved_model_metrics_filename = (
        f"{model_base_name}.{dataset_hash}.{cv_strategy}.metrics"
    )
    saved_model_metrics_dict_filepath = (
        Path(metrics_directory) / Path(saved_model_metrics_filename + ".joblib")
    ).resolve()

    Path(metrics_directory).resolve().mkdir(parents=True, exist_ok=True)

    # Save payload
    jl.dump(metrics_payload, saved_model_metrics_dict_filepath)

    return saved_model_metrics_dict_filepath


def save_summarized_metrics_csv(
    model_base_name: str,
    dataset_hash: str,
    cv_constructor: str,
    cv_model_metrics: Dict,
    metrics_directory: str = "./metrics",
) -> Path:
    metrics_df = pd.DataFrame.from_dict(cv_model_metrics)

    saved_model_metrics_filename = (
        f"{model_base_name}.{dataset_hash}.{cv_constructor}.metrics"
    )
    saved_model_metrics_summary_csv_filepath = (
        Path(metrics_directory)
        / Path(".".join([saved_model_metrics_filename, "summary", "csv"]))
    ).resolve()

    metrics_list = [
        "n",
        "y_true_mean",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "brier_score",
    ]

    metrics_df[metrics_list].describe().drop("count").to_csv(
        saved_model_metrics_summary_csv_filepath,
        float_format="%.4f",
    )

    return saved_model_metrics_summary_csv_filepath


def save_metrics(
    model_base_name: str,
    dataset_hash: str,
    evaluation_results_dict: Dict,
    metrics_directory: str = "./metrics",
) -> Tuple[Path, Path]:

    # Defensively create directory if it does not alreay exist
    Path(metrics_directory).resolve().mkdir(parents=True, exist_ok=True)

    saved_model_metrics_dict_filepath = save_metrics_dict(
        model_base_name, dataset_hash, evaluation_results_dict
    )

    cv_model_metrics = evaluation_results_dict.get("cv_model_metrics") or {}
    saved_model_metrics_csv_filepath = save_metrics_csv(
        model_base_name, dataset_hash, cv_model_metrics
    )

    return saved_model_metrics_csv_filepath, saved_model_metrics_dict_filepath


def load_metrics(csv_files, consolidate=False):

    csv_files = [Path(fp).resolve() for fp in csv_files]

    metrics_dfs = []
    for filename in csv_files:
        this_df = pd.read_csv(filename)
        this_df["model_name"] = filename.name.split(".")[0]
        this_df["dataset_hash"] = filename.name.split(".")[1]
        metrics_dfs.append(this_df)

    return pd.concat(metrics_dfs).reset_index(drop=True) if consolidate else metrics_dfs


def save_predictions(
    *,
    model_filename=None,
    dataset_hash=None,
    predictions_df=None,
    predictions_dir="./predictions",
) -> Path:
    predictions_dir = Path(predictions_dir)
    predictions_filename = Path(f"{model_filename}.{dataset_hash}.predictions.csv")

    predictions_filepath = (predictions_dir / predictions_filename).resolve()

    # Defensively make directory if it doesn't exist.
    predictions_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_filepath, index=False)

    log.debug(
        "Saved Predictions CSV",
        predictions_filename=predictions_filepath.name,
    )

    return predictions_filepath


# TODO Save to parent shapefile directory so in predicitons shapefiles are
# folders with their assets as siblings.
def save_shapefile(
    *,
    geodataframe=None,
    model_filename=None,
    dataset_hash=None,
    predictions_dir="./predictions",
):
    predictions_dir = Path(predictions_dir)
    predictions_filename = Path(f"{model_filename}.{dataset_hash}.predictions.shp")
    predictions_filepath = (predictions_dir / predictions_filename).resolve()

    geodataframe.to_file(predictions_filepath)

    log.debug(
        "Saved Predictions Shapefile",
        predictions_filename=predictions_filepath.name,
    )

    return predictions_filepath

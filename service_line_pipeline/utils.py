#!/usr/bin/env python3

import argparse
import logging
import yaml


from os import listdir
from pathlib import Path


def get_csv_files_from_dir(directory):
    return [
        Path(directory, filename).resolve()
        for filename in filter(
            lambda s: s.split(".")[-1] == "csv", listdir(Path(directory))
        )
    ]


def load_client_configs(clients_filepath: str = "./configs/clients.yaml"):
    with open(clients_filepath, "r") as f:
        clients = yaml.safe_load(f.read())
    return clients


# TODO Change to `clients`, take list, filter client configs for enabled and
# specified. If done correctly, this allows us to fold the end of main up to
# traversing a list where the default is all.
def build_cmd_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Machine learning pipeline orchestrator"
            " for service line replacement models."
        )
    )

    parser.add_argument(
        "--clients",
        dest="clients",
        nargs="+",
        help=(
            "Client name. Dictates which config to consume, and subsequently"
            "which model to run. If unspecified, will run all enabled clients."
        ),
    )

    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Train a model for a client one time.",
    )

    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.add_argument("--predict", dest="predict", action="store_true")
    parser.add_argument("--data-dir", dest="data_dir", default="./data")
    parser.add_argument("--model-dir", dest="model_dir", default="./models")
    parser.add_argument("--metric-dir", dest="metric_dir", default="./metrics")

    parser.add_argument(
        "--clients-config", dest="clients_config", default="./configs/clients.yaml"
    )

    return parser

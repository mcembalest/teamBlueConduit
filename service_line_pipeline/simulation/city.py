"""A simple read-only class that allows us to couple the cost details of a city
with the available ground truth data."""

import pandas as pd
from dataclasses import dataclass

from service_line_pipeline.data import get_data
from service_line_pipeline.utils import load_client_configs


@dataclass
class City:
    client_name: str
    explore_cost: float = 250.0
    excavate_cost: float = 3000.0
    replace_cost: float = 5000.0

    # NOTE consider calculating this based on the sim state instead of mutating
    # this incrementally with each step. Why? Mutating values means that reuse
    # of the same object without reinitializing could produce "surprising"
    # results to a user.
    total_cost: float = 0.0

    unique_id: str = None
    target_column: str = None

    # NOTE Private data member is initialized when calling `city.data` for the
    # first time.
    _data: pd.DataFrame = None

    @property
    def data(self):
        """Written as a property to ensure you cannot accidentally reset data.
        The one caveat that might make this a bad idea is that `unique_id` and
        `target_column` are only initialized after calling/accessing `.data`"""

        # If we already have this in memory, let's just get going.
        if self._data is not None:
            return self._data

        client_config = load_client_configs()[self.client_name]
        split_config = client_config.get("splits")

        self.target_column = split_config["target_column"]
        self.unique_id = split_config["id_column"]

        data_config = client_config.get("data")

        self._data = (
            get_data(
                client_name=self.client_name,
                data_config=data_config,
            )
            # NOTE We drop unknown material types. We might change this.
            .dropna(subset=[self.target_column]).reset_index(drop=True)
        )

        return self._data

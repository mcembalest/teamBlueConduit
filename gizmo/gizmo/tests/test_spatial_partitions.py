from gizmo.spatial_partitions import partitions
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon

import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon


def add_scalar_feature_to_gdf(gdf, scalar_func_xy, feature_name):
    gdf = gdf.copy()
    gdf["x"] = gdf.centroid.x
    gdf["y"] = gdf.centroid.y
    vscalar_func_xy = np.vectorize(scalar_func_xy)
    gdf[feature_name] = vscalar_func_xy(gdf.x, gdf.y)
    return gdf

def add_binary_feature_to_gdf(gdf, feature_name="binary_feature"):
    gdf = gdf.copy()

    gdf[feature_name] = np.random.randint(2, size=len(gdf))

    return gdf


def make_scattered_gdf(n_points=10 ** 3, length_scale=10 ** 2):
    from numpy.random import default_rng

    rng = default_rng(41)
    xy_array = rng.uniform(low=0, high=length_scale, size=(n_points, 2))
    df = pd.DataFrame(xy_array, columns=["x", "y"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    gdf["parcel_id"] = np.arange(len(gdf))
    gdf["has_lead"] = rng.integers(2, size=len(gdf))

    return gdf


def get_partition_from_simulation():
    gdf = make_scattered_gdf()
    gdf = add_scalar_feature_to_gdf(gdf, lambda x, y: x + y, "simulated_data")
    gdf = add_binary_feature_to_gdf(gdf, "binary_feature")

    partitions_builder = partitions.PartitionsBuilder(
        parcel_gdf=gdf,
        copy_all_cols=True,
        target_col="has_lead",
        parcel_id_col="parcel_id",
    )

    hexagons = partitions_builder.Partition(
        partition_type="hexagon",
    )

    hexagons.compute_partition_stats(
        numeric_feature_cols=["simulated_data"],
    )

    return hexagons


def test_partition_creation():
    hexagons = get_partition_from_simulation()
    assert hexagons.target_col == "has_lead"


def test_custom_partition_type():
    hexagons = get_partition_from_simulation()

    hexagons_gdf = hexagons.partition_gdf
    gdf = hexagons.parcel_gdf
    gdf = add_scalar_feature_to_gdf(gdf, lambda x, y: x + y, "simulated_data")
    gdf = add_binary_feature_to_gdf(gdf, "binary_feature")

    partitions_builder = partitions.PartitionsBuilder(
        parcel_gdf=gdf,
        copy_all_cols=True,
        target_col="has_lead",
        parcel_id_col="parcel_id",
    )

    custom = partitions_builder.Partition(
        partition_type="custom",
        partition_gdf=hexagons_gdf)

    parcel_gdf, partition_gdf = custom.compute_partition_stats(
        binary_feature_cols=["binary_feature"],
        numeric_feature_cols=["simulated_data"]
    )

    return parcel_gdf

import pandas as pd
import geopandas as gpd
import numpy as np
import math
import scipy
import networkx as nx
import osmnx as ox
import requests
import os
import shutil
import pickle
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, polygonize
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gizmo.bc_logger import get_simple_logger

from pathlib import Path

log = get_simple_logger(__name__)


##### DEFINITIONS FROM AD & RG ########
# Grid: A collection of regular polygons (could be rectangles, could be hexagons, etc).
#   * represented as a gdf of polygons.
#
#
# Cell: A single polygon within a grid. A cell is necessarily equal shape and area to
#   all of its neighbors. A cell also gets an ID, which is string(lat + lon).
#   * Super important: Cell IDs should be computed fresh each time you run the code.
#
#
# Block: A polygon that arises from the enclosed polygons of a street map from osmnx.
#   * Every parcel is only in a single block. A parcel can't be in two blocks at once,
#     and no parcel is touching more than one block.
#   * Some blocks don't have parcels.
#   * Blocks should always tile the map, such that there's no gaps and no overlaps.
#   * Each block has a unique blockID, based on the arbitrary download ordering of the list of blocks.
#
#
# Parcel: A pre-defined polygon representing a single indivisible property unit.
#   * every parcel has a unique ID.
#   * the source of data is typically from the client / external data source.
#   * The unit at which the final target variable (lead/not-lead) is measured.
#
#
# Graph: a defined set of nodes and edges, and each node also has lat/lon associated with it.
#   * right now, lat/lon is just centroids of parcels or blocks or cells.
#   * the only rule for determining neighbors is sharing queen boundaries (without a buffer).
#
#
# Connected subgraph: centered on a specified node N, then given size k nodes, it will then
#   gather breadth-first-search neighbors until the subgraph is of size k:
#   * once you've all gathered all neighbors {n} around N, and if k is still not met,
#   collect the set of all neighbors to each n_i in {n}, and then order by distance between
#   ||[n_i centroid] - [N centroid]||; choose the smallest distances up until k is satisfied.
#


def get_block_polygons_osmnx(
    *,
    query_str,
    cache_graph_dir=None,
    epsg=None,
    crs=None,
):
    """
    ----------
    query_str: string
        Query string passed to osmnx.
        e.g. 'Toledo, OH, USA'.
    epsg: int
        Optional. Map projection identifier of the returned blocks dataframe.
        If left as none, will default to lat/lon.
    crs: GeoDataFrame.crs
        Optional but highly recommended. Converts output blocks gdf to
        this crs. Recommended source method here: 'crs=my_parcel_gdf.crs'

    Returns
    ------
    gdf: geodataframe
        A geodataframe of block polygons, computed from the street
        graph downloaded from osmnx, and then projected onto the
        given epsg.
    """

    if cache_graph_dir is not None:
        cache_graph_dir = Path(cache_graph_dir)
        cached_graph_path = Path(cache_graph_dir) / "osmnx_street_graph.pkl"
        if not cache_graph_dir.exists():
            log.info("Creating new directory {}".format(cache_graph_dir))
            cache_graph_dir.mkdir(parents=True)
        if cached_graph_path.exists():
            log.info("Loading cached osmnx graph pickle: {}".format(cached_graph_path))
            with open(cached_graph_path, "rb") as in_file:
                G = pickle.load(in_file)
        else:
            log.info("Downloading osmnx street data ...")
            G = ox.graph_from_place(query_str, network_type="drive")
            with open(cached_graph_path, "wb") as out_file:
                pickle.dump(G, out_file)
                log.info("Graph saved to {}".format(cached_graph_path))
    else:
        log.info(
            """Downloading osmnx data. 'cache_graph_dir' arg is not set,
        so this download will repeat each time."""
        )
        G = ox.graph_from_place(query_str, network_type="drive")

    if epsg is not None:
        G = ox.project_graph(G, to_crs="epsg:{}".format(epsg))

    streets_df = ox.utils_graph.graph_to_gdfs(
        G, nodes=False, edges=True, fill_edge_geometry=True
    )

    print("Building polygons from street data ...")
    multi_line = streets_df.geometry.values

    log.debug("unary_union operation")
    border_lines = unary_union(multi_line)

    polies = list(polygonize(border_lines))

    df = pd.DataFrame({"geometry": polies})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = gdf.set_crs(epsg=4326)  # WKT 84 projection.
    if epsg is not None:
        gdf.to_crs(epsg=epsg, inplace=True)
    elif crs is not None:
        gdf = gdf.to_crs(crs)

    log.debug("Total block polygons from osmnx: {}".format(len(gdf)))

    # gdf of block polygons
    return gdf


def add_partition_ID_via_sjoin(
    parcel_gdf, partition_gdf, partition_ID_fieldname="partitionID"
):
    """Takes a parcel gdf and a partition gdf and joins
        them, returning a new parcel gdf, keeping only those
        parcel rows that sit inside one of the partitions. Also
        adds the partition ID to each parcel row.

    Keyword arguments:
    parcel_gdf: geodataframe
        Contains parcel IDs, geometries. We are adding
        partition IDs to each parcel.
    partition_gdf: geodataframe
        The polygons defining the partitions across the
        map.

    Returns:
    joined_: geodataframe
        A geodataframe parcels, but now each as a partition ID,
        determined from the sjoin with the parcels gdf.
    """
    # Build the partition-parcel look-up column and add to self.parcel_gdf.
    log.info(
        "Building the partition-parcel lookup table with add_partition_ID_via_sjoin()"
    )
    parcel_gdf = parcel_gdf.copy()
    partition_gdf = partition_gdf.copy()

    joined_gdf = gpd.sjoin(
        left_df=parcel_gdf,
        right_df=partition_gdf,
        how="left",
        op="within",
    )
    log.debug("Total parcels after join: {}".format(len(joined_gdf)))

    joined_gdf.rename(columns={"index_right": partition_ID_fieldname}, inplace=True)

    return joined_gdf


def plot_nearest(partition_gdf, graph, source, k):
    nearest_dict = nx.single_source_shortest_path(
        G=graph.graph, source=source, cutoff=k
    )

    nearest = [node for node, vals in nearest_dict.items()]

    fig, ax = plt.subplots(figsize=(18, 18))
    partition_gdf.plot(ax=ax, facecolor="gray", edgecolor="black")
    partition_gdf.iloc[nearest].plot(ax=ax, facecolor="yellow", edgecolor="black")
    partition_gdf.iloc[[source]].plot(ax=ax, facecolor="red", edgecolor="black")


def check_partition_type_valid(partition_type):
    # Check whether inputs are valid.
    valid_partition_types = set(["block", "square_grid", "hex_grid"])
    if partition_type not in valid_partition_types:
        raise BaseException(
            "Invalid partition type '{}'. Must be one of {}".format(
                partition_type, valid_partition_types
            )
        )


def get_spatial_dir_name(*, client, partition_type, data_dir="data"):
    check_partition_type_valid(partition_type)

    p = Path("data", "{}-spatial".format(client), "{}-partition".format(partition_type))
    p = p.resolve()

    return p


def get_partition_file_path(partition_dir, filename=None, all=False):
    """If 'all', returns a list of Paths. If 'filename', returns a Path."""

    filenames = [
        "partition_geometry",
        "parcel_partition_lookup.csv",
        "partitions_graph.pkl",
        "partition_visualization.png",
        "graph_visualization.png",
    ]
    if all and filename:
        raise BaseException("Choose one or the other!")
    if all:
        return [partition_dir / file for file in filenames]
    elif filename:
        if not filename in filenames:
            raise BaseException("Specified '{}' not in {}".format(filename, filenames))
        return partition_dir / filename
    else:
        raise BaseException("Must specify either 'filename' or 'all'!")


def check_if_partition_files_exist(client, partition_type, data_dir="data"):
    """Return True if all files exist; False otherwise."""
    partition_dir = get_spatial_dir_name(
        client=client, partition_type=partition_type, data_dir=data_dir
    )
    partition_file_paths = get_partition_file_path(partition_dir, all=True)
    for path in partition_file_paths:
        if not path.exists():
            return False
    return True


class spatial_partition_data_suite:
    def __init__(
        self,
        *,
        partition_type,
        client,
        parcel_gdf=None,
        output_directory="data",
        parcel_id_column="parcel_id",
        partition_id_col="partition_id",
        target_column="has_lead",
        epsg=None,
        osmnx_query_str=None,
        cell_width=None,
        num_cells=None,
        make_partition_plot=False,
        make_graph_plot=False,
        make_parcel_plot=False,
    ):
        check_partition_type_valid(partition_type)
        self.partition_type = partition_type
        self.client = client

        parcel_gdf = parcel_gdf[["geometry", parcel_id_column, target_column]].copy()
        parcel_gdf = parcel_gdf.to_crs(epsg=epsg)
        self.parcel_gdf = parcel_gdf
        self.parcel_id_column = parcel_id_column
        self.partition_id_column = partition_id_col
        self.target_column = target_column
        self.output_directory = output_directory
        self.epsg = epsg
        self.osmnx_query_str = osmnx_query_str
        self.cell_width = cell_width
        self.num_cells = num_cells

        # Get directory name and check if directory already exists.
        partition_dir = get_spatial_dir_name(
            client=client, partition_type=partition_type
        )
        self.partition_dir = partition_dir

        if not partition_dir.exists():
            log.info("Making new directory: {}".format(partition_dir))
            partition_dir.mkdir(parents=True)
        if cell_width is None:
            cell_width = 2 * 10 ** 3

        # make partition_gdf; save to file.
        if self.partition_type == "block":
            if not osmnx_query_str:
                raise BaseException("Missing: must provide osmnx_query_str")
            if not epsg:
                log.info(
                    "Warning! epsg is set to None, which may result in some odd projection behavior."
                )
            partition_gdf = get_block_polygons_osmnx(
                query_str=osmnx_query_str,
                cache_graph_dir=partition_dir,
                epsg=epsg,
            )
            partition_gdf.index = partition_gdf.index.rename(self.partition_id_column)
        elif self.partition_type == "square_grid":
            partition_gdf = tesselate(gdf=parcel_gdf, l=cell_width, shape="square")
        elif self.partition_type == "hex_grid":
            partition_gdf = tesselate(gdf=parcel_gdf, l=cell_width, shape="hexagon")

        self.partition_gdf = partition_gdf

        # Write partition gdf to file.
        partition_gdf_path = get_partition_file_path(
            partition_dir=partition_dir, filename="partition_geometry"
        )
        log.debug("Making partition_gdf directory: {}".format(partition_gdf_path))
        partition_gdf_path.mkdir(exist_ok=True)
        partition_gdf.to_file(partition_gdf_path / "partition_geometry.shp")

        # make partition-parcel look-up table; save to file.
        if parcel_gdf is not None:
            parcel_gdf = add_partition_ID_via_sjoin(
                parcel_gdf=parcel_gdf,
                partition_gdf=partition_gdf,
                partition_ID_fieldname="partition_ID",
            )
            parcel_partition_lookup_df = parcel_gdf[[parcel_id_column, "partition_ID"]]
            parcel_partition_lookup_df.to_csv(
                get_partition_file_path(
                    partition_dir=partition_dir, filename="parcel_partition_lookup.csv"
                ),
                index=False,
            )
            self.parcel_gdf = parcel_gdf

        # make partition connected graph; save to file.
        if partition_type == "block":
            partitions_graph = graph_utils.make_partition_connected_graph(
                partition_gdf=partition_gdf
            )
        else:
            partitions_graph = graph_utils.make_partition_connected_graph(
                partition_gdf=partition_gdf, neighbor_algo="buffer"
            )
        with open(
            get_partition_file_path(
                partition_dir=partition_dir, filename="partitions_graph.pkl"
            ),
            "wb",
        ) as out_file:
            pickle.dump(partitions_graph, out_file)

        if make_partition_plot:
            partitions_graph.plot_graph_map(
                save_fig_path=partition_dir / "partition_visualization.png",
                include_graph=False,
                title=osmnx_query_str,
            )

        if make_graph_plot or make_parcel_plot:
            fig, ax = plt.subplots(figsize=(18, 18))

            partitions_graph.plot_graph_map(
                ax=ax,
                fig=fig,
                save_fig_path=partition_dir / "graph_visualization.png",
                title=osmnx_query_str,
            )

            if make_parcel_plot:
                if parcel_gdf is None:
                    raise BaseException(
                        "Must include parcel_gdf in order to plot parcels."
                    )
                parcel_gdf.plot(ax=ax, color="blue", markersize=6)
                plt.savefig(partition_dir / "graph_with_parcels.png")


class parcel_partition_features_builder:
    """
    Before train/test split, store all spatial information.
    Also be able to compute and return
    a new set of spatial feature columns, to be merged
    with the other data. target_column.
    """

    def __init__(
        self,
        parcel_gdf,
        parcel_id_col,
        target_col,
        partition_type,
        client,
        epsg=None,
        cell_width=None,
        osmnx_query_str=None,
        make_parcel_plot=False,
    ):

        parcel_gdf = parcel_gdf[[target_col, parcel_id_col, "geometry"]].to_crs(
            epsg=epsg
        )
        self.parcel_gdf = parcel_gdf
        self.target_col = target_col
        self.parcel_id_col = parcel_id_col
        self.partition_id_col = "partition_ID"

        self.spatial_partition_data_obj = spatial_partition_data_suite(
            client=client,
            partition_type=partition_type,
            cell_width=cell_width,
            parcel_gdf=self.parcel_gdf,
            target_column=self.target_col,
            output_directory="data",
            parcel_id_column=self.parcel_id_col,
            partition_id_col=self.partition_id_col,
            epsg=epsg,
            osmnx_query_str=osmnx_query_str,
            make_parcel_plot=make_parcel_plot,
        )

        self.partition_gdf = self.add_partition_stats_to_partition_gdf()

    def add_partition_stats_to_partition_gdf(
        self,
        train_inds=None,
        test_inds=None,
        stats_cols=None,
    ):
        """
        Make partition statistics (primarily: lead rate per partition) and
        add as new features to each partition in partition_gdf
        """

        partition_gdf = self.spatial_partition_data_obj.partition_gdf.copy()
        parcel_gdf = self.spatial_partition_data_obj.parcel_gdf.copy()
        partition_id_col = self.partition_id_col
        target_col = self.spatial_partition_data_obj.target_column

        # if test/train indices are given, hide the test index data.
        if test_inds is not None:
            parcel_gdf.loc[test_inds, target_col] = np.nan

        # Typically: partition_id_col is 'partition_id'; target_col is 'has_lead'
        stats_by_partition = (
            parcel_gdf[[partition_id_col, target_col]]
            .groupby(partition_id_col)[target_col]
            .agg(["sum", "count"])
        )
        stats_by_partition = stats_by_partition.rename(
            columns={"sum": "total_lead", "count": "total_labels"}
        )

        # This merges on partition_id, which is the index of both df's.
        partition_gdf = partition_gdf.merge(
            stats_by_partition, how="left", left_index=True, right_index=True
        )

        # think about making this better and less weird.
        partition_gdf = partition_gdf.fillna(0)

        lead_risk = np.empty(len(partition_gdf))
        lead_risk[:] = np.nan

        nonzero_labels_indices = partition_gdf["total_labels"] > 0
        total_lead = partition_gdf["total_lead"].loc[nonzero_labels_indices]
        total_labels = partition_gdf["total_labels"].loc[nonzero_labels_indices]
        lead_risk[nonzero_labels_indices] = total_lead / total_labels

        partition_gdf["lead_risk"] = lead_risk
        partition_gdf[partition_id_col] = partition_gdf.index.copy()

        return partition_gdf

    def add_partition_stats_to_each_parcel(self, partition_gdf):
        """Add selected stats cols to each parcel"""
        partition_id_col = self.partition_id_col
        parcel_gdf = self.spatial_partition_data_obj.parcel_gdf.copy()
        if partition_id_col in partition_gdf.columns:
            partition_gdf = partition_gdf.drop(partition_id_col, axis=1)
        parcel_gdf = parcel_gdf.merge(
            partition_gdf,
            how="left",
            left_on=partition_id_col,
            right_index=True,
            suffixes=[None, "_partition"],
        )

        return parcel_gdf

    def add_partition_stats_features(
        self,
        full_test_parcel_gdf,
        train_inds=None,
        test_inds=None,
        stats_cols=["total_labels", "lead_risk"],
    ):
        partition_id_col = self.partition_id_col

        partition_gdf = self.add_partition_stats_to_partition_gdf(
            train_inds, test_inds, stats_cols=stats_cols
        )
        parcel_gdf = self.add_partition_stats_to_each_parcel(
            partition_gdf=partition_gdf
        )

        parcel_gdf = parcel_gdf[
            stats_cols + [self.parcel_id_col, self.partition_id_col]
        ]

        return full_test_parcel_gdf.merge(parcel_gdf, on=self.parcel_id_col, how="left")

    def add_block_stats(parcels_gdf, blocks_gdf, blockID_field, lead_field):
        gdf = parcels_gdf.copy()
        blocks_gdf = blocks_gdf.copy()

        stats_by_block = (
            gdf[[blockID_field, lead_field]]
            .groupby(blockID_field)[lead_field]
            .agg(["sum", "count"])
        )
        stats_by_block.rename(
            columns={"sum": "total_lead", "count": "total_labels"}, inplace=True
        )
        blocks_gdf = blocks_gdf.merge(
            stats_by_block, how="left", left_index=True, right_index=True
        )
        blocks_gdf = blocks_gdf.fillna(0)

        lead_risk = np.empty(len(blocks_gdf))
        lead_risk[:] = np.nan

        nonzero_labels_indices = blocks_gdf["total_labels"] > 0
        total_lead = blocks_gdf["total_lead"].loc[nonzero_labels_indices]
        total_labels = blocks_gdf["total_labels"].loc[nonzero_labels_indices]
        lead_risk[nonzero_labels_indices] = total_lead / total_labels

        blocks_gdf["lead_risk"] = lead_risk
        blocks_gdf[blockID_field] = blocks_gdf.index.copy()

        parcels_gdf = gdf.merge(
            blocks_gdf, on=blockID_field, suffixes=[None, "_blocks"]
        )

        return parcels_gdf, blocks_gdf


def hexagon(l, x, y):
    """
    Create a hexagon centered on (x, y)
    params:

    l: hexagon side length
    x: x-coordinate of the hexagon's center
    y: y-coordinate of the hexagon's center

    returns:
    hexagon = shapely.Polygon, hexagon of length l and center (x,y)
    """
    sides = []
    for angle in range(0, 360, 60):
        side = [
            x + math.cos(math.radians(angle)) * l,
            y + math.sin(math.radians(angle)) * l,
        ]
        sides.append(side)
    hexagon = Polygon(sides)
    return hexagon


def hexgrid(bbox, side):
    """
    Compute hexagon grid centers
    :param bbox: The containing bounding box. The bbox coordinate should be in Webmercator.
    :param side: The size of the hexagons'

    returns:
    grid = list - list of cemter points for hexagonal grid
    """
    grid = []
    xmin = min(bbox[0], bbox[2])
    xmax = max(bbox[0], bbox[2])
    ymin = min(bbox[1], bbox[3])
    ymax = max(bbox[1], bbox[3])

    vstep = math.sqrt(3) * side
    hstep = 1.5 * side

    hskip = math.ceil(xmin / hstep) - 1
    hstart = hskip * hstep

    vskip = math.ceil(ymin / vstep) - 1
    vstart = vskip * vstep

    hend = xmax + hstep
    vend = ymax + vstep

    if vstart - (vstep / 2.0) < ymin:
        vstart_array = [vstart + (vstep / 2.0), vstart]
    else:
        vstart_array = [vstart - (vstep / 2.0), vstart]

    vstart_idx = int(abs(hskip) % 2)

    cx = hstart
    cy = vstart_array[vstart_idx]
    vstart_idx = (vstart_idx + 1) % 2
    while cx < hend:
        while cy < vend:
            grid.append(Point((cx, cy)))
            cy += vstep
        cx += hstep
        cy = vstart_array[vstart_idx]
        vstart_idx = (vstart_idx + 1) % 2

    return grid


def tesselate(
    gdf,
    num_cells_across=6,
    shape="hexagon",
    partition_ID_fieldname="partitionID",
    plot_grid=False,
):
    """
    Returns a grid projected over a gdf, plus the original
    gdf that now has partitionIDs added to each row.

    params:

    gdf = geopandas.GeoDataFrame
    Geodataframe from which the grid will be constructed

    l = float
    Length of the cells in the grid (in degrees)

    # To do :
    # Allow flexibility for different projection types, more shapes

    shape = String ('square' or 'hexagon')
    Dictates grid cell shape, default = 'square'

    returns:

    grid = geopandas.GeoDataFrame Polygon (square or hexagonal)
           grid matching input geometry
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    l = np.abs(xmax - xmin) / num_cells_across
    width, height = l, l
    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax - height
    polygons = []
    if shape == "hexagon":
        for j in tqdm(hexgrid(gdf.total_bounds, l)):
            polygons.append(hexagon(l, j.x, j.y))
    else:
        for i in tqdm(range(cols)):
            Ytop = YtopOrigin
            Ybottom = YbottomOrigin
            for j in range(rows):
                polygons.append(
                    Polygon(
                        [
                            (XleftOrigin, Ytop),
                            (XrightOrigin, Ytop),
                            (XrightOrigin, Ybottom),
                            (XleftOrigin, Ybottom),
                        ]
                    )
                )
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=gdf.crs)
    grid.index = grid.index.rename(partition_ID_fieldname)

    log.info(
        "Tesselate() is performing gpd.sjoin() to remove partitions that contain no parcels."
    )

    gdf_with_partitionIDs = add_partition_ID_via_sjoin(
        parcel_gdf=gdf,
        partition_gdf=grid,
        partition_ID_fieldname=partition_ID_fieldname,
    )

    grid = grid.loc[grid.index.isin(gdf_with_partitionIDs[partition_ID_fieldname])]

    # Reset the grid index so that it starts from 0.
    corrected_idx_name = "corrected_index"
    grid = grid.reset_index()
    grid.index = grid.index.rename("corrected_index")
    grid = grid.reset_index()
    gdf_with_partitionIDs = gdf_with_partitionIDs.merge(
        grid[[partition_ID_fieldname, corrected_idx_name]], on=partition_ID_fieldname
    )
    gdf_with_partitionIDs[partition_ID_fieldname] = gdf_with_partitionIDs[
        corrected_idx_name
    ]

    grid = grid.drop(columns=[corrected_idx_name, partition_ID_fieldname])
    gdf_with_partitionIDs = gdf_with_partitionIDs.drop(columns=[corrected_idx_name])

    return grid, gdf_with_partitionIDs

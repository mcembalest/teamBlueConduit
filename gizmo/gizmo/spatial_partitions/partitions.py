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
import itertools
import copy
from numpy.random import default_rng
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, polygonize
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gizmo.bc_logger import get_simple_logger
from gizmo.spatial_partitions import partition_utils
from gizmo.spatial_partitions import graph_utils
from gizmo.spatial_partitions import misc_utils


# import graph_utils

from pathlib import Path

log = get_simple_logger(__name__)


class PartitionsBuilder:
    def __init__(
        self,
        *,
        parcel_gdf,
        parcel_id_col=None,
        target_col=None,
        copy_all_cols=False,
        copy_cols_list=[],
    ):
        if not "geometry" in parcel_gdf.columns:
            raise BaseException("parcel_gdf must have a `geometry` column!")

        if copy_all_cols:
            keep_cols = parcel_gdf.columns
        else:
            keep_cols = ["geometry"]
            keep_cols += [col for col in [parcel_id_col, target_col] if col is not None]
            keep_cols += copy_cols_list
        self.parcel_gdf = parcel_gdf[keep_cols].copy()

        # Set the index on the parcel_id_col to avoid internal merging ambiguities.
        if parcel_id_col is not None:
            if not parcel_id_col in self.parcel_gdf.columns:
                raise BaseException("parcel_id_col '{}' missing in parcel_gdf!".format(parcel_id_col))
            self.parcel_id_col = parcel_id_col
            self.parcel_gdf = self.parcel_gdf.set_index(parcel_id_col, drop=False)
        else:
            self.parcel_gdf.index.rename("orig_idx", inplace=True)
            self.parcel_id_col = "idx"
            self.parcel_gdf = self.parcel_gdf.reset_index()
            self.parcel_gdf.index.rename("idx", inplace=True)

        self.target_col = target_col

        log.info(
            "New PartitionsBuilder created, using {} of memory".format(
                misc_utils.get_df_memory_usage_str(self.parcel_gdf)
            )
        )

    def Partition(self, partition_type, **kwargs):
        partition = _Partition(
            parcel_gdf=self.parcel_gdf,
            parcel_id_col=self.parcel_id_col,
            target_col=self.target_col,
            partition_type=partition_type,
            **kwargs,
        )

        return partition


class _Partition:
    def __init__(
        self, parcel_gdf, parcel_id_col, partition_type, target_col=None, **kwargs
    ):
        self.parcel_gdf = parcel_gdf.copy()
        self.crs = self.parcel_gdf.crs
        self.partition_type = partition_type
        self.total_bounds = self.parcel_gdf.total_bounds
        self.partition_gdf = None
        self.parcel_id_col = parcel_id_col
        self.target_col = target_col
        self.partition_id_col = "partition_ID"
        self.partition_graph = None
        self.parcel_partition_lookup_df = None
        self.partition_stats_cols = None

        # Build the geodataframe of partitions (self.partition_gdf).
        log.info("Generating the partition map of type '{}'".format(partition_type))

        if partition_type == "hexagon":
            self.partition_gdf, self.parcel_gdf = partition_utils.tesselate(
                gdf=self.parcel_gdf,
                shape="hexagon",
                partition_ID_fieldname=self.partition_id_col,
                **kwargs,
            )

        elif partition_type == "square":
            self.partition_gdf, self.parcel_gdf = partition_utils.tesselate(
                gdf=self.parcel_gdf,
                shape="square",
                partition_ID_fieldname=self.partition_id_col,
                **kwargs,
            )

        elif partition_type == "block":
            self.partition_gdf = partition_utils.get_block_polygons_osmnx(
                crs=self.parcel_gdf.crs, **kwargs
            )
            if "epsg" in kwargs.keys():
                self.parcel_gdf = self.parcel_gdf.to_crs(epsg=kwargs["epsg"])

            self.parcel_gdf = partition_utils.add_partition_ID_via_sjoin(
                parcel_gdf=self.parcel_gdf,
                partition_gdf=self.partition_gdf,
                partition_ID_fieldname=self.partition_id_col,
            )

        elif partition_type == "custom":
            if not "partition_gdf" in kwargs:
                raise BaseException(
                """Partition() also requires a 'partition_gdf' passed to it when
                partition_type=='custom'. partition_gdf is a geodataframe
                which must have the same crs as the parcel data.
                """)
            if not kwargs["partition_gdf"].crs == self.parcel_gdf.crs:
                raise BaseException("""
                CRS of custom partition_gdf and parcel_gdf does not match! They must match
                in order that they overlap correctly.

                partition_gdf.crs: '{}'

                parcel_gdf.crs: '{}'
                """.format(kwargs["partition_gdf"].crs, self.parcel_gdf.crs))
            self.partition_gdf = kwargs['partition_gdf']
            #self.partition_gdf = self.partition_gdf.index.rename(self.partition_id_col)

            if self.partition_id_col in self.parcel_gdf:
                log.debug("Removing partition_id_col included in parcel_gdf!")
                self.parcel_gdf = self.parcel_gdf.drop(self.partition_id_col, axis=1)

            self.parcel_gdf = partition_utils.add_partition_ID_via_sjoin(
                parcel_gdf=self.parcel_gdf,
                partition_gdf=self.partition_gdf,
                partition_ID_fieldname=self.partition_id_col,
            )

        else:
            raise BaseException("Invalid partition type '{}'!".format(partition_type))

        # Rename the partition_gdf index to the partition ID column name.
        self.partition_gdf.index = self.partition_gdf.index.rename(
            self.partition_id_col
        )

        # Build the connected graph upon the partition_gdf.
        log.info("Building the connected graph on the partitions.")
        if partition_type == "block":
            self.partition_graph = graph_utils.make_partition_connected_graph(
                partition_gdf=self.partition_gdf,
            )
        else:
            self.partition_graph = graph_utils.make_partition_connected_graph(
                partition_gdf=self.partition_gdf,
                neighbor_algo="buffer",
            )

        # Build the parcel-partition lookup table (just two columns copied).
        self.parcel_partition_lookup_df = self.parcel_gdf[
            [self.partition_id_col]
        ].copy()

        log.info(
            "New Partition dataframe of type '{}' created, using at least {} of memory".format(
                self.partition_type, misc_utils.get_df_memory_usage_str(self.parcel_gdf)
            )
        )

    def compute_partition_stats(
        self,
        target_col=None,
        binary_feature_cols=[],
        numeric_feature_cols=[],
        train_ind=None,
    ):
        """
        This both updates self.partition_gdf with new stats columns
        AND returns a merged copy of self.parcel_gdf with these new stats columns.
        When used more than once, this function overwrites partition_stats_cols
        that have the same name, so that
        train-test splits don't add new columns indefinitely.
        """

        if target_col is None:
            if self.target_col is None:
                log.info(
                    """No target_col set for compute_partition_stats().
                Alternatively, set target_col upon Partition instantiation."""
                )
            else:
                target_col = self.target_col
                binary_feature_cols.append(target_col)
        else:
            binary_feature_cols.append(target_col)

        # make sure there aren't repeats. 'list(dict.fromkeys())' construction preserves order.
        binary_feature_cols = list(dict.fromkeys(binary_feature_cols))
        numeric_feature_cols = list(dict.fromkeys(numeric_feature_cols))

        repeated_args = list(set(binary_feature_cols).intersection(set(numeric_feature_cols)))

        if len(repeated_args) > 0:
            raise BaseException(
            """
            Repeat arguments between binary and numeric feature cols! '{}'
            """.format(repeated_args)
            )



        # Make internal copy of partition and parcel gdfs.
        partition_gdf = self.partition_gdf.copy()
        parcel_gdf = self.parcel_gdf[
            [self.partition_id_col]
            + binary_feature_cols
            + numeric_feature_cols
        ].copy()


        # if we are limiting stats to train indices, delete all test index labels.
        if target_col:
            if train_ind is not None:
                parcel_gdf.loc[~parcel_gdf.index.isin(train_ind), target_col] = np.nan


        # Compute binary feature stats (always includes target label here).
        # Binary feature stats are fixed to count (total labels),
        # sum (total positives), and rate (sum/count if count!=0 else 0).
        for binary_col in binary_feature_cols:
            # First just compute total_lead and total_labels.
            feature_stats = (
                parcel_gdf.groupby(self.partition_id_col)[binary_col]
                .agg(["sum", "count"])
            )

            # Add the feature rate.
            feature_stats["{}_rate".format(binary_col)] = (
                feature_stats["sum"].div(feature_stats["count"]).fillna(0)
            )

            feature_stats = feature_stats.rename(
                columns={
                    "sum": "{}_sum".format(binary_col),
                    "count": "{}_count".format(binary_col),
                }
            )

            duplicate_cols_to_drop = [
                col_name
                for col_name in feature_stats.columns
                if col_name in partition_gdf.columns
            ]
            partition_gdf = partition_gdf.drop(duplicate_cols_to_drop, axis=1)

            # Merge binary stats on partition_id, which is the shared index of both df's.
            partition_gdf = partition_gdf.join(feature_stats, how="left")


        # Compute numeric feature stats.

        if len(numeric_feature_cols) > 0:

            def percentile(n):
                def percentile_(x):
                    return x.quantile(n)
                percentile_.__name__ = '{:2.0f}%'.format(n*100)
                return percentile_

            numeric_feature_stats = (
                parcel_gdf[[self.partition_id_col] + numeric_feature_cols]
                .groupby(self.partition_id_col)[numeric_feature_cols]
                #.describe()
                #.agg(["mean", "std", "min", "max", "count", "sum", ])
                .agg([
                    "sum", "mean", "std", "median",
                    np.min, np.max,
                    percentile(0.05), percentile(0.25), percentile(.5), percentile(0.75), percentile(.95)
                     ]
                )
            )

            # Rename each colum to {original feat name}_{statistic}
            numeric_feature_stats.columns = [
                "{}_{}".format(col_name, stat_name)
                for (
                    col_name,
                    stat_name,
                ) in numeric_feature_stats.columns.to_flat_index()
            ]

            duplicate_cols_to_drop = [
                col_name
                for col_name in numeric_feature_stats.columns
                if col_name in partition_gdf.columns
            ]
            partition_gdf = partition_gdf.drop(duplicate_cols_to_drop, axis=1)

            # Merge feature stats back into partition_gdf.
            partition_gdf = partition_gdf.join(numeric_feature_stats, how="left")

        self.partition_gdf = partition_gdf

        parcels_with_stats_gdf = pd.merge(
            parcel_gdf,
            partition_gdf.drop(columns=["geometry"]),
            left_on="partition_ID",
            right_index=True,
            how="left",
        )

        return parcels_with_stats_gdf, partition_gdf  # self.get_parcel_gdf_with_stats()

    def get_parcel_gdf_with_stats(self):
        pass

    def cv_splitter(
        self,
        n_splits=5,
        strategy="ShuffleSplit",
        random_state=None,
        plot=False,
        save_plot_path=None,
        **kwargs,
    ):
        n_splits = min(n_splits, len(self.partition_gdf))

        splitter = SpatialSplitter(
            graph=self.partition_graph,
            strategy=strategy,
            random_state=random_state,
            n_splits=n_splits,
            partition_id_col=self.partition_id_col,
            parcel_id_col=self.parcel_id_col,
            parcel_partition_lookup_df=self.parcel_partition_lookup_df,
            **kwargs,
        )

        log.info(
            "Made Partition.cv_splitter instance with n_splits={}; strategy={}".format(
                n_splits, strategy
            )
        )

        if plot or (save_plot_path is not None):
            splitter.plot_folds()

        return splitter

    def plot(
        self,
        figsize=(6, 6),
        show_parcels=False,
        max_include_parcels=5 * 10 ** 3,
        show_graph=False,
        title="Partition plot",
        column=None,
        save_fig_path=None,
        cbar_range=None,
        cmap=None,
        *args,
        **kwargs,
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # remove all ticks
        plt.xticks([])
        plt.yticks([])

        # Column is the scalar feature to include in the heatmap plot.
        if column is not None:
            column_name = column
            if not column_name in self.partition_gdf.columns:
                raise BaseException(
                    """The column '{}' does not yet exist in the partition_gdf!
                        You might try including the column upon PartitionBuilder creation,
                        using the copy_cols_list or copy_all_cols arguments, or by creating
                        the column after by calling <Partition>.compute_partition_stats().
                    """.format(
                        column_name
                    )
                )
            else:
                column_data = self.partition_gdf[[column]]
            cbar = True
        else:
            column_name = None
            column_data = None
            cbar = False

        self.partition_graph.plot_graph_map(
            ax=ax,
            fig=fig,
            include_graph=show_graph,
            save_fig_path=save_fig_path,
            title=title,
            column_name=column_name,
            column_data=column_data,
            cbar=cbar,
            cbar_range=cbar_range,
            cmap=cmap,
            *args,
            **kwargs,
        )

        if show_parcels:
            if len(self.parcel_gdf) > max_include_parcels:
                log.info(
                    "Maximum number of parcels to plot ({}) exceeded; plotting just that number.".format(
                        max_include_parcels
                    )
                )
                self.parcel_gdf.sample(max_include_parcels).plot(
                    ax=ax, color="blue", markersize=4, alpha=0.7
                )
            else:
                self.parcel_gdf.plot(ax=ax, color="blue", markersize=4, alpha=0.7)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 6
        ax.set_title(title, size=9)

        return ax


class SpatialSplitter:
    def __init__(
        self,
        *,
        graph,
        parcel_partition_lookup_df,
        partition_id_col,
        parcel_id_col,
        n_splits=5,
        random_state=None,
        strategy="ShuffleSplit",
        shuffle=False,
        test_size=0.25,
    ):
        if n_splits < 2:
            raise BaseException(
                "'n_splits', entered as {}, must be at least 2.".format(n_splits)
            )

        self.graph = copy.deepcopy(graph)
        self.parcel_partition_lookup_df = parcel_partition_lookup_df.copy()
        self.parcel_id_col = parcel_id_col
        self.partition_id_col = partition_id_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.strategy = strategy
        self.shuffle = shuffle
        self.test_size = test_size

    def split(self, X=None, y=None, groups=None, output_level="parcel"):
        partition_ID_array = self.graph.gdf.index.to_numpy()
        splitter = None

        log.info(
            "SpatialSplitter.split() is running with strategy '{}'".format(
                self.strategy
            )
        )

        # 'split with replacement'; 'overlapping splits'
        if self.strategy == "ShuffleSplit":
            rs = ShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            splitter = rs.split(partition_ID_array)

        # 'non-overlapping splits'
        elif self.strategy == "KFold":
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
            splitter = kf.split(partition_ID_array)

        elif self.strategy == "spatial-TBA":
            raise BaseException(
                "Strategy {} still under construction".format(self.strategy)
            )

        else:
            raise BaseException("Strategy '{}' not valid.".format(self.strategy))

        for train_partition_idx, test_partition_idx in splitter:

            if output_level == "partition":
                yield train_partition_idx, test_partition_idx

            elif output_level == "parcel":
                df = self.parcel_partition_lookup_df
                train_parcel_idx = df[
                    df[self.partition_id_col].isin(train_partition_idx)
                ].index
                test_parcel_idx = df[
                    df[self.partition_id_col].isin(test_partition_idx)
                ].index
                yield train_parcel_idx, test_parcel_idx

            else:
                raise BaseException("Invalid output_level: {}".format(output_level))

    def plot_folds(self, max_folds_plotted=12):
        n_cols = min(3, self.n_splits)
        n_folds_to_plot = min(max_folds_plotted, self.n_splits)

        # choose number of rows such that it fills a complete grid.
        n_rows = int((n_folds_to_plot + (n_cols - (n_folds_to_plot % n_cols))) / n_cols)
        n_folds_to_plot = n_rows * n_cols

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey=True,
            figsize=(8, 3.5 * n_rows),
            dpi=150,
        )
        plt.rcParams["font.family"] = "serif"

        fig.suptitle("Spatial-cv strategy: {}".format(self.strategy), size=9)
        # fig = plt.figure(figsize=(12, 8 * n_rows))
        label_patches = {
            "train": mpatches.Patch(color="darkcyan", label="train"),
            "test": mpatches.Patch(color="paleturquoise", label="test"),
        }
        # spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)
        for ax in axs.ravel():
            ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)
        row_col_gen = [range(n_rows), range(n_cols)]
        splitter = self.split(output_level="partition")
        for i, ((row, col), (train_ind, test_ind)) in enumerate(
            zip(itertools.product(*row_col_gen), splitter)
        ):

            ax = axs[row, col] if n_rows > 1 else axs[col]
            ax.set_xticks([])
            ax.set_yticks([])
            self.plot_single_fold(
                ax=ax,
                train_partition_IDs=train_ind,
                test_partition_IDs=test_ind,
                label_patches=label_patches,
            )
            ax.set_title("Fold {}".format(i), size=8)
            ax.set_axis_on()
            if i == 0:
                first_ax = ax
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(
            handles=[label_patches["train"], label_patches["test"]],
            loc="upper right",
            prop={"size": 9},
        )
        plt.tight_layout()

    def plot_single_fold(
        self,
        *,
        ax,
        train_partition_IDs,
        test_partition_IDs,
        label_patches,
    ):
        train_color = label_patches["train"].get_facecolor()
        test_color = label_patches["test"].get_facecolor()
        self.graph.gdf.plot(ax=ax, facecolor="white", edgecolor="black")
        self.graph.gdf.iloc[train_partition_IDs].plot(
            ax=ax, facecolor=train_color, edgecolor="black", label="train"
        )
        self.graph.gdf.iloc[test_partition_IDs].plot(
            ax=ax, facecolor=test_color, edgecolor="black", label="test"
        )

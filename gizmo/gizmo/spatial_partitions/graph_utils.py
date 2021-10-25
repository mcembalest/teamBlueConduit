import geopandas
import numpy as np
from shapely.strtree import STRtree
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import networkx
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from gizmo.bc_logger import get_simple_logger


log = get_simple_logger(__name__)


class stopwatch:
    def __init__(self, print_while_running=True, name=""):
        self.start_time = time.time()
        self.name = name
        self.print_while_running = print_while_running
        self.most_recent_time = self.start_time
        self.log = [(self.start_time, "start", 0)]

    def lap(self, message, print_timestamp=False):
        now = time.time()
        most_recent_duration = now - self.most_recent_time
        self.log.append((now - self.start_time, message, most_recent_duration))
        if self.print_while_running or print_timestamp:
            print(
                "{} stopwatch: {} after {:.6f}s".format(
                    self.name, message, most_recent_duration
                )
            )
        self.most_recent_time = now


class neighbors_list:
    def __init__(self, from_polygon_df=None, from_point_df=None):
        self.gdf = None
        if not from_polygon_df is None:
            self.load_from_polygons(gdf=from_polygon_df)
        elif not from_point_df is None:
            self.load_from_points(gdf=from_points_df)
        else:
            raise BaseException("neighbors_list requires an input gdf.")
        self.stats_log = []
        self.gdf["neighbors"] = ""
        self.gdf["neighbors_count"] = np.ones(len(self.gdf)) * -1

    def load_from_polygons(self, gdf):
        df = gdf.copy()
        self.gdf = df
        self.gdf["neighbors"] = ""
        self.gdf["x"] = df.geometry.representative_point().x
        self.gdf["y"] = df.geometry.representative_point().y

        # compute size measures for each polygon.
        self.gdf["envelope"] = df.geometry.envelope
        envelope_height = df.envelope.bounds.maxy - df.envelope.bounds.miny
        envelope_width = df.envelope.bounds.maxx - df.envelope.bounds.minx
        self.gdf["envelope_max_dist"] = np.maximum(envelope_height, envelope_width)
        self.gdf["envelope_min_dist"] = np.minimum(envelope_height, envelope_width)
        self.gdf["env_min_over_2"] = self.gdf["envelope_min_dist"] / 2
        self.gdf["env_max_over_2"] = self.gdf["envelope_max_dist"] / 2

    def load_from_points(self, gdf):
        df = gdf.copy()
        self.gdf = df
        self.gdf["x"] = df.geometry.x
        self.gdf["y"] = df.geometry.y
        # maybe use kdTree here to compute mean and std of distance to nearest points.

    # Neighbors list strings expected: ''; '1'; '1,2,3'.
    def get_neighbors_list_from_str(self, neighbors_str):
        return [] if neighbors_str == "" else [int(i) for i in neighbors_str.split(",")]

    def get_str_from_neighbors_list(self, neighbors_list):
        return ",".join(str(i) for i in np.array(neighbors_list))

    def set_gdf_neighbors(self, index, neighbors_list):
        self.gdf.loc[index, "neighbors_count"] = len(neighbors_list)
        self.gdf.loc[index, "neighbors"] = self.get_str_from_neighbors_list(
            neighbors_list
        )

    # Uses shapely.disjoint, runs much more slowly so better for sub-2,000 node graphs.
    def add_neighbors_disjoint_buffer(self, buffer):
        df = self.gdf.copy()
        if buffer > 0:
            df.geometry = df.buffer(buffer)

        for index, row in df.iterrows():
            neighbors = df[~df.geometry.disjoint(row.geometry)].index.copy()
            neighbors = list(neighbors)
            if index in neighbors:
                neighbors.remove(index)
            # df.loc[index, "neighbors"] = ','.join(str(i) for i in list(neighbors))
            self.set_gdf_neighbors(index=index, neighbors_list=list(neighbors))

    # Uses shapely STRtree for high speed but often yields false positives.
    def add_STRtree_neighbors(self, buffer=0, watch=stopwatch()):
        df = self.gdf.copy()
        if buffer > 0:
            df.geometry = df.buffer(buffer)

        tree = STRtree(df.geometry)
        watch.lap("Built STRtree")
        index_by_id = dict((id(geo), i) for i, geo in enumerate(df.geometry))
        watch.lap("Built STRtree index")

        for index, row in df.iterrows():
            neighbors = [
                (index_by_id[id(pt)]) for pt in tree.query(df.iloc[index].geometry)
            ]
            neighbors.remove(
                index
            )  # don't include a node's own index in its neighbors list.
            # df.loc[index, 'neighbors'] = self.get_str_from_neighbors_list(neighbors)
            self.set_gdf_neighbors(index=index, neighbors_list=neighbors)
        watch.lap("Performed STRtree neighbor lookups")

    # This code is to check the local neighbors list for
    # false positives and remove them.
    def remove_non_touching_neighbors(self):
        df = self.gdf.copy()

        # Delete entire neighbors column before replacing it with corrected version.
        self.gdf["neighbors"] = ""
        for index, row in df.iterrows():
            neighbors = self.get_neighbors_list_from_str(row["neighbors"])
            filt = df.geometry.iloc[neighbors].touches(df.geometry.iloc[index])
            # df.loc[index, 'neighbors'] = self.get_str_from_neighbors_list(np.array(neighbors)[filt])
            self.set_gdf_neighbors(
                index=index, neighbors_list=np.array(neighbors)[filt]
            )

    def add_neighbors_KDBall(self, r, print_stopwatch=True):
        watch = stopwatch(print_while_running=print_stopwatch)
        X = self.gdf[["x", "y"]]
        # building tree
        # querying

    def add_neighbors_KD_nearest(self, k, watch=stopwatch()):
        X = self.gdf[["x", "y"]]
        tree = sklearn.neighbors.BallTree(X)
        watch.lap("built BallTree")
        dist, ind = tree.query(X, k=k)
        watch.lap("queried the tree")

        # ind is a list of lists of neighbors indexes.
        ind_array = np.array(ind)
        for index, neighbors_list in enumerate(ind_array):
            self.set_gdf_neighbors(index, neighbors_list)

    # e.g. mean and std number of neighbors per row.
    def get_neighbors_stats(self, live_print=False, function=None, duration=None):
        stats_df = pd.DataFrame(
            {
                "nodes": [len(self.gdf)],
                "neighbors: count": self.gdf.neighbors_count.sum(),
                "mean": self.gdf.neighbors_count.mean(),
                "std": self.gdf.neighbors_count.std(),
                "max": self.gdf.neighbors_count.max(),
            }
        )
        if live_print:
            print("Distribution of number of neighbors for each node:")
            display(stats_df.round(2))

    def add_neighbors_data(self, gdf):
        df = gdf.copy()
        df["neighbors"] = self.gdf.neighbors
        return df


class gdf_and_connected_graph:
    def __init__(self, gdf):
        if not "neighbors" in gdf.columns:
            raise BaseException(
                """
            The geodataframe has no 'neighbors' column yet.
            You need to add the column first, using the
            neighbors_utils.py functions."""
            )
        self.gdf = gdf[["geometry", "neighbors"]].copy()
        self.graph = None
        self.build_graph()

    def build_graph(self):
        # Neighbors list strings expected: ''; '1'; '1,2,3'.
        def get_list_from_str(neighbors_str):
            return (
                []
                if neighbors_str == ""
                else [int(i) for i in neighbors_str.split(",")]
            )

        self.graph = networkx.Graph()
        edge_list = []

        for node_index, neighbors_list_str in enumerate(self.gdf["neighbors"]):
            if neighbors_list_str == "":  # add nodes that have no neighbors.
                self.graph.add_node(node_index)
            else:
                for neighbor_index in get_list_from_str(neighbors_list_str):
                    if neighbor_index != node_index:
                        edge_list.append((node_index, neighbor_index))

        self.graph.add_edges_from(edge_list)

    def compute_assign_component_IDs(self):
        components = networkx.connected_components(self.graph)
        # print([(i, x) for (i, x) in enumerate(components)])
        # print(np.sort(np.array(list(self.graph.nodes))))
        components = networkx.connected_components(self.graph)
        comp_dict = {idx: comp for idx, comp in enumerate(components)}
        attr = {n: comp_id for comp_id, nodes in comp_dict.items() for n in nodes}

        networkx.set_node_attributes(self.graph, attr, "component_id")

        nodes_with_component_id_dict = networkx.get_node_attributes(
            self.graph, "component_id"
        )
        component_df = pd.DataFrame.from_dict(
            nodes_with_component_id_dict, orient="index", columns=["component_id"]
        )

        self.gdf = self.gdf.merge(component_df, left_index=True, right_index=True)

    def get_gdf(self):
        return self.gdf

    def add_island_IDs(self, gdf):
        gdf = gdf.copy()
        self.compute_assign_component_IDs()
        gdf["island_ID"] = self.gdf.component_id
        # gdf['island_ID'] = gdf.astype({'island_ID': int})

        return gdf

    # this has dimensions NxN where N is number of nodes.
    def get_adjacency_matrix(self, print_report=False):
        pass

    # return Morris nbs object
    def get_nbs_object(self):
        nbs = {}
        nbs["N"] = len(list(self.graph.nodes))
        edge_list = list(self.graph.edges)
        nbs["N_edges"] = len(edge_list)
        # print(edge_list)
        nbs["node1"] = [edge[0] for edge in edge_list]
        nbs["node2"] = [edge[1] for edge in edge_list]

        return nbs

    def plot_graph_map(
        self,
        include_partitions=True,
        include_graph=True,
        ax=None,
        fig=None,
        figsize=None,
        save_fig_path=None,
        max_edges=10 ** 4,
        title="Partition Map",
        column_name=None,
        column_data=None,
        cbar=False,
        cbar_range=None,
        edgecolor=None,
        cmap="viridis",
        *args,
        **kwargs,
    ):

        if ax is None:
            if not figsize:
                figsize = (18, 18)
            fig, ax = plt.subplots(figsize=figsize)
            # remove all ticks
            plt.xticks([])
            plt.yticks([])

        if include_partitions:
            gdf = self.gdf.copy()

            if column_name is None:
                gdf["random_feature"] = np.random.random(len(gdf))
                gdf.plot(ax=ax, column="random_feature", *args, **kwargs)
                title += "\nColor is random feature for readability"

            else:
                gdf_plus_col = gdf.merge(column_data, left_index=True, right_index=True)

                if cbar:
                    if cbar_range is not None:
                        norm = mpl.colors.Normalize(cbar_range[0], cbar_range[1])
                    else:
                        norm = mpl.colors.Normalize(
                            gdf_plus_col[column_name].min(),
                            gdf_plus_col[column_name].max(),
                        )
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="5%", pad=0.1)
                    gdf_plus_col.plot(
                        edgecolor=edgecolor,
                        ax=ax,
                        cax=cax,
                        column=column_name,
                        cmap=cmap,
                        norm=norm,
                        legend=True,
                        legend_kwds={"label": column_name, "orientation": "horizontal"},
                        *args,
                        **kwargs,
                    )

        if include_graph:

            nbs = self.get_nbs_object()
            node1, node2 = nbs["node1"], nbs["node2"]
            gdf = self.gdf.copy()

            x0 = np.array(gdf.centroid[node1].x)
            y0 = np.array(gdf.centroid[node1].y)
            x1 = np.array(gdf.centroid[node2].x)
            y1 = np.array(gdf.centroid[node2].y)

            x0x1y0y1_matrix = np.column_stack((x0, x1, y0, y1))

            for index, row in enumerate(x0x1y0y1_matrix):
                if max_edges and index > max_edges:
                    break
                ax.plot(row[[0, 1]], row[[2, 3]], linewidth=1, color="white", alpha=0.9)
        if title:
            ax.set_title(title)

        if save_fig_path:
            if fig is None:
                raise BaseException("Need to pass fig as argument.")
            fig.savefig(save_fig_path)
        else:
            pass
            # plt.show()


### These functions are meant to be called from outside the script ###

# Takes a gdf and returns the same with an added list of neighbors -- contiguity criterion.
def add_contiguous_neighbors(gdf):
    watch = stopwatch(name="add_contiguous_neighbors")
    nbs_list = neighbors_list(from_polygon_df=gdf)
    nbs_list.add_STRtree_neighbors(watch=watch)
    # nbs_list.add_neighbors_disjoint_buffer(buffer=10)
    nbs_list.remove_non_touching_neighbors()
    log.debug("Removed non-touching neighbors")
    nbs_list.get_neighbors_stats(live_print=False)
    gdf = nbs_list.add_neighbors_data(gdf)

    return gdf


# Takes a gdf and returns the same with an added list of neighbors -- k-nearest criterion.
def add_k_nearest_neighbors(gdf, k):
    gdf = gdf.copy()
    orig_index = gdf.index.copy()
    gdf.reset_index(inplace=True)

    watch = stopwatch(name="add_contiguous_neighbors")
    nbs_list = neighbors_list(from_polygon_df=gdf)
    nbs_list.add_neighbors_KD_nearest(k, watch=watch)
    gdf = nbs_list.add_neighbors_data(gdf)

    gdf = gdf.set_index(orig_index)

    return gdf


# Takes a gdf and returns the same with an added list of neighbors -- disjoint + buffer criterion.
def add_buffer_neighbors(gdf, buffer=0):
    watch = stopwatch(name="add_buffer_neighbors")
    gdf = gdf.copy()
    # gdf['geometry'] = gdf.geometry.buffer(buffer)
    nbs_list = neighbors_list(from_polygon_df=gdf)
    # nbs_list.add_STRtree_neighbors(watch=watch)
    nbs_list.add_neighbors_disjoint_buffer(buffer=buffer)
    # watch.lap("added neighbors - disjoint buffer")
    nbs_list.get_neighbors_stats(live_print=False)
    # nbs_list.remove_non_touching_neighbors(); watch.lap('Removed non-touching neighbors')
    nbs_list.get_neighbors_stats(live_print=False)
    gdf = nbs_list.add_neighbors_data(gdf)

    return gdf


# Takes a gdf that has a neighbors list and adds an island ID to each row.
def add_island_IDs_from_neighbors(gdf):
    gdf = gdf.copy()
    orig_index = gdf.index.copy()
    gdf = gdf.reset_index(drop=True)
    watch = stopwatch(name="add island IDs from graph")
    graph = gdf_and_connected_graph(gdf=gdf)
    watch.lap("built graph")
    gdf = graph.add_island_IDs(gdf)
    watch.lap("computed island IDs")
    # print(networkx.to_edgelist(graph.graph))

    gdf = gdf.set_index(orig_index)

    return gdf


# Get Morris-style nbs-object from gdf that already has neighbors column.
def nbs_dict_from_neighbors(gdf):
    watch = stopwatch(name="add island IDs from graph")
    graph = gdf_and_connected_graph(gdf=gdf)
    watch.lap("built graph")
    nbs = graph.get_nbs_object()
    watch.lap("computed nbs object")

    return nbs


def make_partition_connected_graph(
    partition_gdf, neighbor_algo="touching", buffer_fract=None
):
    """Returns networkx graph; assumes partitions share borders.

    neighbor_algo = touching or buffer
    """
    partition_gdf = partition_gdf.copy()

    if neighbor_algo == "buffer":
        if buffer_fract == None:
            buffer_fract = 0.01  # buffer = 1% of perimeter length
        median_perimeter = partition_gdf.geometry.length.describe()["50%"]
        buffer_distance = median_perimeter * buffer_fract
        partition_gdf = add_buffer_neighbors(partition_gdf, buffer_distance)
    else:
        partition_gdf = add_contiguous_neighbors(partition_gdf)

    gdf_graph = gdf_and_connected_graph(gdf=partition_gdf)

    return gdf_graph


# Pipeline unit function, barely maintained.
def test_graph(printlog=False):
    blocks_df = map_simulation.get_map_with_4_blocks()

    blocks_df["neighbors"] = neighbors_builder.neighbors_builder(
        from_polygon_df=blocks_df
    ).compute_neighbors_disjoint_buffer(buffer=0)

    # create graph object.
    graph = gdf_and_connected_graph(gdf=blocks_df)

    graph.compute_assign_component_IDs()

    graphing_funcs.pretty_plot_map(
        graph.get_gdf(), show_field_value=True, color_field="component_id"
    )

    return graph.gdf

import pandas as pd
import numpy as np
import joblib as jl
import geopandas as gpd
import shapely
import fiona
from service_line_pipeline import data as slirp_data

import plotly.graph_objects as go
import plotly.express as px


from gizmo.spatial_partitions.partitions import PartitionsBuilder


def generate_aggregate_map_layer(
    micro_data,
    id_column,
    target_column="has_lead",
    partition_type=("hexagon", 20),
    agg_columns={},
    save_layer=False,
):
    if type(micro_data) == gpd.GeoDataFrame:
        partitions_builder = PartitionsBuilder(
            parcel_gdf=micro_data.to_crs(epsg=4326),
            parcel_id_col=id_column,
            target_col=target_column,
            copy_all_cols=False,
            copy_cols_list=list(agg_columns.keys()),
        )
        part_validate = partition_type[1]
        if type(part_validate) == int:
            partitions = partitions_builder.Partition(
                partition_type=partition_type[0], num_cells_across=partition_type[1]
            )
        elif type(part_validate) == gpd.GeoDataFrame:
            partitions = partitions_builder.Partition(
                partition_type=partition_type[0],
                partition_gdf=partition_type[1].to_crs(micro_data.crs),
            )
        else:
            print("Custom partition dataset is not a GeoDataFrame")
        del part_validate
        micro_gdf, macro_gdf = partitions.compute_partition_stats(
            numeric_feature_cols=list(agg_columns.keys())
        )
        # drop unwanted columns
        drop_columns = []
        for column in agg_columns:
            targets = [col for col in macro_gdf.columns if column in col]
            stats_to_keep = agg_columns[column]
            for target in targets:
                if target not in [column + "_" + stat for stat in stats_to_keep]:
                    drop_columns.append(target)
        macro_gdf.drop(columns=drop_columns, inplace=True)
        # save output or return geodataframe
        if not save_layer:
            return macro_gdf
        else:
            if "geojson" in save_layer:
                macro_gdf.to_file(save_layer, driver="GeoJSON")
            else:
                macro_gdf.to_file(save_layer)


def print_program_summary(
    client_data,
    target_column="has_lead",
    hvac_column="hvac_visit",
    slr_column="replaced",
    pred_column="y_score",
    verified_target="verified_lead",
):
    # Important stats
    total_hv_visits = client_data[hvac_column].value_counts()[True]
    total_sl_visits = client_data[slr_column].value_counts()[True]
    # total_conf_nonlsl = toledo_data.has_lead.value_counts()[False]
    total_conf_lsl = client_data[verified_target].value_counts()[True]
    total_lsl = int(client_data[pred_column].sum())
    print("===== Program Summary =====")
    print(
        f"Inspections Completed:  {total_hv_visits}\nLSLs Replaced:         {total_sl_visits}\nTotal Confirmed LSL: {total_conf_lsl}\nTotal Estimated LSL: {total_lsl}"
    )


def plot_sl_material_pie_chart(
    client_data,
    pred_column="y_score",
    thresholds={
        "Known Lead": (1, 1, True),
        "Likely Lead": (0.7, 1, False),
        "Potentially Lead": (0.3, 0.7, True),
        "Likely Non-Lead": (0.0, 0.3, False),
        "Known Non-Lead": (0, 0, True),
    },
):
    inv_bdn = pd.DataFrame(
        pd.Series(
            {
                x: client_data[pred_column]
                .between(thresholds[x][0], thresholds[x][1], inclusive=thresholds[x][2])
                .value_counts()[True]
                for x in thresholds
            }
        )
    ).reset_index()

    inv_bdn.rename(columns={"index": "Material", 0: "Service Lines"}, inplace=True)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=inv_bdn["Material"],
                values=inv_bdn["Service Lines"],
                marker_colors=["#d60000", "#ff7070", "#fcc203", "#47ff0a", "#2cc98d"],
                # title="Service Line Inventory Breakdown",
                # pull=[0, 0.2, 0, 0, 0],
                # Second, make sure that Plotly won't reorder your data while plotting
                sort=False,
            )
        ]
    )
    fig.update_traces(
        hoverinfo="label+value", textposition="inside", textinfo="percent"
    )
    fig.update_layout(
        # height=600,
        margin=dict(
            l=10,
            r=60,
            t=40,
            b=40,
            autoexpand=True,
        ),
        showlegend=True,
        legend_title="Inventory Breakdown",
        paper_bgcolor="#184E84",
        plot_bgcolor="#184E84",
        font_color="#ffffff",
        font_size=24,
        uniformtext_minsize=20,
    )  # ,
    return fig


def plot_predictions_histogram(data, pred_column="y_score"):
    # plots a 10-bin histogram in yellow with blue background
    fig = px.histogram(
        data, x=pred_column, nbins=10, color_discrete_sequence=["#F4D396"]
    )
    fig.update_layout(
        title="Service Lines by Predicted Risk of Lead",
        paper_bgcolor="#184E84",
        plot_bgcolor="#184E84",
        font_color="#ffffff",
        height=240,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=20,
            autoexpand=True,
        ),
        xaxis_title="Predicted Probability",
        yaxis_title="Service Lines",
    )
    return fig


def plot_dist_by_target(data, dist_column, target_column="has_lead"):
    cmap = ["#F4D396", "#5CA2E6"]
    counts = pd.DataFrame({dist_column: data[dist_column].sort_values().unique()})
    for i in counts.index:
        has_lead_counts = data[data[dist_column] == counts[dist_column][i]][
            target_column
        ]
        has_lead_counts = has_lead_counts.value_counts()
        for z in has_lead_counts.index:
            counts.at[i, z] = has_lead_counts[z]

    fig = go.Figure()
    for i, col in enumerate(counts.columns[1:]):
        fig.add_trace(
            go.Scatter(
                x=counts[dist_column],
                y=counts[col],
                mode="lines",
                name=col,
                line=dict(width=0.5, color=cmap[i]),
                stackgroup="one",  # define stack group
            )
        )
    fig.update_layout(
        title=f"Service Lines by {dist_column}",
        paper_bgcolor="#184E84",
        plot_bgcolor="#184E84",
        font_color="#ffffff",
        height=240,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=20,
            autoexpand=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            # type="date",
            # range=(xmin, xmax)
        ),
        hovermode="x",
    )

    return fig

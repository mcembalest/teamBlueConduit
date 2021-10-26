#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Union
import os
from pathlib import Path
import boto3
from gizmo.bc_logger import get_simple_logger
import fiona

import json
import hashlib

from collections import defaultdict

from functools import lru_cache

import io
import zipfile

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError

log = get_simple_logger(__name__)


def to_processed_dir(getdata_func):

    def wrapper(*args, **kwargs):
        gdf = getdata_func(*args, **kwargs)

        Path("processed_data").resolve().mkdir(parents=True, exist_ok=True)
        processed_path = (Path("processed_data")).resolve()
        gdf.to_feather(
            os.path.join(processed_path, str(getdata_func.__name__) + ".feather")
        )
        return gdf

    return wrapper

def to_postgres(
    gdf,
    client_name: str,
    username: str = "postgres",
    password: str = "password",
    hostname: str = "localhost",
    port: int = 5432,
    database_name: str = "bc",
):

    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"
    )

    gdf.to_postgis(f"{client_name}_sl_records", engine, if_exists="replace")


def to_pg_json_data(
    gdf,
    client_name: str,
    username: str = "postgres",
    password: str = "password",
    hostname: str = "localhost",
    port: int = 5432,
    database_name: str = "bc_geodata_app_dev",
):
    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"
    )

    gdf["clientname"] = client_name

    try:
        with engine.connect() as connection:

            for ix, ser in gdf.iterrows():

                # TODO: Flint is throwing errors here (unescaped parentheses in data)

                json_str = ser.to_json(default_handler=str)
                digest = hashlib.md5(json_str.encode("utf-8")).hexdigest()
                query = f"""
                    INSERT INTO public.bc_data_app_clientdata(digest, client_name, data)
                    VALUES ('{digest}', '{client_name}', '{json_str}');
                """
                # TODO: Getting TypeError: 'dict' object does not support indexing
                # Need to escape query string using sqlalchemy.text?
                connection.execute(query)

    except OperationalError as ex:
        log.error("There was an issue connecting to the database.")


def from_postgres(
    client_name: str,
    username: str = "postgres",
    password: str = "password",
    hostname: str = "localhost",
    port: int = 5432,
    database_name: str = "bc",
):
    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"
    )
    return gpd.read_postgis(
        f"select * from {client_name}_sl_records", engine, geom_col="geometry"
    )


def check_db_for_client_data(
    client_name: str,
    username: str = "postgres",
    password: str = "password",
    hostname: str = "localhost",
    port: int = 5432,
    database_name: str = "bc",
):
    db_has_table = False

    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"
    )

    check_query_string = f"""
        SELECT EXISTS (
            SELECT FROM pg_tables
            WHERE  schemaname = 'public'
            AND    tablename  = '{client_name}_sl_records'
        );
    """

    try:
        with engine.connect() as connection:
            result_iterator = connection.execute(check_query_string)
            first_result = next(result_iterator)
            db_has_table = first_result[0]
    except OperationalError as ex:
        log.error("There was an issue connecting to the database.")

    return db_has_table


def get_data(
    client_name: str = "",
    data_dir: Union[Path, str] = "data",
    data_config=None,
    **kwargs,
) -> pd.DataFrame:

    db_has_table = check_db_for_client_data(client_name)

    if db_has_table:
        try:
            client_data = from_postgres(client_name)
            return client_data
        except Exception as e:
            log.error("Error")

    # This is effectively a switch statement to dynamically select the correct
    # data loader.
    client_data = {
        "flint": get_flint_data,
        "toledo": get_toledo_data,
        "halifax": get_halifax_data,
        "benton_harbor": get_benton_harbor_data,
        "detroit": get_detroit_data,
        "trenton": get_trenton_data,
        "fayetteville": get_fayetteville_data,
        "charlotte": get_charlotte_data,
        "suez": get_suez_data,
        # "helios": get_helios_data,
    }[client_name](data_dir=str(data_dir), **data_config)

    # TODO Validate that data has minimal columns and values
    # client_data.apply(ServiceLineData.validate, axis='rows')

    # Persist data to database, if geodataframe, and if running
    try:
        to_postgres(client_data, client_name)
        to_pg_json_data(client_data, client_name)
    except Exception as e:
        # TODO Don't catch base exception for everything
        log.error(
            "Failed to persist client data to postgres."
            " This is likely for one of the following three reasons."
            " 1) Database is not running."
            " 2) Database details are not correct."
            " 3) Startup script failed to run. Try `docker-compose down &&"
            " docker-compose up -d postgres`"
            " 4) The client data function does not return a GeoDataFrame"
            " (required for `gdf.postgis`)."
        )

    return client_data


# TODO Make this generic, so we can fetch models too without getting confused
def get_source(
    source_filename: str = "", data_dir: Union[Path, str] = "", client_bucket: str = ""
):
    """Take a source data filename in an s3 bucket and return the path to a
    local copy. If the file doesn't exist locally, download it."""

    local_source_filepath = Path(data_dir).resolve() / Path(source_filename)
    if not local_source_filepath.exists():
        # Defensively make parent data directory if necessary
        local_source_filepath.parents[0].mkdir(parents=True, exist_ok=True)

        boto3.resource("s3").Bucket(client_bucket).download_file(
            source_filename, str(local_source_filepath)
        )

    return local_source_filepath


# TODO Ensure strict folder matching
# TODO Could I call get_source from in here to DRY up code?
def download_folder(
    source_folder=None,
    data_dir: Union[Path, str] = "data",
    client_bucket: str = "",
):

    sink_directory = Path(data_dir) / source_folder
    sink_directory.mkdir(parents=True, exist_ok=True)

    s3_resource = boto3.resource("s3")

    bucket = s3_resource.Bucket(client_bucket)
    obj_list = [obj for obj in bucket.objects.filter(Prefix=source_folder)]

    for obj in bucket.objects.filter(Prefix=source_folder):

        local_filepath = sink_directory / str(obj.key).split("/")[-1]

        if not local_filepath.exists():
            bucket.download_file(obj.key, str(local_filepath))

    return sink_directory


def get_osm_source(source_folder, data_dir, location_string):
    """
    Download OSM files from S3 into data_dir, returns the filepath.
    Either make a folder in data_dir specific to the city, and put the data there;
    or just change the filename.
    """

    return


def get_shapefile_source(source_folder, data_dir, client_bucket):

    # We first download the shapefile directory
    local_directory = download_folder(source_folder, data_dir, client_bucket)

    # We then get the path of the shapefile
    # TODO -- implement `select`
    target_shapefilename = list(
        filter(lambda fp: fp.split(".")[-1] == "shp", os.listdir(local_directory))
    )[0]

    target_shapefilepath = local_directory / target_shapefilename

    return target_shapefilepath

@to_processed_dir
def get_benton_harbor_data(
    data_dir: Union[Path, str] = "data/benton_harbor",
    bucket_name: str = "bc-benton-harbor",
    gdb_folder: str = "2021-02-24 Benton_Harbor.gdb",
    active_accounts: str = "Active accounts 4-6-2021.xlsx",
    active_shutoffs: str = "Active shut-offs 4-6-2021.xlsx",
    inactive_accounts: str = "Inactive accounts 4-6-2021.xlsx",
    full_replacements: str = "Full Water Service Replacements - Highlighted Blue on Summary Map.xlsx",
    inspections: str = "2018 Pilot Grant Replacement - Service Verifications.xlsx",
    recent_project_shp: str = "DWSRF_Projects",
    parcels: str = "Benton_Harbor_Parcel",
    new_constr: str = "New_Development_Areas",
    nonresident: str = "Large_Non_Residential_Parcels",
):

    # pd.read_csv(get_flint_source(old_slr_hvi_filename))

    get_bh_source = lambda sfn: get_source(
        source_filename=sfn, data_dir=data_dir, client_bucket=bucket_name
    )

    spreadsheets = [
        active_accounts,
        active_shutoffs,
        inactive_accounts,
        full_replacements,
        inspections,
    ]
    act_acct, act_shut, inact_acct, replace, inspect = [
        pd.read_excel(get_bh_source(f)) for f in spreadsheets
    ]

    parcels_gdf = gpd.read_file(get_shapefile_source(parcels, data_dir, bucket_name))

    newconstr_gdf = gpd.read_file(
        get_shapefile_source(new_constr, data_dir, bucket_name)
    )

    nonresid_gdf = gpd.read_file(
        get_shapefile_source(nonresident, data_dir, bucket_name)
    )

    recent_proj_gdf = gpd.read_file(
        get_shapefile_source(recent_project_shp, data_dir, bucket_name)
    )

    gdb_path = download_folder(
        source_folder=gdb_folder, data_dir=data_dir, client_bucket="bc-benton-harbor"
    )

    # Tap records are the starting point for analysis,
    # according to Brandon.
    tap_records = gpd.read_file(gdb_path, layer="wTapRecords")

    ##############################
    # LABELS
    ###############################
    # Assign public / private lead labels;
    # assuming that
    # COP = copper, GP = galvanized, LP = lead

    lead_label_filter = (
        lambda x: True if x in ["LP", "GP"] else False if x == "COP" else None
    )

    tap_records["public_label"] = tap_records["UTILMAT"].apply(lead_label_filter)
    tap_records["private_label"] = tap_records["CUSTMAT"].apply(lead_label_filter)

    ##############################
    # FEATURES
    ###############################

    categorical_features = []
    numerical_features = []

    # UTILINSTDAT timestamp
    from datetime import datetime

    datestring_format = "%Y-%m-%dT%H:%M:%S"
    datestring_to_timestamp = (
        lambda s: np.nan
        if pd.isnull(s)
        else datetime.strptime(s, datestring_format).timestamp()
    )

    # tap_records['util_inst_timestamp'] = tap_records['UTILINSDAT'].apply(datestring_to_timestamp)
    # tap_records['cust_inst_timestamp'] = tap_records['CUSTINSDAT'].apply(datestring_to_timestamp)

    # numerical_features += ['util_inst_timestamp', 'cust_inst_timestamp']

    # RESYRBLT needs no processing
    numerical_features += ["RESYRBLT"]

    # LATSIZE, maybe?
    # Treat as categorical
    categorical_features += ["LATSIZE"]

    # tap_size will be categorical
    # Remove apostrophes and "IN" (inches) and trailing whitespace
    clean_tap_size = (
        lambda s: np.nan
        if pd.isnull(s)
        else s.replace("'", "").replace('"', "").replace("IN", "").strip()
    )
    tap_records["tap_size"] = tap_records["TapSize"].apply(clean_tap_size)
    categorical_features.append("tap_size")

    # TapDate
    # Needs special cleaning; a bunch of different formats are included.
    from datetime import datetime

    def clean_tap_date(s):
        if pd.isnull(s):
            return np.nan
        else:
            s = s.strip()
            dformats = [
                "%m-%d-%Y",
                "%m-%d-%y",
                "%m/%Y",
                "%m/%d/%Y",
                "%m/%d/%y",
                "%Y",
                "%m-%Y",
            ]
            for dformat in dformats:
                try:
                    return datetime.strptime(s, dformat).timestamp()
                except ValueError:
                    pass
            # If none of the date formats worked:
            return np.nan

    tap_records["tap_date"] = tap_records["TapDate"].apply(clean_tap_date)
    numerical_features.append("tap_date")

    # Fine without cleaning
    categorical_features.append("VacantLot")

    # CUSTSIZE categorical?
    categorical_features.append("CUSTSIZE")

    # Depth numerical
    numerical_features.append("DEPTH")

    tap_records['safe'] = tap_records.public_label.map({True:False, False:True, None:None})
    tap_records['domainid'] = tap_records.GlobalID

    tap_records['has_lead'] = tap_records.public_label.map({True:True, False:False, None:None})

    return tap_records[
        categorical_features
        + numerical_features
        + ["public_label", "GlobalID", "geometry", "safe", "domainid"]
    ].to_crs('epsg:4326')


def read_archive(bucket, obj, in_memory=True):

    # TODO: Add functionality to handle zip archives that we
    #   actually can download (in_memory=False)
    if in_memory:

        file_stream = io.BytesIO()
        arch_obj = boto3.resource("s3").Bucket(bucket).Object(obj)
        arch_obj.download_fileobj(file_stream)
        zip_obj = zipfile.ZipFile(file_stream)

    return zip_obj


def get_suez_data(
    data_dir: Union[Path, str] = "data/suez",
    bucket_name: str = "bc-suez",
    bc: str = 'BlueConduit.gdb.zip',
    bc5: str = 'BlueConduit5.gdb.zip',
    bc6: str = 'BlueConduit6.gdb.zip',
    bc7: str = 'BlueConduit7.gdb.zip',
    bc8: str = 'BlueConduit8.gdb.zip',
    bc9: str = 'BlueConduit9.gdb.zip',
    bc11: str = 'BlueConduit11.gdb.zip',
    bc0702: str = 'BlueConduit_0702.gdb.zip',
    bc0716: str = 'BlueConduit_0716.gdb.zip',
    bc0729: str = 'BlueConduit_0729.gdb.zip',
    bc0818: str = 'BlueConduit_0818.gdb.zip',
    bc0909: str = 'BlueConduit_0909.gdb.zip',
    bc0924: str = 'BlueConduit_0924.gdb.zip',
    bc1013: str = 'BlueConduit_1013.gdb.zip',
    bc1029: str = 'BlueConduit_1029.gdb.zip',
    bc1201: str = 'BlueConduit_1201.gdb.zip',
    bc1221: str = 'BlueConduit_1221.gdb.zip',
    bc2021: str = 'BlueConduit2021.gdb.zip',
    bc2021_0427: str = 'BlueConduit2021_0427.gdb.zip',
    bc2021_0618: str = 'BlueConduit2021_0618.gdb.zip',
    bc2021_0730: str = 'BlueConduit2021_07302021.gdb.zip',
    ms_info_2018: str = '2018_MS_Info.dbf',
    parcel_data: str = 'mod4_suez_parcels_2021.csv',
    **kwargs

) -> pd.DataFrame:

    # log.info("Loading data from bc-halifax-ca bucket")

    '''s3_xlsx = (
        "s3://bc-halifax-ca/HWDS_service_civic_link.xlsx",
        "s3://bc-halifax-ca/HWDS_service_maintenance_1of2.xls",
        "s3://bc-halifax-ca/HWDS_service_maintenance_2of2.xls",
    )

    civic_link_, maint_1_, maint_2_ = (pd.read_excel(xlsx) for xlsx in s3_xlsx)

    gdb_layers = (
        "Service_Maintenance",
        "Service_Authoritative_Source",
        "Service_Material_Observation",
    )

    services_, service_auth_, service_matobs_ = (
        gpd.read_file("zip+s3://bc-halifax-ca/945518.gdb.zip", layer=lr)
        for lr in gdb_layers
    )'''

    filenames = ['BlueConduit5.gdb.zip',
             'BlueConduit6.gdb.zip',
             'BlueConduit7.gdb.zip',
             'BlueConduit8.gdb.zip',
             'BlueConduit9.gdb.zip',
             'BlueConduit11.gdb.zip',
             'BlueConduit_0702.gdb.zip',
             'BlueConduit_0716.gdb.zip',
             'BlueConduit_0729.gdb.zip',
             'BlueConduit_0818.gdb.zip',
             'BlueConduit_0909.gdb.zip',
             'BlueConduit_0924.gdb.zip',
             'BlueConduit_1013.gdb.zip',
             'BlueConduit_1029.gdb.zip',
             'BlueConduit_1201.gdb.zip',
             'BlueConduit_1221.gdb.zip',
             'BlueConduit2021.gdb.zip',
             'BlueConduit2021_0427.gdb.zip',
             'BlueConduit2021_0618.gdb.zip',
             'BlueConduit2021_07302021.gdb.zip',
             'BlueConduit2021_09022021.gdb.zip',
             'BlueConduit2021_0920.gdb.zip'
             ]

    layer_names = ['Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services',
             'Services2021',
             'Services2021',
             'FinalizedLeadRPLCareaservices2021',
             'FinalizedLeadRPLCareaservices2021_0702',
             'FinalizedLeadRPLCareaservices2021_0702',
             'FinalizedLeadRPLCareaservices2021_0702']

    dates = ['2020-03-02', '2020-03-03', '2020-03-06', '2020-03-30',
         '2020-04-15', '2020-06-09', '2020-07-02', '2020-07-16', '2020-07-29',
         '2020-08-18', '2020-09-09', '2020-09-24', '2020-10-13', '2020-10-29',
         '2020-12-01', '2020-12-21', '2021-04-20', '2020-04-27', '2021-06-18',
         '2021-07-30', '2021-09-02', '2021-09-20']

    log.info("Loading data from bc-suez bucket")

    '''services_, service_auth_, service_matobs_ = (
        gpd.read_file("zip+s3://bc-halifax-ca/945518.gdb.zip", layer=lr)
        for lr in gdb_layers
    )'''

    sl_dataframes = []
    non_conf_dataframes = []
    df = None
    for filename, layer_name, batch_date in zip(filenames, layer_names, dates):
        filepath = "zip+s3://bc-suez/shape_files/%s" % filename
        log.info("Loading: %s" % filepath)
        # print("Loading: %s" % filepath)

        df = gpd.read_file(filepath, layer = layer_name, ignore_geometry = True)
        df_ = df[df.SUEZMaterialConfirmed.isna()].copy()
        df = df[~df.SUEZMaterialConfirmed.isna()]
        df['batch_date'] = batch_date
        df_['batch_date'] = batch_date


        if 'SUEZMaterialConfirmedSource' not in df.columns:
            df['SUEZMaterialConfirmedSource'] = None
            df['CustMaterialConfirmedSource'] = None
            df_['SUEZMaterialConfirmedSource'] = None
            df_['CustMaterialConfirmedSource'] = None

        sl_dataframes.append(df)
        non_conf_dataframes.append(df_)

    log.info("Loading parcel geometry")
    parcel_df = gpd.read_file("zip+s3://bc-suez/shape_files/%s" % "BlueConduit_1221.gdb.zip", layer='Parcels')
    log.info("Loading road paving data")
    recent_paved_df = gpd.read_file("zip+s3://bc-suez/shape_files/%s" % filenames[-1], layer='Paving')

    recent_paved_df['geometry'] = recent_paved_df['geometry'].buffer(100)
    pave_2021_intersection = gpd.overlay(parcel_df, recent_paved_df[recent_paved_df['YEARPAVED'] >= '2021'], how='intersection')

    parcel_df['recently_paved'] = parcel_df.PAMS_PIN.isin(pave_2021_intersection.PAMS_PIN)

    log.info("Concatenating dataframes")
    sl_dataframes_concat = pd.concat(sl_dataframes)
    non_conf_dataframes_concat = pd.concat(non_conf_dataframes)


    log.info("Aggregating dataframe")
    sl_df = sl_dataframes_concat.groupby('SP_ID').agg({'batch_date':'first',
                                                'SUEZMaterialAssumed2019':'last',
                                                'SUEZMaterialConfirmed':'last',
                                                'CustomerMaterialAssumed2019':'last',
                                                'CustomerMaterialConfirmed':'last',
                                                'GooseneckMaterial':'last',
                                                'PAMS_PIN':'last',
                                                'EAMID':'last',
                                                'CustMaterialConfirmedSource':'last',
                                                'SUEZMaterialConfirmedSource':'last',
                                                'MainExt':'last',
                                                'CITY':'last'}).reset_index()

    non_conf_sl_df = non_conf_dataframes_concat.groupby('SP_ID').agg({'batch_date':'first',
                                                'SUEZMaterialAssumed2019':'last',
                                                'SUEZMaterialConfirmed':'last',
                                                'CustomerMaterialAssumed2019':'last',
                                                'CustomerMaterialConfirmed':'last',
                                                'GooseneckMaterial':'last',
                                                'PAMS_PIN':'last',
                                                'EAMID':'last',
                                                'CustMaterialConfirmedSource':'last',
                                                'SUEZMaterialConfirmedSource':'last',
                                                'MainExt':'last',
                                                'CITY':'last'}).reset_index()

    non_conf_sl_df = non_conf_sl_df[~non_conf_sl_df.SP_ID.isin(sl_df.SP_ID)]

    sl_df = pd.concat([sl_df, non_conf_sl_df])

    # Load 2018 Pipe data
    log.info("Loading: 2018 pipe data")
    filepath = "s3://bc-suez/2018_pipe_data.csv"
    pipe_df = pd.read_csv(filepath)

    # Load mod4 data
    log.info("Loading: MOD4 data")
    filepath = "s3://bc-suez/mod4_suez_parcels_2021.csv"
    mod4_df = pd.read_csv(filepath)

    # Merging Data

    mod4_df = mod4_df[mod4_df.PAMS_PIN.isin(sl_df.PAMS_PIN)]

    def extract_building_desc():
        stors = []
        struc = []
        style = []
        garag = []

        for value in mod4_df.BLDG_DESC.values:
            if value is not None:
                try:
                    tokens = value.split('-')
                except:
                    tokens = []
                if len(tokens) == 4:
                    stors.append(tokens[0])
                    struc.append(tokens[1])
                    style.append(tokens[2])
                    garag.append(tokens[3])
                else:
                    stors.append(None)
                    struc.append(None)
                    style.append(None)
                    garag.append(None)
            else:
                stors.append(None)
                struc.append(None)
                style.append(None)
                garag.append(None)

        mod4_df['stors'] = stors
        mod4_df['struc'] = struc
        mod4_df['style'] = style
        mod4_df['garag'] = garag

    def extract_deed_year():
        years = []

        for value in mod4_df.DEED_DATE.values.astype(str):
            if value is not None:
                try:
                    years.append(int(value[:2]))
                except:
                    years.append(-1)
            else:
                years.append(-1)
        mod4_df['deed_year'] = years

    val_counts = mod4_df.FAC_NAME.value_counts()
    names = val_counts[val_counts > 25].keys().tolist()

    def extract_fac_name(name):

        if pd.isna(name):
            return 'None'
        else:
            if name in names:
                return name
            else:
                return 'Other'

    val_counts = mod4_df.PROP_USE.value_counts()
    uses = val_counts[val_counts > 10].keys().tolist()

    def extract_prop_use(use):

        if pd.isna(use):
            return 'None'
        else:
            if use in uses:
                return use
            else:
                return 'Other'

    val_counts = mod4_df.BLDG_CLASS.value_counts()
    classes = val_counts[val_counts > 10].keys().tolist()

    def extract_bldg_class(class_):

        if pd.isna(class_):
            return 'None'
        else:
            if class_ in classes:
                return class_
            else:
                return 'Other'


    def extract_install_date(date):

        try:
            return pd.to_datetime(date).year
        except:
            return -1


    extract_building_desc()
    extract_deed_year()
    mod4_df['FAC_NAME'] = mod4_df['FAC_NAME'].apply(extract_fac_name)
    mod4_df['PROP_USE'] = mod4_df['PROP_USE'].apply(extract_prop_use).astype(str)
    mod4_df['BLDG_CLASS'] = mod4_df['BLDG_CLASS'].apply(extract_bldg_class)

    pipe_df['install_year'] = pipe_df['INSTALLDAT'].apply(extract_install_date)

    keep_cols_1 = ['PAMS_PIN', 'stors', 'struc', 'style', 'garag', 'deed_year', 'COUNTY', 'MUN_NAME', 'LAND_VAL',
             'IMPRVT_VAL', 'NET_VALUE', 'LAST_YR_TX', 'CALC_ACRE', 'SALE_PRICE', 'DWELL', 'COMM_DWELL',
             'PROP_USE', 'BLDG_CLASS', 'YR_CONSTR', 'PROP_CLASS'] #,'SHAPE_Length', 'SHAPE_Area',

    keep_cols_2 = ['ASBUILTLEN', 'DIAMETER', 'MATERIAL', 'PRESSUREZO', 'INSTALLDAT', 'EAMID', 'SUBTYPECOD', 'install_year']

    Xdata = sl_df.copy()
    log.info("Total number of SL records before merge: %d" % len(Xdata))
    Xdata = Xdata.merge(pipe_df[keep_cols_2], how='left')
    log.info("Total number of SL records after merge 1: %d" % len(Xdata))
    Xdata = mod4_df[keep_cols_1].merge(Xdata, how='left')
    log.info("Total number of SL records after merge 2: %d" % len(Xdata))
    Xdata = Xdata.merge(parcel_df[['PAMS_PIN', 'recently_paved', 'geometry']])

    poss_lead_materials = ['LZ', 'GZ', 'LEAD ', 'GZ_LL', 'LEAD', 'Galvanized', 'GALVONIZED STEEL', 'STEEL', 'Lead']
    poss_lead_materials_pub = ['LZ', 'GZ', 'LEAD', 'BR', 'BRASS', ' BRASS', 'LEAD ', 'GZ_LL', 'Galvanized', 'GALVONIZED STEEL', 'STEEL']
    poss_lead_materials_pri = ['LZ', 'GZ', 'Lead', 'LEAD', 'LEAD ', 'Galvanized', 'GALVONIZED STEEL']

    Xdata['target_1'] = None
    Xdata['target_2'] = None
    Xdata['target_3'] = None
    Xdata['target_4'] = None
    Xdata['target_5'] = None
    Xdata['target_6'] = None
    Xdata['target_7'] = None

    conf_mask = ~Xdata.SUEZMaterialConfirmed.isna()

    Xdata.loc[conf_mask, 'target_1'] = (Xdata.loc[conf_mask].SUEZMaterialConfirmed.isin(poss_lead_materials_pub) & \
                ~Xdata.loc[conf_mask].CustomerMaterialConfirmed.isin(poss_lead_materials_pri)).astype(int)

    Xdata.loc[conf_mask, 'target_2'] = (Xdata.loc[conf_mask].SUEZMaterialConfirmed.isin(poss_lead_materials_pub) | \
                    Xdata.loc[conf_mask].CustomerMaterialConfirmed.isin(poss_lead_materials_pri)).astype(int)

    Xdata.loc[conf_mask, 'target_3'] = (Xdata.loc[conf_mask].SUEZMaterialConfirmed.isin(poss_lead_materials)).astype(int)

    Xdata.loc[conf_mask, 'target_4'] = (Xdata.loc[conf_mask].SUEZMaterialConfirmed.isin(poss_lead_materials) & \
                    ~Xdata.loc[conf_mask].CustomerMaterialConfirmed.isin(poss_lead_materials_pri)).astype(int)

    Xdata.loc[conf_mask, 'target_5'] = (Xdata.loc[conf_mask].SUEZMaterialConfirmed.isin(poss_lead_materials_pub)).astype(int)

    Xdata.loc[conf_mask, 'target_6'] = (Xdata.loc[conf_mask].CustomerMaterialConfirmed.isin(poss_lead_materials_pri)).astype(int)

    Xdata.loc[conf_mask, 'target_7'] = (Xdata.loc[conf_mask].GooseneckMaterial.isin(poss_lead_materials)).astype(int)

    return gpd.GeoDataFrame(Xdata)




@to_processed_dir
def get_flint_data(
    data_dir: Union[Path, str] = "data/flint",
    bucket_name: str = "bc-flint",
    old_slr_hvi_filename: str = "old_slr_hvi_records.csv",
    database_slr_filename: str = "database_slr_records.csv",
    database_hvi_filename: str = "database_hvi_records.csv",
    features_filename: str = "parcel_data_preds.csv",
    parcel_filename: str = "parcel_data.csv",
    fast_aug_2018: str = "fast_start_aug_2018.csv",
    fast_nov_2018: str = "fast_start_nov_2018.csv",
    discrep_sheets_1: str = "2019.03.11 City Response.xlsx",
    discrep_sheets_2: str = "2019.03.15 City Response.xlsx",
    jan_2019_file: str = "fast_start_data_jan_2019.csv",
    feb_2019_file: str = "FAST Reports Merged.xlsx",
    may_2019_file: str = "2019.05.30 FAST Report.xlsx",
    jun_2019_file: str = "2019-06-14 Fast Start Exploration Status.xlsx",
    jul_2019_file: str = "2019-07-14 Fast Start Exploration Status.xlsx",
    aug_2019_file: str = "2019.08.28 FAST Report.xlsx",
    sep_2019_file: str = "2019-09-14 Fast Start Exploration Status.xlsx",
    oct_2019_file: str = "2019-10-14 Fast Start Exploration Status.xlsx",
    dec_2019_file: str = "2019.12.02 FAST Report.xlsx",
    mar_2020_file: str = "2019.12.02 FAST Report_Rev 2019.12.23.xlsx",
    mar_2020_2_file: str = "2020.03.02 SLR Report.xlsx",
    jun_2020_file: str = "2020.02.15-2020.06.30 SLR Activity Data.xlsx",
    aug_2020_file: str = "2020.08.28 SLR Report.xlsx",
    sep_2020_file: str = "2020-09-14 SL Exp-Rep Status.xlsx",
    oct_2020_file: str = "2020-10-14 Fast Start Exploration Status.xlsx",
    dec_2020_file: str = "2020-12-14 Fast Start Exploration Status.xlsx",
    **kwargs,
) -> pd.DataFrame:

    dangerous_materials = [
        "LEAD",
        "GALVANIZED",
        "COPPER AND GALVANIZED",
        "NON-COPPER",
        "UNKNOWN",
    ]
    safe_materials = ["COPPER", "PLASTIC"]
    valid_materials = safe_materials + dangerous_materials

    # Get source is kind of verbose. Here's something easier on the eyes
    get_flint_source = lambda sfn: get_source(
        source_filename=sfn, data_dir=data_dir, client_bucket=bucket_name
    )

    def incorporate_city_response():
        def get_materials(sheet):

            public_materials = []
            private_materials = []
            # print(sheet)
            for line in sheet["material"].values:
                if pd.isna(line):
                    public_materials.append(np.nan)
                    private_materials.append(np.nan)
                else:
                    tokens = line.split("-")
                    if len(tokens) == 1:
                        public_materials.append(line.upper())
                        private_materials.append(line.upper())

                    else:
                        public_materials.append(tokens[0].upper())
                        private_materials.append(tokens[1].upper())
            return public_materials, private_materials

        sheet_5 = pd.read_excel(
            get_flint_source(discrep_sheets_1),
            sheet_name="5. Dangerous HV Non Replaced",
        )

        sheet_5 = sheet_5[
            [
                "pid",
                "Property Address",
                "status",
                "Date of replacement",
                "original type of line material",
                "sl_public_type",
                "sl_private_type",
            ]
        ]
        sheet_5 = sheet_5.rename(
            {
                "Date of replacement": "created_at",
                "original type of line material": "material",
            },
            axis=1,
        )

        rep_sheet_5 = sheet_5[sheet_5["status"] == "line replaced"]
        # print(len(rep_sheet_5), len(sheet_5))

        # print(sheet_5.columns.tolist())

        # Until we hear back, I'm assuming that the first material is public.
        pub, pri = get_materials(sheet_5)

        sheet_5["fs_sl_public_type"] = pub
        sheet_5["fs_sl_private_type"] = pri

        # 6. Dangerous Non-HV Non Replaced

        sheet_6 = pd.read_excel(
            get_flint_source(discrep_sheets_1),
            sheet_name="6. Danger Non HV Non Replaced",
        )

        sheet_6 = sheet_6[
            [
                "pid",
                "Property Address",
                "status",
                "Date of replacement",
                "original type of line material",
                "sl_public_type",
                "sl_private_type",
            ]
        ]
        sheet_6 = sheet_6.rename(
            {
                "Date of replacement": "created_at",
                "original type of line material": "material",
            },
            axis=1,
        )

        pub, pri = get_materials(sheet_6)

        sheet_6["fs_sl_public_type"] = pub
        sheet_6["fs_sl_private_type"] = pri

        # 7. Uknown SL No Replacement

        sheet_7 = pd.read_excel(
            get_flint_source(discrep_sheets_1),
            sheet_name="7. Unknown SL No Replacement",
        )
        sheet_7 = sheet_7[
            [
                "pid",
                "Property Address",
                "status",
                "Date of replacement",
                "original type of line material",
                "sl_public_type",
                "sl_private_type",
            ]
        ]
        sheet_7 = sheet_7.rename(
            {
                "Date of replacement": "created_at",
                "original type of line material": "material",
            },
            axis=1,
        )

        pub, pri = get_materials(sheet_7)

        sheet_7["fs_sl_public_type"] = pub
        sheet_7["fs_sl_private_type"] = pri

        # 8. Empty Rows

        sheet_8 = pd.read_excel(
            get_flint_source(discrep_sheets_2), sheet_name="8. Empty Rows"
        )
        sheet_8 = sheet_8[
            [
                "pid",
                "Property Address",
                "status",
                "Date of replacement",
                "original type of line material",
                "sl_public_type",
                "sl_private_type",
            ]
        ]
        sheet_8 = sheet_8.rename(
            {
                "Date of replacement": "created_at",
                "original type of line material": "material",
            },
            axis=1,
        )

        pub, pri = get_materials(sheet_8)

        sheet_8["fs_sl_public_type"] = pub
        sheet_8["fs_sl_private_type"] = pri

        # 9. Empty Rows

        sheet_9 = pd.read_excel(
            get_flint_source(discrep_sheets_2), sheet_name="9. Empty Rows with Date"
        )

        sheet_9 = sheet_9[
            [
                "pid",
                "Property Address",
                "status",
                "Date of replacement",
                "original type of line material",
                "sl_public_type",
                "sl_private_type",
            ]
        ]
        sheet_9 = sheet_9.rename(
            {
                "Date of replacement": "created_at",
                "original type of line material": "material",
            },
            axis=1,
        )

        pub, pri = get_materials(sheet_9)

        sheet_9["fs_sl_public_type"] = pub
        sheet_9["fs_sl_private_type"] = pri

        frames = [sheet_5, sheet_6, sheet_7, sheet_8, sheet_9]

        response_df = pd.concat(frames)

        response_df["replaced"] = (response_df["status"] == "line replaced") | (
            (response_df["status"] == "no FAST START record")
            & (~response_df["created_at"].isna())
        )
        response_df["sl_visit"] = (
            response_df["status"] == "locate only-FAST START"
        ) | (response_df["status"] == "line replaced")
        response_df["hv_visit"] = False

        response_df["sl_public_type"] = response_df["fs_sl_public_type"].combine_first(
            response_df["sl_public_type"]
        )

        response_df["sl_private_type"] = response_df[
            "fs_sl_private_type"
        ].combine_first(response_df["sl_private_type"])

        response_df["dangerous"] = response_df.sl_private_type.isin(
            dangerous_materials
        ) | response_df.sl_public_type.isin(dangerous_materials)

        response_df = response_df[~response_df["pid"].isna()]
        response_df["pid"] = response_df["pid"].astype(int)

        # return response_df
        return response_df[
            [
                "pid",
                "Property Address",
                "created_at",
                "dangerous",
                "hv_visit",
                "replaced",
                "sl_private_type",
                "sl_public_type",
                "sl_visit",
            ]
        ]

    response_df = incorporate_city_response()

    old_slr_hvi_df = pd.read_csv(get_flint_source(old_slr_hvi_filename))
    database_slr_df = pd.read_csv(get_flint_source(database_slr_filename))
    database_hvi_df = pd.read_csv(get_flint_source(database_hvi_filename))
    parcel_ids_2016 = pd.read_csv(get_flint_source(parcel_filename))[
        ["PID no Dash", "Property Address"]
    ]
    fast_start_aug_df = pd.read_csv(get_flint_source(fast_aug_2018))
    fast_start_nov_df = pd.read_csv(get_flint_source(fast_nov_2018))

    # We need to combine on addresses because they don't give us pids.  Luckily it looks like
    # they are using the same address that we find in our parcel data set.
    fast_start_df = fast_start_aug_df.append(fast_start_nov_df)
    fast_start_df.rename(
        {
            "Address of Service Line Exploration ": "address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
            "Service Line Portion Replaced": "part_replaced",
            "Service Line Replacement Date ": "replacement_date",
        },
        inplace=True,
        axis=1,
    )

    parcel_ids_2016.rename({"PID no Dash": "pid"}, inplace=True, axis=1)
    fast_start_df["address"] = fast_start_df["address"].apply(str.upper)
    fast_start_df = fast_start_df.merge(
        parcel_ids_2016[["pid", "Property Address"]],
        left_on="address",
        right_on="Property Address",
    ).drop(["Property Address", "address", "replacement_date", "part_replaced"], axis=1)
    fast_start_df.rename({"PID no Dash": "pid"}, axis=1, inplace=True)
    fast_start_df["source"] = "fast_start_2018"

    # Rename columns in other tables to match convention
    old_slr_hvi_df.rename(
        columns={
            "parcel_id": "pid",
            "original_private_material": "sl_private_type",
            "original_public_material": "sl_public_type",
        },
        inplace=True,
    )
    old_slr_hvi_df["source"] = "old_slr_hvi"

    database_slr_df.rename(
        columns={
            "parcel_id": "pid",
            "original_private_material": "sl_private_type",
            "original_public_material": "sl_public_type",
        },
        inplace=True,
    )
    database_slr_df["source"] = "database_slr"

    database_hvi_df.rename(
        columns={
            "parcel_id": "pid",
            "original_private_material": "sl_private_type",
            "original_public_material": "sl_public_type",
        },
        inplace=True,
    )
    database_hvi_df["source"] = "database_hvi"

    # We are only concerned with service line materials and when they were replaced.
    columns = ["pid", "sl_private_type", "sl_public_type", "created_at", "source"]

    sl_df = old_slr_hvi_df[columns]
    sl_df = (
        sl_df.append(database_hvi_df[columns])
        .append(database_slr_df[columns])
        .append(fast_start_df)
    )

    # Add columns.  I deserve to be shot for this block.
    sl_df["hv_visit"] = False
    sl_df.iloc[
        len(old_slr_hvi_df) : len(old_slr_hvi_df) + len(database_hvi_df),
        sl_df.columns.get_loc("hv_visit"),
    ] = True
    sl_df.iloc[
        len(old_slr_hvi_df) + len(database_hvi_df) :, sl_df.columns.get_loc("hv_visit")
    ] = False

    sl_df["sl_visit"] = True
    sl_df.iloc[
        len(old_slr_hvi_df) : len(old_slr_hvi_df) + len(database_hvi_df),
        sl_df.columns.get_loc("sl_visit"),
    ] = False
    sl_df.iloc[
        len(old_slr_hvi_df) + len(database_hvi_df) :, sl_df.columns.get_loc("sl_visit")
    ] = True

    # Make all service line labels upper case and strip leading/trailing spaces.
    sl_df["sl_private_type"] = (
        sl_df["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    sl_df["sl_public_type"] = (
        sl_df["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    public_mask = sl_df["sl_private_type"].isin(dangerous_materials)
    private_mask = sl_df["sl_public_type"].isin(dangerous_materials)

    sl_df.loc[sl_df["hv_visit"] == True, "replaced"] = False

    # We are assuming that replacements occur where hazardous materials are found.

    sl_df.loc[
        (sl_df["hv_visit"] == False)
        & (
            sl_df["sl_private_type"].isin(safe_materials)
            & sl_df["sl_public_type"].isin(safe_materials)
        ),
        "replaced",
    ] = False
    sl_df.loc[
        (sl_df["hv_visit"] == False)
        & ~(
            sl_df["sl_private_type"].isin(safe_materials)
            & sl_df["sl_public_type"].isin(safe_materials)
        ),
        "replaced",
    ] = True

    sl_df = sl_df[sl_df.pid.astype(str).apply(str.isdigit)]
    sl_df["pid"] = sl_df["pid"].astype(int)

    sl_df.created_at = pd.to_datetime(sl_df.created_at)
    sl_df.reset_index(inplace=True, drop=True)

    ################################ January 2019
    fast_start_df = pd.read_csv(get_flint_source(jan_2019_file))

    sl_public_type_col = fast_start_df["SLR Original Public Material"].combine_first(
        fast_start_df["Original Public Material"]
    )
    sl_private_type_col = fast_start_df["SLR Original Private Material"].combine_first(
        fast_start_df["Original Private Material"]
    )
    created_at_col = fast_start_df["SLR Work Date Performed"].combine_first(
        fast_start_df["SLE Work Date Performed"]
    )
    pid_col = fast_start_df["Parcel  ID"].apply(str.replace, args=("-", "")).astype(int)

    #  We want to know the label, how we know the label, and whether SL was replaced.

    # Whether the home is hydrovac, excavated, or replaced
    hv_visit_col = (
        fast_start_df["HVI Inspection"] == "Yes"
    )  # & (fast_start_df['SLR Inspection'] != 'Yes')

    # Whether the home was visited by replacement crew
    m1 = ~(
        (fast_start_df["HVI Inspection"] == "Yes")
        & (fast_start_df["SLR Inspection"] != "Yes")
    )
    m2 = pd.notnull(fast_start_df["SLE Work Date Performed"]) | pd.notnull(
        fast_start_df["SLR Work Date Performed"]
    )
    m3 = (
        pd.notnull(fast_start_df["SLR Work Date Performed"])
        & pd.isnull(fast_start_df["SLR Inspection"])
        & (fast_start_df.index < 9666)
    )

    sl_visit_col = (m1 & m2) ^ m3

    sl_df_ = pd.DataFrame(
        {
            "pid": pid_col,
            "sl_public_type": sl_public_type_col,
            "sl_private_type": sl_private_type_col,
            "created_at": created_at_col,
            "hv_visit": hv_visit_col,
            "sl_visit": sl_visit_col,
        }
    )

    sl_df_["created_at"] = pd.to_datetime(sl_df_.created_at)
    sl_df_["sl_private_type"] = (
        sl_df_["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    sl_df_["sl_public_type"] = (
        sl_df_["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    # Whether a replacement occurred.
    public_mask = sl_df_["sl_private_type"].isin(dangerous_materials)
    private_mask = sl_df_["sl_public_type"].isin(dangerous_materials)
    replaced_col = pd.notnull(fast_start_df["SLR Work Date Performed"]) & (
        public_mask | private_mask
    )
    sl_df_["replaced"] = replaced_col

    sl_df_ = sl_df_.sort_values(by=["created_at"], ascending=False)
    sl_df_ = sl_df_.drop_duplicates("pid", keep="first")

    sl_df_ = sl_df_[
        (
            sl_df_.sl_public_type.isin(valid_materials)
            | (sl_df_.sl_private_type.isin(valid_materials))
        )
    ]
    sl_df_["dangerous"] = sl_df_.sl_private_type.isin(
        dangerous_materials
    ) | sl_df_.sl_public_type.isin(dangerous_materials)
    sl_df_["source"] = "fast_start_jan_2019"

    sl_df = sl_df.append(sl_df_)

    ################################ February 2019

    fast_2019 = pd.read_excel(
        get_flint_source(feb_2019_file), sheet_name="1-Service Line Ex. and Rep."
    )
    fast_2019.rename(
        {
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
            "Parcel ID ": "pid",
        },
        axis=1,
        inplace=True,
    )

    fast_2019["replaced"] = ~pd.isna(fast_2019["Service Line Replacement Date "])
    fast_2019 = fast_2019[
        [
            "Property Address",
            "created_at",
            "sl_public_type",
            "sl_private_type",
            "replaced",
            "pid",
        ]
    ]

    fast_2019["Property Address"] = fast_2019["Property Address"].astype(str)
    fast_2019["Property Address"] = fast_2019["Property Address"].apply(str.upper)

    mask = ~fast_2019["Property Address"].isin(parcel_ids_2016["Property Address"])
    bad_adds = fast_2019[mask]["Property Address"]

    def strip_dashes(s):
        # Remove dashes from a string
        return s.replace("-", "")

    num_rows = len(fast_2019)
    fast_2019 = fast_2019[fast_2019["pid"] != "No Matching PID Found"]
    num_rows_post_drop = len(fast_2019)
    # print('Number of dropped rows: %d' % (num_rows-num_rows_post_drop))
    fast_2019["pid"] = fast_2019["pid"].apply(strip_dashes).astype(int)

    # print(len(feb_2019))
    # fast_2019 = fast_2019.merge(parcel_ids_2016, on='pid').drop('Property Address', axis=1)
    # print(len(feb_2019))

    fast_2019["hv_visit"] = False
    fast_2019["sl_visit"] = True

    fast_2019["sl_private_type"] = (
        fast_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    fast_2019["sl_public_type"] = (
        fast_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    public_mask = fast_2019["sl_private_type"].isin(dangerous_materials)
    private_mask = fast_2019["sl_public_type"].isin(dangerous_materials)

    fast_2019["dangerous"] = fast_2019.sl_private_type.isin(
        dangerous_materials
    ) | fast_2019.sl_public_type.isin(dangerous_materials)
    fast_2019["source"] = "Fast Reports Merged SLER"

    sl_df.set_index("pid", inplace=True)
    fast_2019.set_index("pid", inplace=True)

    sl_df = sl_df.append(fast_2019)

    sl_df.reset_index(inplace=True)

    ################################ City Discrepancies

    response_df["source"] = "discrepancy response"
    sl_df = sl_df.append(response_df[sl_df.columns.tolist()])

    ################################ March 2019

    mar_2019 = pd.read_excel(
        get_flint_source(feb_2019_file), sheet_name="Monthly Report"
    )

    # print(mar_2019.columns.tolist())
    mar_2019.rename(
        {
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
            "Parcel ID": "pid",
        },
        axis=1,
        inplace=True,
    )

    # print(mar_2019['Service Line Portion Replaced'].value_counts())

    mar_2019["replaced"] = ~pd.isna(mar_2019["Service Line Replacement Date "])
    mar_2019 = mar_2019[
        [
            "Property Address",
            "created_at",
            "sl_public_type",
            "sl_private_type",
            "replaced",
            "pid",
        ]
    ]

    mar_2019["Property Address"] = mar_2019["Property Address"].astype(str)
    mar_2019["Property Address"] = mar_2019["Property Address"].apply(
        str.replace, args=[".", ""]
    )
    mar_2019["Property Address"] = mar_2019["Property Address"].apply(str.upper)

    mask = ~mar_2019["Property Address"].isin(parcel_ids_2016["Property Address"])
    bad_adds = bad_adds.append(mar_2019[mask]["Property Address"])

    mar_2019 = mar_2019[mar_2019["pid"] != "No Matching PID Found"]
    # num_rows_post_drop = len(fast_2019)
    # print('Number of dropped rows: %d' % (num_rows-num_rows_post_drop))
    mar_2019["pid"] = mar_2019["pid"].apply(strip_dashes).astype(int)

    # print(len(mar_2019))
    # mar_2019 = mar_2019.merge(parcel_ids_2016).drop('Property Address', axis=1)
    # print(len(mar_2019))

    mar_2019["hv_visit"] = False
    mar_2019["sl_visit"] = True

    mar_2019["sl_private_type"] = (
        mar_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    mar_2019["sl_public_type"] = (
        mar_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    public_mask = mar_2019["sl_private_type"].isin(dangerous_materials)
    private_mask = mar_2019["sl_public_type"].isin(dangerous_materials)

    mar_2019["dangerous"] = mar_2019.sl_private_type.isin(
        dangerous_materials
    ) | mar_2019.sl_public_type.isin(dangerous_materials)
    mar_2019["source"] = "Fast Reports Merged MR"

    sl_df.set_index("pid", inplace=True)
    mar_2019.set_index("pid", inplace=True)

    sl_df = sl_df.append(mar_2019)
    sl_df.reset_index(inplace=True)

    ################################ May 2019

    may_2019 = pd.read_excel(get_flint_source(may_2019_file))

    may_2019.rename(
        {
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    # print(mar_2019['Service Line Portion Replaced'].value_counts())

    may_2019["replaced"] = ~pd.isna(may_2019["Service Line Replacement Date "])
    may_2019 = may_2019[
        [
            "Property Address",
            "created_at",
            "sl_public_type",
            "sl_private_type",
            "replaced",
        ]
    ]

    may_2019["Property Address"] = may_2019["Property Address"].astype(str)
    may_2019["Property Address"] = may_2019["Property Address"].apply(
        str.replace, args=[".", ""]
    )
    may_2019["Property Address"] = may_2019["Property Address"].apply(str.upper)

    num_rows = len(may_2019)

    may_2019 = may_2019.merge(parcel_ids_2016)

    num_rows_post_drop = len(may_2019)

    mask = ~may_2019["Property Address"].isin(parcel_ids_2016["Property Address"])
    bad_adds = bad_adds.append(may_2019[mask]["Property Address"])

    may_2019["hv_visit"] = False
    may_2019["sl_visit"] = True

    may_2019["sl_private_type"] = (
        may_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    may_2019["sl_public_type"] = (
        may_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    public_mask = may_2019["sl_private_type"].isin(dangerous_materials)
    private_mask = may_2019["sl_public_type"].isin(dangerous_materials)

    may_2019["dangerous"] = may_2019.sl_private_type.isin(
        dangerous_materials
    ) | may_2019.sl_public_type.isin(dangerous_materials)
    may_2019["source"] = "Fast Report Merged May"

    may_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(may_2019)
    sl_df.reset_index(inplace=True)

    ################################ June 2019

    june_2019 = pd.read_excel(get_flint_source(jun_2019_file))

    june_2019.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    june_2019.dropna(subset=["pid"], inplace=True)
    june_2019["pid"] = june_2019["pid"].astype(str).apply(strip_dashes).astype(int)

    june_2019["replaced"] = (
        june_2019["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    june_2019 = june_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    june_2019["hv_visit"] = False
    june_2019["sl_visit"] = True

    june_2019["sl_private_type"] = (
        june_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    june_2019["sl_public_type"] = (
        june_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    public_mask = june_2019["sl_private_type"].isin(dangerous_materials)
    # private_mask = june_2019['sl_public_type'].isin(dangerous_materials)

    june_2019["dangerous"] = june_2019.sl_private_type.isin(
        dangerous_materials
    ) | june_2019.sl_public_type.isin(dangerous_materials)
    june_2019["source"] = "Fast Start Exploration Status June"

    june_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(june_2019)
    sl_df.reset_index(inplace=True)

    ################################ July 2019

    july_2019 = pd.read_excel(get_flint_source(jul_2019_file), engine="openpyxl")

    july_2019.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    # print(mar_2019['Service Line Portion Replaced'].value_counts())

    july_2019["created_at"] = pd.to_datetime(
        july_2019["created_at"].combine_first(
            july_2019["Service Line Replacement Date "]
        )
    )

    july_2019["pid"] = july_2019["pid"].apply(strip_dashes).astype(int)

    july_2019["replaced"] = (
        july_2019["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    july_2019 = july_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    july_2019["hv_visit"] = False
    july_2019["sl_visit"] = True

    july_2019["sl_private_type"] = (
        july_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    july_2019["sl_public_type"] = (
        july_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    # public_mask = july_2019['sl_private_type'].isin(dangerous_materials)
    # private_mask = june_2019['sl_public_type'].isin(dangerous_materials)

    july_2019["dangerous"] = july_2019.sl_private_type.isin(
        dangerous_materials
    ) | july_2019.sl_public_type.isin(dangerous_materials)
    july_2019["source"] = "Fast Start Exploration Status July"

    july_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(july_2019)
    sl_df.reset_index(inplace=True)

    ################################ August 2019

    aug_2019 = pd.read_excel(
        get_flint_source(aug_2019_file),
        sheet_name="1-SL Exp and Rep",
        engine="openpyxl",
    )
    aug_2019.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    aug_2019["created_at"] = pd.to_datetime("2019-08-01")
    aug_2019["pid"] = aug_2019["pid"].apply(strip_dashes).astype(int)

    aug_2019["replaced"] = (
        aug_2019["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    aug_2019 = aug_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    aug_2019["hv_visit"] = False
    aug_2019["sl_visit"] = True

    aug_2019["sl_private_type"] = (
        aug_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    aug_2019["sl_public_type"] = (
        aug_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    # public_mask = july_2019['sl_private_type'].isin(dangerous_materials)
    # private_mask = june_2019['sl_public_type'].isin(dangerous_materials)

    aug_2019["dangerous"] = aug_2019.sl_private_type.isin(
        dangerous_materials
    ) | aug_2019.sl_public_type.isin(dangerous_materials)
    aug_2019["source"] = "Fast Start Exploration Status August"

    aug_2019 = aug_2019[
        ~(
            aug_2019.pid.isin(july_2019.index)
            | aug_2019.pid.isin(june_2019.index)
            | aug_2019.pid.isin(may_2019.index)
        )
    ]

    aug_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(aug_2019)
    sl_df.reset_index(inplace=True)

    ################################ September 2019
    sep_2019 = pd.read_excel(get_flint_source(sep_2019_file), engine="openpyxl")
    sep_2019.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    sep_2019.dropna(subset=["pid"], inplace=True)
    sep_2019["pid"] = sep_2019["pid"].apply(strip_dashes).astype(int)

    sep_2019["replaced"] = (
        sep_2019["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    sep_2019 = sep_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    sep_2019["hv_visit"] = False
    sep_2019["sl_visit"] = True

    sep_2019["sl_private_type"] = (
        sep_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    sep_2019["sl_public_type"] = (
        sep_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    # public_mask = july_2019['sl_private_type'].isin(dangerous_materials)
    # private_mask = june_2019['sl_public_type'].isin(dangerous_materials)

    sep_2019["dangerous"] = sep_2019.sl_private_type.isin(
        dangerous_materials
    ) | sep_2019.sl_public_type.isin(dangerous_materials)
    sep_2019["source"] = "Fast Start Exploration Status September"

    sep_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(sep_2019)
    sl_df.reset_index(inplace=True)

    ################################ October 2019

    oct_2019 = pd.read_excel(get_flint_source(oct_2019_file), engine="openpyxl")
    oct_2019.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Exploration Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    oct_2019.dropna(subset=["pid"], inplace=True)
    oct_2019["pid"] = oct_2019["pid"].apply(strip_dashes).astype(int)

    oct_2019["replaced"] = (
        oct_2019["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    oct_2019 = oct_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    oct_2019["hv_visit"] = False
    oct_2019["sl_visit"] = True

    oct_2019["sl_private_type"] = (
        oct_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    oct_2019["sl_public_type"] = (
        oct_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    # public_mask = july_2019['sl_private_type'].isin(dangerous_materials)
    # private_mask = june_2019['sl_public_type'].isin(dangerous_materials)

    oct_2019["dangerous"] = oct_2019.sl_private_type.isin(
        dangerous_materials
    ) | oct_2019.sl_public_type.isin(dangerous_materials)
    oct_2019["source"] = "Fast Start Exploration Status October"

    oct_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(oct_2019)
    sl_df.reset_index(inplace=True)

    ################################ December 2019

    dec_2019 = pd.read_excel(get_flint_source(dec_2019_file), engine="openpyxl")
    dec_2019.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Completion Date": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    dec_2019["pid"] = dec_2019["pid"].apply(strip_dashes).astype(int)

    dec_2019["replaced"] = (
        dec_2019["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    dec_2019 = dec_2019[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    dec_2019["hv_visit"] = False
    dec_2019["sl_visit"] = True

    dec_2019["sl_private_type"] = (
        dec_2019["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    dec_2019["sl_public_type"] = (
        dec_2019["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    dec_2019["dangerous"] = dec_2019.sl_private_type.isin(
        dangerous_materials
    ) | dec_2019.sl_public_type.isin(dangerous_materials)
    dec_2019["source"] = "Fast Start Exploration Status November"

    dec_2019.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(dec_2019)
    sl_df.reset_index(inplace=True)

    ################################ March 2020

    mar_2020 = pd.read_excel(get_flint_source(mar_2020_file), engine="openpyxl")
    mar_2020.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Completion Date": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    mar_2020["pid"] = mar_2020["pid"].apply(strip_dashes).astype(int)

    mar_2020["replaced"] = (
        mar_2020["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    mar_2020 = mar_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    mar_2020["hv_visit"] = False
    mar_2020["sl_visit"] = True

    mar_2020["sl_private_type"] = (
        mar_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    mar_2020["sl_public_type"] = (
        mar_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    mar_2020["dangerous"] = mar_2020.sl_private_type.isin(
        dangerous_materials
    ) | mar_2020.sl_public_type.isin(dangerous_materials)
    mar_2020["source"] = "Fast Start Exploration Status March 2020"

    mar_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(mar_2020)
    sl_df.reset_index(inplace=True)

    ################################ March 2020....again

    mar_2020_2 = pd.read_excel(get_flint_source(mar_2020_2_file), engine="openpyxl")
    mar_2020_2.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Completion Date": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    mar_2020_2.dropna(subset=["pid"], inplace=True)
    mar_2020_2["pid"] = mar_2020_2["pid"].apply(strip_dashes).astype(int)

    mar_2020_2["replaced"] = (
        mar_2020_2["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    mar_2020_2 = mar_2020_2[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    mar_2020_2["hv_visit"] = False
    mar_2020_2["sl_visit"] = True

    mar_2020_2["sl_private_type"] = (
        mar_2020_2["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    mar_2020_2["sl_public_type"] = (
        mar_2020_2["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    mar_2020_2["dangerous"] = mar_2020_2.sl_private_type.isin(
        dangerous_materials
    ) | mar_2020_2.sl_public_type.isin(dangerous_materials)
    mar_2020_2["source"] = "Fast Start Exploration Status March 2020_2"

    mar_2020_2.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(mar_2020_2)
    sl_df.reset_index(inplace=True)

    ################################ June 2020

    jun_2020 = pd.read_excel(get_flint_source(jun_2020_file), engine="openpyxl")
    jun_2020.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Completion Date": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    jun_2020["pid"] = jun_2020["pid"].apply(strip_dashes).astype(int)

    jun_2020["replaced"] = (
        jun_2020["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    jun_2020 = jun_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    jun_2020["hv_visit"] = False
    jun_2020["sl_visit"] = True

    jun_2020["sl_private_type"] = (
        jun_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    jun_2020["sl_public_type"] = (
        jun_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    jun_2020["dangerous"] = jun_2020.sl_private_type.isin(
        dangerous_materials
    ) | jun_2020.sl_public_type.isin(dangerous_materials)
    jun_2020["source"] = "Fast Start Exploration Status June 2020"

    jun_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(jun_2020)
    sl_df.reset_index(inplace=True)

    ################################ August 2020

    aug_2020 = pd.read_excel(get_flint_source(aug_2020_file), engine="openpyxl")
    aug_2020.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Completion Date": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    aug_2020.dropna(subset=["pid"], inplace=True)
    aug_2020["pid"] = aug_2020["pid"].apply(strip_dashes).astype(int)

    aug_2020["replaced"] = (
        aug_2020["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    aug_2020 = aug_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    aug_2020["hv_visit"] = False
    aug_2020["sl_visit"] = True

    aug_2020["sl_private_type"] = (
        aug_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    aug_2020["sl_public_type"] = (
        aug_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    aug_2020["dangerous"] = aug_2020.sl_private_type.isin(
        dangerous_materials
    ) | aug_2020.sl_public_type.isin(dangerous_materials)
    aug_2020["source"] = "Fast Start Exploration Status August 2020"

    aug_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(aug_2020)
    sl_df.reset_index(inplace=True)

    ################################ September 2020

    sep_2020 = pd.read_excel(get_flint_source(sep_2020_file), engine="openpyxl")
    sep_2020.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Replacement Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    sep_2020.dropna(subset=["pid"], inplace=True)
    sep_2020["pid"] = sep_2020["pid"].apply(strip_dashes).astype(int)

    sep_2020["replaced"] = (
        sep_2020["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    sep_2020 = sep_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    sep_2020["hv_visit"] = False
    sep_2020["sl_visit"] = True

    sep_2020["sl_private_type"] = (
        sep_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    sep_2020["sl_public_type"] = (
        sep_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    sep_2020["dangerous"] = sep_2020.sl_private_type.isin(
        dangerous_materials
    ) | sep_2020.sl_public_type.isin(dangerous_materials)
    sep_2020["source"] = "Fast Start Exploration Status September 2020"

    sep_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(sep_2020)
    sl_df.reset_index(inplace=True)

    ################################ October 2020

    oct_2020 = pd.read_excel(get_flint_source(oct_2020_file), engine="openpyxl")
    oct_2020.rename(
        {
            "Parcel ID": "pid",
            "Address of Service Line Exploration ": "Property Address",
            "Service Line Replacement Date ": "created_at",
            "Service Line Exploration Public Composition": "sl_public_type",
            "Service Line Exploration Private Composition": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    oct_2020.dropna(subset=["pid"], inplace=True)
    oct_2020["pid"] = oct_2020["pid"].apply(strip_dashes).astype(int)

    oct_2020["replaced"] = (
        oct_2020["Service Line Portion Replaced"]
        != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    oct_2020 = oct_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    oct_2020["hv_visit"] = False
    oct_2020["sl_visit"] = True

    oct_2020["sl_private_type"] = (
        oct_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    oct_2020["sl_public_type"] = (
        oct_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    oct_2020["dangerous"] = oct_2020.sl_private_type.isin(
        dangerous_materials
    ) | oct_2020.sl_public_type.isin(dangerous_materials)
    oct_2020["source"] = "Fast Start Exploration Status October 2020"

    oct_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(oct_2020)
    sl_df.reset_index(inplace=True)

    ################################ December 2020

    dec_2020 = pd.read_excel(get_flint_source(dec_2020_file), engine="openpyxl")
    dec_2020.rename(
        {
            "Location": "pid",
            "Address": "Property Address",
            "Actual Finish": "created_at",
            "Public Pipe Type": "sl_public_type",
            "Prv. Exp. Pipe Type": "sl_private_type",
        },
        axis=1,
        inplace=True,
    )

    dec_2020["pid"] = dec_2020["pid"].apply(strip_dashes).astype(int)

    dec_2020["replaced"] = (
        dec_2020["Replacement Status"] != "NO PORTION OF SERVICE LINE WAS REPLACED"
    )
    dec_2020 = dec_2020[
        ["pid", "created_at", "sl_public_type", "sl_private_type", "replaced"]
    ]

    dec_2020["hv_visit"] = False
    dec_2020["sl_visit"] = True

    dec_2020["sl_private_type"] = (
        dec_2020["sl_private_type"].astype(str).apply(str.upper).apply(str.strip)
    )
    dec_2020["sl_public_type"] = (
        dec_2020["sl_public_type"].astype(str).apply(str.upper).apply(str.strip)
    )

    dec_2020["dangerous"] = dec_2020.sl_private_type.isin(
        dangerous_materials
    ) | dec_2020.sl_public_type.isin(dangerous_materials)
    dec_2020["source"] = "Fast Start Exploration Status December 2020"

    dec_2020.set_index("pid", inplace=True)
    sl_df.set_index("pid", inplace=True)
    sl_df = sl_df.append(dec_2020)
    sl_df.reset_index(inplace=True)

    features_df = pd.read_csv(
        get_flint_source(features_filename), dtype={"PID no Dash": int}
    ).rename({"PID no Dash": "parcel_id"}, axis="columns")

    sl_df.rename(
        {"pid": "parcel_id", "dangerous": "has_lead"}, axis="columns", inplace=True
    )
    sl_df = sl_df.merge(features_df, how="right")

    rename_mapper = {
        "Year Built": "year_built",
        "HomeSEV": "home_sev",
        "Parcel Acres": "parcel_acres",
        "Housing Condition 2014": "housing_condition_2014",
    }

    data = sl_df.rename(rename_mapper, axis="columns")

    # Convert to a GeoPandas GeoDataframe and set the CRS projection.
    data = data.loc[data.Latitude > 40]
    flint_gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude)
    )
    flint_gdf = flint_gdf.set_crs(epsg=4326)  # lat/lon
    flint_gdf = flint_gdf.to_crs(2251)  # michigan state plane.

    flint_gdf = flint_gdf[
        (
            flint_gdf.sl_public_type.isin(valid_materials + [np.nan])
            | (flint_gdf.sl_private_type.isin(valid_materials + [np.nan]))
        )
    ]

    flint_gdf["sl_public_type_target"] = flint_gdf.loc[flint_gdf["sl_public_type"].notna(), "sl_public_type"].isin(dangerous_materials)
    flint_gdf["sl_private_type_target"] = flint_gdf.loc[flint_gdf["sl_private_type"].notna(), "sl_private_type"].isin(dangerous_materials)

    return flint_gdf.drop(columns=['created_at'])


# TODO move to persistance, or models
def fetch_model(model_id=None, model_dir=None, model_bucket="bc-models"):
    # Should be on disk.
    model_filepath = Path(f"{model_dir}/{model_id}.joblib").absolute()

    # If not on disk, we should be able to download to that spot.
    if not model_filepath.exists():
        # Repurpose data.get_source to fetch a model
        try:
            model_filepath = get_source(
                source_filename=model_filepath.name,
                data_dir=model_dir,
                client_bucket="bc-models",
            )
        except Exception as e:
            log.warn(
                "Tried fetching model, and failed",
                model_filename=model_filepath.name,
                model_bucket=model_bucket,
            )
            model_filepath = None

    # If we couldn't find it, well, we won't have a path, will we?
    return model_filepath


@lru_cache(maxsize=8)
def get_halifax_data(
    data_dir: Union[Path, str] = "data",
    bucket_name: str = "bc-halifax",
    building_outlines: str = "Building_Outlines-shp",
    **kwargs,
) -> pd.DataFrame:

    ### LOADING DATA

    log.info("Loading data from bc-halifax-ca bucket")

    s3_xlsx = (
        "s3://bc-halifax-ca/HWDS_service_civic_link.xlsx",
        "s3://bc-halifax-ca/HWDS_service_maintenance_1of2.xls",
        "s3://bc-halifax-ca/HWDS_service_maintenance_2of2.xls",
    )

    civic_link_, maint_1_, maint_2_ = (pd.read_excel(xlsx) for xlsx in s3_xlsx)

    gdb_layers = (
        "Service_Maintenance",
        "Service_Authoritative_Source",
        "Service_Material_Observation",
    )

    services_, service_auth_, service_matobs_ = (
        gpd.read_file("zip+s3://bc-halifax-ca/945518.gdb.zip", layer=lr)
        for lr in gdb_layers
    )

    osdata_archive = read_archive(
        "bc-halifax-ca", "ExternalData_OpenSource.zip", in_memory=True
    )

    osdata_tables = (
        "ExternalData_OpenSource/Civic_Addresses.csv",
        "ExternalData_OpenSource/Assessed_Value_and_Taxable_Assessed_Value_History.csv",
        "ExternalData_OpenSource/Residential_Dwelling_Characteristics.csv",
    )

    civic_addr_, assess_df_, dwell_df_ = (
        pd.read_csv(osdata_archive.open(osdt)) for osdt in osdata_tables
    )

    bldg_fp_ = gpd.read_file(
        get_shapefile_source(building_outlines, data_dir, bucket_name)
    )

    ### AGGREGATE AND TRANSFORM

    log.info("Working: bc-halifax-ca aggregate and transform")

    service_auth = (
        service_auth_.groupby(["DS_SERVICEID"])
        .agg({"AUTHORITATIVESOURCE": "max", "PHYSICALINSPECTION": "max"})
        .reset_index()
    )

    assess_df = assess_df_.merge(
        dwell_df_, how="left", on="Assessment Account Number", suffixes=(None, "_r")
    )

    assess_agg_coord = (
        assess_df.sort_values("Tax Year")
        .groupby(["X Map Coordinate", "Y Map Coordinate"])
        .agg(
            {
                "Assessed Value": ["first", "last", "mean", "median"],
                "Taxable Assessed Value": ["first", "last", "mean", "median"],
                "Year Built": ["first", "median"],
                "Bedrooms": ["first"],
                "Bathrooms": ["first"],
                "Square Foot Living Area": ["median"],
                "Living Units": ["first"],
            }
        )
        .reset_index()
    )

    assess_agg_coord.columns = [
        "_".join(col) for col in assess_agg_coord.columns.to_flat_index()
    ]

    assess_agg_coord["increase_pv"] = (
        assess_agg_coord["Assessed Value_first"]
        < assess_agg_coord["Assessed Value_last"]
    )

    assess_agg_coord["discount_pv"] = (
        assess_agg_coord["Taxable Assessed Value_mean"]
        < assess_agg_coord["Assessed Value_mean"]
    )

    log.info("Working: bc-halifax-ca joins")

    xy_df = services_.merge(service_auth, how="left", on="DS_SERVICEID")

    xy_df = xy_df.merge(
        civic_link_[["DS Service ID", "Civic ID"]],
        how="left",
        left_on="DS_SERVICEID",
        right_on="DS Service ID",
    )

    xy_df = xy_df.merge(civic_addr_, how="left", left_on="Civic ID", right_on="CIV_ID")

    # Generating label

    material_code = defaultdict(
        lambda: False, {999: None, 5: False, 15: True}  # unknown  # copper  # lead
    )

    xy_df["cust_lead"] = xy_df.PRIVATESERVICEMATERIAL.map(material_code)
    xy_df["muni_lead"] = xy_df.MATERIAL.map(material_code)

    # Setting non-authoritative to unknown
    xy_df.loc[(xy_df.AUTHORITATIVESOURCE != 1), ["cust_lead"]] = None
    xy_df.loc[(xy_df.AUTHORITATIVESOURCE != 1), ["muni_lead"]] = None

    xy_gdf = gpd.GeoDataFrame(xy_df, geometry=gpd.points_from_xy(xy_df.X, xy_df.Y))

    assess_gdf = gpd.GeoDataFrame(
        assess_agg_coord,
        geometry=gpd.points_from_xy(
            assess_agg_coord["X Map Coordinate_"], assess_agg_coord["Y Map Coordinate_"]
        ),
    )

    assess_gdf = gpd.sjoin(bldg_fp_, assess_gdf, how="inner", op="contains")

    xy_gdf = gpd.sjoin(
        xy_gdf, assess_gdf.drop(columns="index_right"), how="left", op="within"
    )

    target_columns = {
        "CURBTIE2DISTANCE": "curb_tie_dist_2",
        "CURBTIE1DISTANCE": "curb_tie_dist_1",
        "MAINPIPEKEY": "main_key",
        "MAINDEPTH": "main_depth",
        "REGION": "region",
        "X": "lon",
        "Y": "lat",
        "Assessed Value_mean": "mean_av",
        "Assessed Value_median": "median_av",
        "SHAPE_Area": "shape_area",
        "Year Built_first": "yrbuilt",
        "Bedrooms_first": "n_bed",
        "Bathrooms_first": "n_bath",
        "Square Foot Living Area_median": "sqft",
        "Living Units_first": "n_units",
    }

    xy_gdf.rename(
        columns=target_columns,
        inplace=True,
    )

    xy_gdf = (
        xy_gdf.groupby("DS_SERVICEID")
        .agg(
            {
                "curb_tie_dist_1": "median",
                "curb_tie_dist_2": "median",
                "main_depth": "median",
                "lon": "median",
                "lat": "median",
                "region": "first",
                "main_key": "first",
                "geometry": "first",
                "cust_lead": "max",
                "muni_lead": "max",
                "mean_av": "mean",
                "median_av": "median",
                "increase_pv": "first",
                "discount_pv": "first",
                "shape_area": "median",
                "yrbuilt": "median",
                "n_bed": "median",
                "n_bath": "median",
                "sqft": "median",
            }
        )
        .reset_index()
    )

    ####
    # Updating Labels
    ####

    maint = pd.concat([maint_1_, maint_2_], ignore_index=True)
    maint["workyear"] = pd.to_datetime(maint.WorkDate, errors="coerce").dt.year

    lead_rep_df = (
        maint[maint.workyear > 1960]
        .groupby("DS Service ID")
        .agg({"Public Lead Replacement?": "max", "Private Lead Replacement?": "max"})
        .reset_index()
    )

    xy_gdf = pd.merge(
        xy_gdf,
        lead_rep_df,
        how="left",
        left_on="DS_SERVICEID",
        right_on="DS Service ID",
    )

    # Switching labels for rows with replacements
    xy_gdf.loc[
        ((xy_gdf["Private Lead Replacement?"] == "Yes") & (xy_gdf.cust_lead == False)),
        ["cust_lead"],
    ] = True
    xy_gdf.loc[
        ((xy_gdf["Public Lead Replacement?"] == "Yes") & (xy_gdf.muni_lead == False)),
        ["muni_lead"],
    ] = True

    xy_gdf = gpd.GeoDataFrame(xy_gdf, geometry=xy_gdf.geometry)

    return xy_gdf


# TODO Include toledo matching logic from source files
#      currently pinning to cached csv
# TODO Expose all meaningful features in this data
@to_processed_dir
def get_toledo_data(
    data_dir: Union[Path, str] = "data",
    bucket_name: str = "bc-toledo",
    # lead_services_shapefile: str = "LeadServices",
    # non_lead_services_shapefile: str = "NonLeadServices",
    # city_outline_shapefile: str = "CotOutline",
    toledo_census_csv: str = "toledo_census_downscaled.csv",
    service_lateral_shapefile: str = "CotServiceLaterals",
    parcels_areis_shapefile: str = "toledo_parcels_cot_areis",
    water_assets_csv: str = "water_assets_matched_3_24_21.csv",
    parcels_general_csv: str = "parcels_general.csv",
    parcels_residential_csv: str = "parcels_residential.csv",
    resolved_matches_csv: str = "resolved_matches_3_2021.csv",
    census_features_csv: str = "census_features.csv",
    hvac_sample1_csv: str = "toledo_random_sample_3_3_2020.csv",
    hvac_sample2_csv: str = "toledo_random_sample_5_19_21.csv",
    hvac_results_csv: str = "hvac_toledo_5_11_21.csv",
    replacements_csv: str = "replacements_list.csv",
    **kwargs,
) -> pd.DataFrame:
    def ecdf(data, column, idcol):
        data1 = data[[column, idcol]].dropna(subset=[column])
        data1.sort_values(column, inplace=True)
        data1.reset_index(inplace=True)

        n = len(data1)
        y = np.arange(1, n + 1) / n

        data1[column + "_ecd"] = y
        return data.merge(data1[[idcol, column + "_ecd"]], on=idcol, how="left")

    def get_inst_year(datestr):
        """Helper for SL install year - parse year from date"""
        return float(datestr.split("-")[0]) if datestr else np.nan

    def get_pid(i):
        """Helper for int parcels -- pad to known length id"""
        return str(i) if len(str(i)) == 7 else "0" + str(i)

    def getSTREET(x):
        """Helper for getting street name from historical records"""
        return x.split(" ")[1] if len(x.split(" ")) == 3 else x.split(" ")[0]

    # NOTE use get_toledo_source to download single files
    get_toledo_source = lambda sfn: get_source(
        source_filename=sfn, data_dir=data_dir, client_bucket=bucket_name
    )
    wsl_data = gpd.read_file("https://cityworks.toledo.oh.gov/arcgis443/rest/services/Cityworks/WaterOperational/MapServer/24/query?where=1%3D1&text=&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&relationParam=&outFields=AssetID%2C+Material%2C+Owner%2C+Address%2C+Status%2C+Diameter%2C+Length%2C+WO%2C+Mat_Cust&returnGeometry=true&returnTrueCurves=false&maxAllowableOffset=&geometryPrecision=&outSR=&having=&returnIdsOnly=false&returnCountOnly=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&returnZ=false&returnM=false&gdbVersion=&historicMoment=&returnDistinctValues=false&resultOffset=&resultRecordCount=&queryByDistance=&returnExtentOnly=false&datumTransformation=&parameterValues=&rangeValues=&quantizationParameters=&featureEncoding=esriDefault&f=geojson")

    wsl_data.rename(columns={"Material":"Material_live"}, inplace=True)

    # NOTE use get_shapefile_source to download shapefiles
    lsli = gpd.read_file(
        get_shapefile_source(service_lateral_shapefile, data_dir, bucket_name)
    )

    for own, col in [("COT", "public_material"), ("PRVT", "private_material")]:
        df = lsli[lsli.Owner == own]
        lsli.loc[df.index, col] = df.Material

    lsli["InstallYear"] = lsli["Installati"].apply(get_inst_year)
    # fix crazy install years
    idx = lsli[lsli["InstallYear"] < 1800].index[-1]
    lsli.at[idx, "InstallYear"] = 2008.0
    idx = lsli[lsli["InstallYear"] > 2020].index
    lsli.loc[idx, "InstallYear"] = 2020.0

    lsli.loc[lsli.Diameter == 0, "Diameter"] = np.nan
    lsli.loc[lsli.Length == 0, "Length"] = np.nan

    # Left merge lsli to matches
    matches = pd.read_csv(get_toledo_source(resolved_matches_csv))

    matches["parcel_id"] = matches["parcel_id"].apply(get_pid)
    lsli = pd.merge(
        lsli[
            [
                "AssetID",
                "Address",
                "Material",
                "public_material",
                "private_material",
                "InstallYear",
                "Diameter",
                "Length",
                "Status",
                "Owner",
                "Plan_",
                "geometry",
            ]
        ],
        matches,
        on="AssetID",
        how="left",
    )

    # Sort by installation year and drop duplicate service line records
    n_pre = len(lsli)
    lsli.drop(
        lsli[(lsli.AssetID.duplicated(keep=False)) & (lsli.Status == "K")].index,
        inplace=True,
    )
    lsli.sort_values("InstallYear", inplace=True)
    lsli.drop_duplicates(subset=["AssetID"], keep="first", inplace=True)
    n_post = len(lsli)
    log.warn("Dropping Duplicates for lsli matches", n_pre=n_pre, n_post=n_post)

    print("Reading LSL Replacements...", end="\r")
    slr_data = pd.read_csv(get_toledo_source(replacements_csv))
    slr_data["lslr_date"] = slr_data["Completed Date"].astype(str)
    lsli = lsli.merge(
        slr_data[["AssetID", "lslr_date"]].drop_duplicates(subset=["AssetID"]),
        on="AssetID",
        how="left",
    )
    # read in live updates
    live_updates = gpd.read_file("https://cityworks.toledo.oh.gov/Cityworks/gis/3/6373/rest/services/cw/FeatureServer/2/query?f=shapefile")
    live_updates = live_updates.drop_duplicates(subset=["WoAddress"])
    live_updates = live_updates[live_updates.Status.isin(["COMPLETED","CLOSED"])]
    live_updates=live_updates.reset_index(drop=True)
    live_updates["WoAddress"] = [i[0] for i in live_updates.WoAddress.str.upper().str.split(",")]
    idx = ~live_updates.WoAddress.isin(lsli.Address)
    live_updates.loc[idx, "WoAddress"] = pd.Series(
        [i[:2] for i in live_updates[idx]["WoAddress"].str.split(" ")]
    ).str.join(" ")
    live_updates = live_updates.dropna(subset=["WoAddress"])
    
    lslr_live = pd.merge(
        live_updates[["WoAddress","WOID","InitDate"]], 
        wsl_data[["Address","AssetID"]], 
        left_on="WoAddress", right_on="Address", 
        how="left"
    )
    lsli = lsli.merge(
        lslr_live[["AssetID","InitDate"]], 
        on="AssetID", 
        how="left"
    )
    
    lsli.loc[
        (lsli.lslr_date.notna())|(lsli.AssetID.isin(lslr_live.AssetID)),
        "replaced"
    ] = True
    lsli.loc[
        lsli[(lsli.replaced == True)].index,
        ["verified_public", "verified_private"],
    ] = "COPP"
    lsli["verified"] = lsli.replaced == True
    
    lsli.loc[
        (lsli.lslr_date.isna())&(lsli.InitDate.notna()),
        "lslr_date"
    ] = lsli[
        (lsli.lslr_date.isna())&(lsli.InitDate.notna())
    ].InitDate
    
    def lookup(df, column, str_num, str_name="none"):
        df1 = df.copy()
        df1["lookup1"] = [(str_num in str(i)) for i in df1[column]]
        if str_name == "none":
            str_name = str_num

        df1["lookup2"] = [(str_name in str(i)) for i in df1[column]]
        return df1[(df1["lookup1"] == True) & (df1["lookup2"] == True)].index

    def update_records(data, idx, pubmat, primat):
        if "LEAD" in str(pubmat).upper():
            pubmat = "LEAD"
        elif "IRON" in str(pubmat).upper():
            pubmat = "IRON"
        else:
            pubmat = "COPP"
        if "LEAD" in str(primat).upper():
            primat = "LEAD"
        elif "IRON" in str(pubmat).upper():
            primat = "IRON"
        else:
            primat = "COPP"
        if len(idx) > 0:
            data.loc[idx, "hvac_visit"] = True
            data.loc[idx, "verified"] = True
            data.loc[idx, "verified_public"] = pubmat
            data.loc[idx, "verified_private"] = primat
            data.loc[idx, "has_lead"] = pubmat == "LEAD" or primat == "LEAD"
            match = (
                pubmat == data.public_material[idx[0]]
                or primat == data.private_material[idx[0]]
            )
            data.loc[idx, "records_match"] = match

    print("Reading HVAC inspections...", end="\r")
    hvac_list = pd.read_csv(get_toledo_source(hvac_sample1_csv))
    hvac_list1 = pd.read_csv(get_toledo_source(hvac_sample2_csv))
    cols = ["AssetID", "Address"]
    hvac_list = pd.concat([hvac_list[cols], hvac_list1[cols]])
    hvac_list["ADDRESS"] = [i.split(" ")[0] for i in hvac_list["Address"]]

    hvac_list["STREET"] = [
        i.replace(i.split(" ")[0] + " ", "") for i in hvac_list["Address"]
    ]
    hvac_list["STREET"] = hvac_list.STREET.apply(getSTREET)

    hvac_results = pd.read_csv(get_toledo_source(hvac_results_csv))
    hvac_results["ADDRESS"] = hvac_results.ADDRESS.astype(str)
    hvac_results = pd.merge(
        hvac_results,
        hvac_list.drop_duplicates(subset=["ADDRESS"], keep="first")[
            ["AssetID", "ADDRESS"]
        ],
        on="ADDRESS",
        how="left",
    )

    for i in hvac_results[hvac_results.AssetID.isna()].index:
        test_df = hvac_results.loc[[i]].merge(
            hvac_list[~hvac_list.AssetID.isin(hvac_results.AssetID)], on="STREET"
        )
        if len(test_df) > 0:
            test_address = test_df.Address[0].replace(
                str(test_df.ADDRESS_x[0]), str(test_df.ADDRESS_y[0])
            )
            hvac_results.loc[i, "AssetID"] = (
                lsli[lsli.Address == test_address].AssetID.describe().top
            )
    c = 1
    for n in hvac_results.dropna(subset=["AssetID", "CITY'S MATERIAL"]).index:
        print(f"Updating Records        ...{c}", end="\r")
        c += 1
        pubmat, primat = hvac_results["CITY'S MATERIAL"][n], hvac_results["PRIVATE"][n]
        id_ = hvac_results.AssetID[n]
        idx = lookup(lsli, "AssetID", id_)
        update_records(lsli, idx, pubmat, primat)  # , split)
        # else:
        #    idx = lookup(lsli, "Address",hvac_results.ADDRESS[n],hvac_results.STREET[n])
        #    update_records(lsli, idx, pubmat, primat)

    lsli["hist_has_lead"] = lsli.Material.isin(["LEAD", "LDCP"]).astype(int).to_numpy()
    lsli.loc[lsli.Material.isna(), "hist_has_lead"] = np.nan

    lsli["verified_lead"] = (lsli.verified_public == "LEAD") | (
        lsli.verified_private == "LEAD"
    )
    bad_install_year = (lsli.InstallYear < 1980) | lsli.InstallYear.isna()
    lsli["has_lead"] = lsli.verified_lead | (lsli.replaced & bad_install_year)
    lsli.loc[~lsli.verified, "has_lead"] = np.nan

    # n = lsli.has_lead.value_counts()[1.0]
    # lsli.loc[(lsli.records_match)&(lsli.Material=="LEAD"), "has_lead"] = 0.0
    # print(f"{n - lsli.has_lead.value_counts()[1.0]}")

    lsli = gpd.GeoDataFrame(lsli)
    xy = lsli.geometry.centroid
    lsli["lon"], lsli["lat"] = xy.x, xy.y
    print("Total verified lead:", lsli.verified_lead.value_counts()[True], end="\r")

    print("Merging in water system data...       ", end="\r")

    wtr_assets_matched = pd.read_csv(get_toledo_source(water_assets_csv))
    wtr_assets_matched.drop_duplicates(subset=["AssetID"], inplace=True)
    lsli = lsli.merge(wtr_assets_matched, on="AssetID", how="left")
    lsli["MainID"] = lsli.MainID.fillna("UNK")
    parcels = gpd.read_file(
        get_shapefile_source(parcels_areis_shapefile, data_dir, bucket_name)
    )
    residential = pd.read_csv(
        get_toledo_source(parcels_residential_csv), usecols=["Parcel", "Occupancy"]
    )
    residential["vacant"] = residential.Occupancy.isin((1, 8))
    residential["onefam"] = residential.Occupancy == 2
    residential["twofam"] = residential.Occupancy == 3
    residential["parcel_id"] = residential.Parcel.apply(get_pid)
    parcels = parcels.merge(
        residential[["parcel_id", "vacant", "onefam", "twofam", "Occupancy"]],
        on="parcel_id",
        how="inner",
    )
    parcels.drop_duplicates(subset=["parcel_id"], inplace=True)

    n_pre = len(lsli)
    toledo_data = lsli.merge(
        parcels.drop(columns=["geometry"]), on="parcel_id", how="left"
    )
    n_post = len(toledo_data)
    log.warn("Merging parcel data...", n_pre=n_pre, n_post=n_post)

    for col in ["YearBlt", "Rooms", "BedRms", "Grade", "Cond"]:
        toledo_data[col] = toledo_data[col].astype(float)
        idx = toledo_data[toledo_data[col] == 0].index
        toledo_data.loc[idx, col] = np.nan

    # Census data
    tdo_bg = pd.read_csv(get_toledo_source(toledo_census_csv))
    for col in tdo_bg.columns:
        if "'" in col:
            tdo_bg.rename(columns={col: col.replace("'", "")}, inplace=True)
    toledo_data = toledo_data.merge(tdo_bg, on="AssetID", how="left")

    feature_columns = [
        "InstallYear",
        "Diameter",
        "Length",
        "dist_to_hyd",
        "MainDate",
        "dist_to_Main",
        "MainConnections",
        "dist_to_Valve",
        "dist_to_SvcValve",
        "YearBlt",
        "Lotsize",
        "Frontage",
        "Depth",
        "Grade",
        "Land",
        "Total",
        "LatestSale",
        "TotalSales",
        "CurrSaleMa",
        "estimate_median_household_income_in_the_past_12_months_(in_2019_inflation-adjusted_dollars)",
        "estimate_aggregate_contract_rent",
        "estimate_median_value_(dollars)",
        "estimate_percent_white_alone",
        "estimate_percent_black_or_african_american_alone",
        "estimate_percent_no_diploma_or_ged",
        "estimate_percent_associates_degree",
        "estimate_percent_built_1939_or_earlier",
        "estimate_percent_vacant",
    ]
    for column in feature_columns:
        toledo_data[column].fillna(-1, inplace=True)

    toledo_data['domainid'] = toledo_data.AssetID
    toledo_data['has_lead'] = toledo_data.has_lead.map({np.nan:None, True: True, False:False})
    toledo_data['safe'] = toledo_data.has_lead.map({False:True, True:False, None:None})

    return gpd.GeoDataFrame(toledo_data, geometry=toledo_data.geometry.centroid).to_crs('epsg:4326')


@to_processed_dir
def get_detroit_data(
    data_dir: Union[Path, str] = "data",
    bucket_name: str = "bc-detroit",
    water_service_meter: str = "WaterServiceMeter",
    water_stop_box: str = "WaterStopBox",
    parcels_shp: str = "detroit_parcels_data",
    parcels_feather: str = "detroit_parcels.feather",
) -> pd.DataFrame:

    wsm_gdf = gpd.read_file(
        get_shapefile_source(water_service_meter, data_dir, bucket_name)
    )
    wsb_gdf = gpd.read_file(get_shapefile_source(water_stop_box, data_dir, bucket_name))

    parcel_gdf = gpd.read_feather(get_source(parcels_feather, data_dir, bucket_name))

    # Reprojecting into common crs
    parcel_gdf = parcel_gdf.set_crs("epsg:4326").to_crs("epsg:2898")

    numeric_cols = [
        "sale_price",
        "year_built",
        "depth",
        "frontage",
        "total_squa",
        "total_acre",
    ]

    for col in numeric_cols:
        parcel_gdf[col] = pd.to_numeric(parcel_gdf[col])

    # Attribute join on parcel number
    xy_gdf = wsb_gdf.merge(parcel_gdf, left_on="PARCELNUMB", right_on="parcel_num")
    xy_gdf = gpd.GeoDataFrame(xy_gdf, geometry="geometry_x")

    xy_gdf["has_lead"] = np.nan

    xy_gdf.loc[
        (xy_gdf.HISTORICDW == "Lead") | (xy_gdf.HISTORICCU == "Lead"), ["has_lead"]
    ] = True
    xy_gdf.loc[
        (xy_gdf.HISTORICDW == "Copper") & (xy_gdf.HISTORICCU == "Copper"), ["has_lead"]
    ] = False

    xy_gdf["known"] = False

    xy_gdf.loc[
        (xy_gdf.BURIEDCUST.notna()) & (xy_gdf.BURIEDDWSD.notna()), ["known"]
    ] = True

    xy_gdf["partial_known"] = False

    xy_gdf.loc[
        (xy_gdf.BURIEDCUST.notna()) | (xy_gdf.BURIEDDWSD.notna()), ["partial_known"]
    ] = True

    xy_gdf = xy_gdf.to_crs('epsg:4326')

    xy_gdf["lat"] = xy_gdf.geometry_x.y
    xy_gdf["lon"] = xy_gdf.geometry_x.x

    agg_rules_ = {col: "mean" for col in numeric_cols}

    first_cols = [
        "has_lead",
        "geometry_x",
        "lat",
        "lon",
        "known",
        "partial_known",
        "parcel_num",
        "address",
        "zip_code",
        "parcel_num",
    ]

    agg_first_ = {col: "first" for col in first_cols}

    agg_rules_ = dict(agg_rules_, **agg_first_)

    xy_gdf = xy_gdf.groupby("CIPMOID").agg(agg_rules_).reset_index()

    xy_gdf = gpd.GeoDataFrame(xy_gdf, geometry="geometry_x")

    xy_gdf["safe"] = xy_gdf.has_lead.map(
        {True: False, False: True, None: None, np.nan: None}
    )

    xy_gdf["domainid"] = xy_gdf.CIPMOID
    xy_gdf["geometry"] = xy_gdf.geometry_x

    return xy_gdf


def get_trenton_data(
    data_dir: Union[str, Path] = "data",
    bucket_name: str = "bc-trenton",
    wsa_shapefile_folder: str = "WaterServiceAccounts_071320",
    parcel_shapefile_folder: str = "MERCER_Mod4_parcels",
    addr_shapefile_folder: str = "Addr_NG911_MERCER",
    material_assumptions_filename: str = "Material_Assumptions_AllMunis_KristinsSpreadsheets_1.csv",
) -> pd.DataFrame:

    data_dir = Path(data_dir) / bucket_name

    # wsa_df = gpd.read_file(
    #     get_shapefile_source(wsa_shapefile_folder, data_dir, bucket_name)
    # )

    # mercer_parcels = gpd.read_file(
    #     get_shapefile_source(parcel_shapefile_folder, data_dir, bucket_name)
    # )

    # nj_addrs = gpd.read_file(
    #    get_shapefile_source(addr_shapefile_folder, data_dir, bucket_name)
    # ).rename({'OBJECTID':'OBJECTID_addr'}, axis=1)

    # mercer_parcels = gpd.sjoin(nj_addrs, mercer_parcels, how = 'right',op='within', rsuffix = '_addr').drop(['index_left','index__addr'], axis=1, errors='ignore')​
    # shortcut to here
    # Get source is kind of verbose. Here's something easier on the eyes
    get_trenton_source = lambda sfn: get_source(
        source_filename=sfn, data_dir=data_dir, client_bucket=bucket_name
    )

    # TODO this is a shortcut, hardcoded; I should fix it later
    df = pd.read_csv(get_trenton_source("df.csv"))

    material_assumptions = gpd.read_file(
        get_trenton_source("Material_Assumptions_AllMunis_KristinsSpreadsheets_1.csv")
    )

    # TODO: Had to temporarily delete METER_MODEL, TWW_dangerous, and GIS_SERVICE_YEAR
    categorical_columns = [
        "BLDG_CLASS",
        "BLDG_DESC",
        "CD_CODE",
        "CorpStopSize",
        "DWELL",
        "FAC_NAME",
        "hasWell",
        "INC_MUNI",
        "LAND_DESC",
        "LST_TYPE",
        "LST_PREDIR",
        "LSTPOSDIR",
        "MailingStatus",  #'METER_MODEL',
        "MUN_NAME",
        "MUNI",
        "NewCorpStop",
        "NewServiceLineDiameter",
        "Neighborhood",
        "OWNER_NAME",
        "PCL_MUN",
        "PLACE_TYPE",
        "PLACEMENT",
        "POST_COMM",
        "POST_CODE",
        "POST_CODE4",
        "PorpClass",
        "PropertyType",
        "PROP_CLASS",
        "PROP_USE",
        "SALES_CODE",
        "ServiceArea",
        "SizeTxt",
        "SidewalkSurface",
        "StreetSurface",
        "ST_PRETYP",
        "ST_POSTYP",
        "ST_POSMOD",
        "TWWCombined",  #'TWW_dangerous',
        "ContractorObsMaterialTWW",
        "TWW_Replaced_YorN",
        "TWWFinalMat",
        "GIS_TWW_MA",
        "TWWMT",
        "TWW_INF",
        "TWW_COMBIN",
        "ZIP5",
        "ZIP_CODE",
    ]

    numerical_columns = [
        "CALC_ACRE",
        #'GIS_SERVICE_YEAR',
        "IMPRVT_VAL",
        "LAND_SF",
        "LAND_VAL",
        "LAST_YR_TX",
        "MainSize",
        "NET_VALUE",
        "PROP_USE",
        "SALE_PRICE",
        "SHAPE_Area",
        "SHAPE_Leng",
        "Serv_YR",
        "ServiceLineLength",
        "SizeNumb",
        "TopOfMain",
        "YR_BUILT",
        "YR_CONSTR",
    ]

    # Cast to string to avoid mixed floats and strings
    df.loc[:, categorical_columns] = df[categorical_columns].astype(str)

    data = (
        df[categorical_columns + numerical_columns + ["UPROPID", "dangerous"]]
        .rename({"UPROPID": "parcel_id", "dangerous": "has_lead"}, axis="columns")
        .drop_duplicates()
    )

    # Our shortcut has resulted in some duplicate columns:
    data = data.loc[:, ~data.columns.duplicated()]

    import numpy as np

    for col in numerical_columns:
        data.loc[data[col] == -1, col] = np.nan

    return data


def get_charlotte_data(
    data_dir: Union[Path, str] = "data",
    bucket_name: str = "bc-charlotte",
    parcels: str = "mecklenburg_parcels",
) -> pd.DataFrame:

    data_dir = Path(data_dir) / bucket_name

    parcel_gdf = gpd.read_file(get_shapefile_source(parcels, data_dir, bucket_name))

    return parcel_gdf


def get_fayetteville_data(
    data_dir: Union[Path, str] = "data",
    bucket_name: str = "bc-fayetteville-nc",
    parcels: str = "cumberland_parcels",
) -> pd.DataFrame:

    data_dir = Path(data_dir) / bucket_name

    parcel_gdf = gpd.read_file(get_shapefile_source(parcels, data_dir, bucket_name))

    return parcel_gdf

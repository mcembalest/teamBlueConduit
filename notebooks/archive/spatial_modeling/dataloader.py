import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit, RepeatedStratifiedKFold


def get_data():

    # Shape files truncate column names.  This dictionary is used to restore the whole names.

    col_name_dictionary = {'pid': 'pid', 'Property Z': 'Property Zip Code', 'Owner Type': 'Owner Type',
                           'Owner Stat': 'Owner State', 'Homestead': 'Homestead', 'Homestea_1': 'Homestead Percent',
                           'HomeSEV': 'HomeSEV', 'Land Value': 'Land Value', 'Land Impro': 'Land Improvements Value',
                           'Residentia': 'Residential Building Value', 'Resident_1': 'Residential Building Style',
                           'Commercial': 'Commercial Building Value', 'Building S': 'Building Storeys',
                           'Parcel Acr': 'Parcel Acres', 'Rental': 'Rental', 'Use Type': 'Use Type',
                           'Prop Class': 'Prop Class', 'Old Prop c': 'Old Prop class', 'Year Built': 'Year Built',
                           'USPS Vacan': 'USPS Vacancy', 'Zoning': 'Zoning', 'Future Lan': 'Future Landuse',
                           'DRAFT Zone': 'DRAFT Zone', 'Housing Co': 'Housing Condition 2012',
                           'Housing _1': 'Housing Condition 2014', 'Commerci_1': 'Commercial Condition 2013',
                           'Latitude': 'Latitude', 'Longitude': 'Longitude', 'Hydrant Ty': 'Hydrant Type',
                           'Ward': 'Ward', 'PRECINCT': 'PRECINCT', 'CENTRACT': 'CENTRACT', 'CENBLOCK': 'CENBLOCK',
                           'SL_Type': 'SL_Type', 'SL_Type2': 'SL_Type2', 'SL_Lead': 'SL_Lead', 'Ed_July': 'Ed_July',
                           'Ed_March': 'Ed_March', 'Last_Test': 'Last_Test', 'Max_Lead': 'Max_Lead',
                           'Med_Lead': 'Med_Lead', 'Num_Tests': 'Num_Tests', 'Res_Test': 'Res_Test',
                           'Sen_Test': 'Sen_Test', 'SL_private': 'SL_private_inspection',
                           'B_median_a': 'B_median_age_all_women', 'B_median_1': 'B_median_age_all_men', 
                           'B_median_2': 'B_median_age_all', 'B_median_3': 'B_median_age_all_women_white',
                           'B_median_4': 'B_median_age_all_men_white', 'B_median_5': 'B_median_age_all_white',
                           'B_median_6': 'B_median_age_all_women_black', 'B_median_7': 'B_median_age_all_men_black',
                           'B_median_8': 'B_median_age_all_black', 'B_total_bl': 'B_total_black_pop',
                           'B_total_wh': 'B_total_white_pop', 'B_married_': 'B_married_couples',
                           'B_single_w': 'B_single_women', 'B_marrie_1': 'B_married_couples_white',
                           'B_single_1': 'B_single_women_white', 'B_marrie_2': 'B_married_couples_black',
                           'B_single_2': 'B_single_women_black', 'B_marrie_3': 'B_married_couples_w_children',
                           'B_single_m': 'B_single_mothers_w_children', 'B_househol': 'B_households_w_elderly',
                           'B_househod': 'B_househod_no_elderly', 'B_aggregat': 'B_aggregate_income',
                           'B_speak_sp': 'B_speak_spanish', 'B_speak_on': 'B_speak_only_english',
                           'B_no_engli': 'B_no_english', 'B_hispanic': 'B_hispanic_household',
                           'B_imputed_': 'B_imputed_rent', 'B_impute_1': 'B_imputed_value',
                           'known_priv': 'known_private_sl', 'known_publ': 'known_public_sl', 'hydrovac': 'hydrovac',
                           'sl_priva_1': 'sl_private_type', 'sl_public_': 'sl_public_type', 'created_at': 'created_at',
                           'source': 'source', 'hv_visit': 'hv_visit', 'sl_visit': 'sl_visit', 'replaced': 'replaced',
                           'dangerous': 'dangerous', 'geometry': 'geometry'}

    # Update path to the new shapefile I sent.

    sl_df = gpd.read_file('../data/flint_sl_materials/flint_sl_materials.shx')
    sl_df = sl_df.rename(col_name_dictionary, axis=1)

    # Drop columns that aren't used by the classifier

    drop_cols = ['known_private_sl', 'known_public_sl', 'hydrovac', 'created_at', 'source',
                 'hv_visit', 'sl_visit', 'replaced', 'geometry']

    data = sl_df.drop(drop_cols, axis = 1)

    # Only keep labelled data
    data = data[~pd.isnull(data.dangerous)].reset_index()





    continuous_features = ['Latitude', 'Longitude']
    discrete_features = ['PRECINCT']
    target = ['dangerous']

    training_data = data[continuous_features + discrete_features + target]
    # for now we just ignore missingness because for our toy features, only 6 rows dropped
    training_data = training_data.dropna()
    #training_data = training_data.loc[training_data['Year Built'] > 1850]
    # print('Number of rows dropped due to missingness:', data.shape[0] - training_data.shape[0])

    return training_data


def get_train_test(training_data, n_data, random_state=297):


    continuous_features = ['Latitude', 'Longitude']
    discrete_features = ['PRECINCT']
    target = ['dangerous']

    
    sample = training_data.sample(n=n_data, random_state=random_state)

    Xdata = sample[continuous_features + discrete_features]
    Ydata = sample[target]

    # Groups for spatial cross validation
    groups = Xdata['PRECINCT']
    Xdata = Xdata.drop('PRECINCT', axis=1)

    # Group Shuffle Split example.  train_test_split could go here, but we like spatial cross validation 
    # better than a uniform random sample.
    train_idx, test_idx = list(GroupShuffleSplit(n_splits=1, train_size=.75, random_state=297).split(Xdata, Ydata, groups))[0]

    Xtrain = Xdata.iloc[train_idx].values
    Xtest = Xdata.iloc[test_idx].values
    Ytrain = Ydata.iloc[train_idx.tolist()].values.flatten().astype(int)
    Ytest = Ydata.iloc[test_idx.tolist()].values.flatten().astype(int)
    
    return Xtrain, Xtest, Ytrain, Ytest
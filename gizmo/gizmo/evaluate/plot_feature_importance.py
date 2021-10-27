import seaborn as sns
from matplotlib import pyplot as plt

import re



from gizmo.evaluate.feature_importance import (
    extract_feature_importances_as_df,
    # get_estimator_name,
    # get_numeric_feature_names,
    # get_categorical_feature_names,
    # get_feature_names,
)



def make_ax_feat_imp_plots(model):

    importances_df = extract_feature_importances_as_df(model)

    importances_cols = [col for col in importances_df if
        is_importance_type_col(col)]

    importances_col_names = {col: get_importance_type_name(col) for
        col in importances_cols}


    ax_plotted_list = []
    # for importance_col, importance_type in importances_col_names.items():
    #     print(importance_col, importance_type)

    return importances_col_names

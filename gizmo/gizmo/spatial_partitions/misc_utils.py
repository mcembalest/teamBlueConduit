import numpy as np
import pandas as pd


def get_df_memory_usage_str(df):
    bytes = df.memory_usage(deep=False).sum()
    if bytes < 1:
        bytes = 1
    power = int(np.log10(bytes))
    if power > 2 and power < 6:
        SI = "KB"
    elif power > 5 and power < 9:
        SI = "MB"
    elif power > 8 and power < 12:
        SI = "GB"
    else:
        SI = "e+0" + str(power) + " bytes"

    return "{:.2f} {}".format(bytes / (10 ** (power - power % 3)), SI)

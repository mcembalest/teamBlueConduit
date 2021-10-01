import numpy as np
import geopandas as gpd 

class DistanceMatrix:
    """From a Geopandas dataframe, will create a distance matrix
    from either a Euclidean or Road Network perspective. Will
    instantiate with the raw dataframe to utilize"""
    def __init__(self, method):
        self.method = method
    
    def calculate_euclidean_distance

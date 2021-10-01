import numpy as np
import geopandas as gpd 
import pickle
import pandas as pd

class DistanceMatrix:
    """From a Geopandas dataframe, will create a distance matrix
    from either a Euclidean or Road Network perspective. Will
    instantiate with the raw dataframe to utilize"""

    def __init__(self, method='euclidean'):
        self.method = method

    @classmethod
    def fromSaved(cls, filepath):
        """Creates and unpacks a fit distance matrix previously calculated"""
        value_dict = pd.read_pickle(filepath)
        self.method = value_dict['method']
        self.units = value_dict['units']
        self.distance_matrix = value_dict['distance_matrix']
        self.parcel_id2idx = value_dict['parcel_id2idx']
        self.idx2parcel_id = value_dict['idx2parcel_id']
    
    def fit(self, geo_df, n=None, parcel_id='pid'):
        """Will calculate the distance matrix according to the supplied.
        Note that this anticipates a geopandas dataframe as the base
        data structure.

        Inputs:
            - geo_df (geopandas.DataFrame): the raw geopandas dataframe
                        to calcualte the distance matrix for.
            - n (int): limit of how many entries to calculate the distance
                        matrix over
            - parcel_id (str): Parcel ID field in GDF if not "pid"

        Returns:
            None
        """
        self.parcel_id2idx = {}
        self.idx2parcel_id = {}
        self.units = geo_df.crs.axis_info[0].unit_name

        df = geo_df.copy()
        # Subset out all parcels which are missing a geometry
        df = df[df['geometry'].isna() == False]

        base_array = df['geometry'].values
        copied_array = base_array.copy(deep=True)
        rows_truncated = []

        # Walk through each parcel and calculate pairwise distance
        # to every other parcel
        for i, a in enumerate(base_array):
            if i % 1000 == 0:
                print(i)
            # Add to the dictionaries
            self.parcel_id2idx[df.iloc[i][parcel_id]] = i
            self.idx2parcel_id[i] = df.iloc[i][parcel_id]
            
            distances = copied_array[i:].distance(a)
            rows_truncated.append(distances)

        # rows_truncated will be upper triangular matrix; convert into 
        # distance matrix
        max_len = rows_truncated[0].shape[0]
        upper_matrix = []
        for i, x in enumerate(rows_truncated):
            if i + x.shape[0] != max_len:
                raise ValueError

            # To get a "full" row, pad w/zeros to ensure
            # same length
            full_row = np.concatenate([np.zeros(i), x])
            upper_matrix.append(full_row)

        upper_matrix = np.array(upper_matrix)
        # Distance matrix is upper matrix + transpose (i.e. lower matrix)
        # Subtract out diagonal since o/w will double add (should be zero anyways)
        distance_matrix = upper_matrix + upper_matrix.T - np.diag(upper_matrix.diagonal())

        # Sanity check self-distances are zero
        if distance_matrix.diagonal().sum() != 0:
            raise Exception(f"Non-zero self distances, check procedure")

        self.distance_matrix = distance_matrix
    
    def save_distance_matrix(self, filepath):
        """Saves data out as a pickled file to the relative filepath
        
        Inputs:
            filepath: relative filepath. Should be a .pkl file

        Returns:
            None"""
        output = dict(
            parcel_id2idx = self.parcel_id2idx,
            idx2parcel_id = self.idx2parcel_id,
            distance_matrix = self.distance_matrix,
            units = self.units,
            method = self.method
        )

        with open(filepath, 'wb') as outfile:
            pickle.dump(output, outfile)
        outfile.close()





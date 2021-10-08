import requests
from blue_conduit_spatial.utilities import *
import geopandas as gpd

class RoadDistanceMatrix:
    def __init__(self, N, df):
        self.road_dist_arr = self._create_road_dist_array(N)
        self.N = self.road_dist_arr.shape[0]
        self.lat_long_df = df[['Latitude', 'Longitude']]
        self.pids = df[['pid']]
        
        # Set up error tracking for largest number of points queried
        self.max_query = 0
        
        
    def fit(self, base_dists, ip, limit=0.5):
        """Fits road distance matrix"""
        have_dists_adj = self._convert_subset_baseline_dists(base_dists, N_road=self.N)
        self.lat_long_df = self.lat_long_df.iloc[ : self.N]
        self.populate_road_dist_matrix(have_dists_adj, ip, limit)
        self.limit=limit
        
    def save(self, filepath):
        idx2pid = {idx: self.pids.iloc[i]['pid'] for i, idx in enumerate(self.lat_long_df.index)}
        np.savez(filepath, road_distances = self.road_dist_arr, idx2pid = idx2pid, limit=self.limit, allow_pickle=True)
        
        
    def populate_road_dist_matrix(self, base_dists, ip, limit=0.5):
        """Populates the road distance array"""
        for i in range(self.N):
            if i % 100 == 0:
                print(f"Finished Row {i}")
            idx = np.argwhere(base_dists[i] < limit).flatten()
            n_query = len(idx)
            if n_query > 0:
                if n_query > self.max_query:
                    self.max_query = n_query
                x = RoadDistanceMatrix.calculate_street_distance(i, idx, ip_add=ip, df=self.lat_long_df)
            else:
                x = []
            self.road_dist_arr[i][idx] = x
        
    def _create_road_dist_array(self, N_road=10):
        road_dist_arr = np.ones(N_road**2, dtype='float64').reshape(N_road, N_road) * 1e5
        return road_dist_arr

    def _convert_subset_baseline_dists(self, dists, N_road=10):
        """Converts haversine distance matrix to be in km and subset to same size as road array"""
        return dists[:N_road, :N_road] * 6371 # 6371 is radius of earth in km
    
    @staticmethod
    def create_long_lat_string(df, j):
        """Helper method to concatenate strings into OSM-accepted format"""
        long_lat_str =  str(df.iloc[j]['Longitude'].round(6)) + ',' + str(df.iloc[j]['Latitude'].round(6))
        return long_lat_str
    
    @staticmethod
    def calculate_street_distance(i, j_list, ip_add, df, how='walking'):
        """Returns street distance time in seconds"""

        # Find origin lat/long list
        i_long_lat = RoadDistanceMatrix.create_long_lat_string(df, i)

        longlat_concatstr = ''
        for i, c in enumerate(j_list):
            if i == 0:
                longlat_concatstr = RoadDistanceMatrix.create_long_lat_string(df, c)
            else:
                longlat_concatstr += ';' + RoadDistanceMatrix.create_long_lat_string(df, c)


        url = f"http://{ip_add}:5000/table/v1/{how}/{i_long_lat};{longlat_concatstr}?sources=0"
        r = requests.get(url)
        output = r.json()

        return np.array(output['durations'][0][1:])
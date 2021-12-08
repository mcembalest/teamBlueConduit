# Distance Matrix

- [Description of process](#Description)
- [Code documentation](#Documentation)

## Description

There are two primary distance matrices used in this project, [Haversine distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html) and street distances. Haversine distances represent the air distance between two points. This is conceptually similar to the Euclidean distance, but accounts for the fact that the Earth is spherical. While the distances within Flint are close enough such that the Euclidean and Haversine distances should be nearly equal, we elected to use the Haversine distance due to the fact that its implementation from `sklearn` is no more difficult nor less performant than the Euclidean distance calculation.[^1] As described, we find this through the `sklearn` implementation of this function.

Because the physical infrastructure of a city is more likely to be correlated with its lead status, we construct a second measure of distance between two points: the "street distance". For our purposes, we measure the street distance in terms of the number of seconds required to navigate between two points provided by OpenStreetMaps (OSM). For context, OSM provides both walking and driving directions and times (albeit not direct distances). We elect to use walking times rather than drive times primarily due to the existence of highways and varying speed limits, which could result in parcels on opposite sides of the city being considered "close" neighbors.

There are two components to this documentation:

1. Setting up OpenStreetMaps Routing Machine on an AWS instance. This requires substantial RAM and compute and was done on the cloud to keep local resources available.
   - [Official OSRM Backend documentation](https://github.com/Project-OSRM/osrm-backend)
     - The `osrm_setup` directory contains two shell scripts. These can be uploaded to an EC2 instance and run (pre-and-post-restart) to setup and run OSRM similar to the "Quick Start" guide using Docker provided.
     - One challenge is that an SSH connection must be opened to the instance and then out / back in. Thus we have divided the shell scripts into pre/post restart of the server.
   - Resources:
     - [OSM on AWS](https://registry.opendata.aws/osm/)
     - [Geofabrik data for Michigan](http://download.geofabrik.de/north-america/us/michigan.html): downloaded minimal size to cover area to save space constraints. 
     - [Helpful blog walking through resources, may need to adjust swap space](https://datawookie.dev/blog/2017/09/building-a-local-osrm-instance/)
   
2. Downloading street distances. To limit the number of computations required, we implemented the following algorithmic approach:

   - Find Haversine distances between each parcel
   - Set a threshold for neighbors (we choose 0.5 km) and only find distances which have neighbors less than this threshold. This dramatically reduces the computation, as under this case the matrix will only have $\sim 2%$ of entries filled.

It is very costly to compute the actual distance between each parcel since this algorithm scales in $O(n^2)$. For Flint, there are roughly 27,000 parcels with a `lead` determination, and thus this matrix will have roughly 675 million entries. Even strategies tested to e.g. fill the upper triangular component of the matrix will scale in $O(n^2)$, as this would require 1 entry for the 2-parcel case (assuming the diagonals, or self-distance, is zero), three for the 3-parcel case, and the 4-parcel case will require 6. By sparisfying the computation ahead of time, we can dramatically speed up the computation time requried.[^2]

## Documentation

###


[^1]: In mathematical terms, the Haversine distance is calculated by first determining the central angle between points and then using the radius of the Earth to convert to the distance travelled. 
[^2]: We also considered a scaling approach with regard to the size of the EC2 instance used. Some resources indicted that OSRM was somewhat parallelized by being written in C++ but we were not able to effectively compute multiple concurrent requests and were able to compute the matrix in 90 minutes with these limits.
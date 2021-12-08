# Notebook Index

This README file provides an index of notebooks, categorized by topic. The areas described below are *BlueConduit*, *Data Processing*, *Modeling*, *Discussion / Analysis*, and *Archive*. The BlueConduit section contains notebooks explicitly provided by BlueConduit. The second encompasses all work done to explore the data (EDA) as well as to parameterize the road distances, in particular. The Modeling section includes all notebooks for models considered beyond a toy stage. The Discussion / Analysis section reports the results of the narrative and exploration of modeling. Finally, the *Archive* contains helpful developmental notebooks to discuss / consider removing.

### Instructions / Notes

- The 'Run Location' below specifies a location of either 'Local' or 'Colab'. All notebooks which have a Run Location of 'Colab' were originally run on Google Colaboratory. These notebooks include particular instructions to connect with Google Drive, where the Harvard IACS team stored the data, rather than AWS S3 or GCS due to its relatively small size. These tidbits can be removed if running locally. Moreover, filepaths may need to be adjusted. When this is required, we have noted the following with a:

  ```python
  save("# FILEPATH TO SAVE") # `save` here may indicate a `pd.to_csv()` or some other data structure 
  load("# FILEPATH TO LOAD") # `load` here may indicate a `pd.read_csv()` or some other data structure 
  ```

  to allow for better functionality and reproducibility of these results.

### BlueConduit

| **Notebook Title**            | **Link**                                                | **Description**                                              | **Run Location** | **STATUS** |
| ----------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ---------------- | ---------- |
| Example_Model_BlueConduit     | [link](blueconduit/Example_Model_BlueConduit.ipynb)     | Initial document provided by BlueConduit to demonstrate use of XGBoost model and Flint data. For illustrative purposes. | Local            | Keep       |
| Spatial_Partitions_Demo_Flint | [link](blueconduit/Spatial_Partitions_Demo_Flint.ipynb) | BlueConduit-provided notebook describing use of spatial partitioning tools in `gizmo` with Flint Parcels data. | Local            | Keep       |

### Data Processing

| **Notebook Title**            | **Link**                                                    | **Description**                                              | **Run Location** | **STATUS**       |
| ----------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ---------------- | ---------------- |
| Shapefile EDA                 | [link](data_processing/Shapefile EDA.ipynb)                 | Initial exploratory data analysis if Flint Parcels shapefile; incomplete version currently remains on GitHub. | Local            | Discuss removing |
| Spatial_Partitions_Demo_Flint | [link](data_processing/Spatial_Partitions_Demo_Flint.ipynb) | BlueConduit-provided notebook describing use of spatial partitioning tools in `gizmo` with Flint Parcels data. | Local            | Discuss removing |
| DistanceMatrices              | [link](data_processing/DistanceMatrices.ipynb)              | Uses OpenStreetMaps Routing Machine (OSRM) spun up on an AWS instance to find the street distances, as well as Haversine distances, between all parcels in Flint, MI. | Colab            | Keep             |

### Modeling / Evaluation

| **Notebook Title**                     | **Link**                                                     | **Description**                                              | **Run Location** | **STATUS**       |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- | ---------------- |
| Flint_BC_Model_W_Plots                 | [link](modeling/Flint_BC_Model_W_Plots.ipynb)                | Contains initial usage of evaluation utilities for BlueConduit baseline models. | Local            | Discuss Removing |
| Compare_Evaluation_Methods_Naive_Model | [link](modeling/Compare_Evaluation_Methods_Naive_Model.ipynb) | (1) Performs demonstration of `select_data` function; (2) Compares performance of baseline with parcel-ordering and partition-ordering; (3) Compares to other naïve baselines. | Colab            | Keep             |
| Evaluation-by-partitions [JAVIERA]     |                                                              |                                                              |                  |                  |

### Discussion / Analysis

### Archive

| **Notebook Title**  | **Link**                                  | **Description**                                              | **STATUS** |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------ | ---------- |
| query_osrm_dys      | [link](archive/query_osrm_dys.ipynb)      | Initial OSRM querying done by DYS; overtaken by Find_Road_Distances | To Remove  |
| Find_Road_Distances | [link](archive/Find_Road_Distances.ipynb) | Local version for considering road distance formulation      |            |
|                     |                                           |                                                              |            |




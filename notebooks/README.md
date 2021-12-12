# Notebook Index

This README file provides an index of notebooks, categorized by topic. The areas described below are *BlueConduit*, *Data Processing*, *Modeling*, *Discussion / Analysis*, *Examples*, and *Archive*. The BlueConduit section contains notebooks explicitly provided by BlueConduit. The second encompasses all work done to explore the data (EDA) as well as to parameterize the road distances, in particular. The Modeling section includes all notebooks for models considered beyond a toy stage. The Discussion / Analysis section reports the results of the narrative and exploration of modeling. "Examples" contains a set of code demonstrations for using `blue_conduit_spatial`. Finally, the *Archive* contains helpful developmental notebooks to discuss / consider removing.

### Instructions / Notes

- The 'Run Location' below specifies a location of either 'Local' or 'Colab'. All notebooks which have a Run Location of 'Colab' were originally run on Google Colaboratory. These notebooks include particular instructions to connect with Google Drive, where the Harvard IACS team stored the data, rather than AWS S3 or GCS due to its relatively small size. These tidbits can be removed if running locally. Moreover, filepaths may need to be adjusted. When this is required, we have noted the following with a:

  ```python
  save("# FILEPATH TO SAVE") # `save` here may indicate a `pd.to_csv()` or some other data structure 
  load("# FILEPATH TO LOAD") # `load` here may indicate a `pd.read_csv()` or some other data structure 
  ```

  to allow for better functionality and reproducibility of these results.
  
- The files below are categorized into "Low", "Medium", and "High" priority. Due to the length of this project, we report our results across a wide set of models but hope to focus in on a few key areas:

  - *High*: Consists of results that are central to the findings and analysis highlighted in the technical report, presentation, and other project deliverables.
  - *Medium*: These notebooks consist of data processing steps as well as results which are not central findings (e.g. comparison of the BlueConduit baseline to other naïve models, demonstrations of highly-utilized code).
  - *Low*: Consists primarily of models not used for production but which occupied a significant share of our thought process during the course of the project.

### BlueConduit

| **Notebook Title**            | **Link**                                                | **Description**                                              | **Run Location** | Priority |
| ----------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ---------------- | -------- |
| Example_Model_BlueConduit     | [link](blueconduit/Example_Model_BlueConduit.ipynb)     | Initial document provided by BlueConduit to demonstrate use of XGBoost model and Flint data. For illustrative purposes. | Local            | Low      |
| Spatial_Partitions_Demo_Flint | [link](blueconduit/Spatial_Partitions_Demo_Flint.ipynb) | BlueConduit-provided notebook describing use of spatial partitioning tools in `gizmo` with Flint Parcels data. | Local            | Low      |

### Data Processing

| **Notebook Title** | **Link**                                       | **Description**                                              | **Run Location** | Priority |
| ------------------ | ---------------------------------------------- | ------------------------------------------------------------ | ---------------- | -------- |
| Flint_Data_EDA     | [link](data_processing/Flint_Data_EDA.ipynb)   | Initial exploratory analysis of Flint parcels data, examining features, missingness, etc. | Local            | Low      |
| DistanceMatrices   | [link](data_processing/DistanceMatrices.ipynb) | Uses OpenStreetMaps Routing Machine (OSRM) spun up on an AWS instance to find the street distances, as well as Haversine distances, between all parcels in Flint, MI. | Colab            | Medium   |

### Modeling / Evaluation

| **Notebook Title**                     | **Link**                                                     | **Description**                                              | **Run Location** | **Priority** |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- | ------------ |
| Flint_BC_Model_W_Plots                 | [link](modeling/Flint_BC_Model_W_Plots.ipynb)                | Contains initial usage of evaluation utilities for BlueConduit baseline models. | Local            | Low          |
| Compare_Evaluation_Methods_Naive_Model | [link](modeling/Compare_Evaluation_Methods_Naive_Model.ipynb) | (1) Performs demonstration of `select_data` function; (2) Compares performance of baseline with parcel-ordering and partition-ordering; (3) Compares to other naïve baselines. | Colab            | Medium       |
| DiffusionModel                         | [link](modeling/DiffusionModel.ipynb)                        | Sets up & runs basic diffusion model.                        | Colab            | High         |
| Hyperparameter_Tuning_Diffusion        | [link](modeling/Hyperparameter_Tuning_Diffusion.ipynb)       | Runs hyperparameter tuning grid for diffusion model and stores optimal results. Also presents results with highlighting for convenient viewing. | Colab            | Low          |
| Costs                                  | [link](modeling/Costs.ipynb)                                 | Summarizes the cost savings using diffusion, produces cost (average & cumulative) used in deliverables. | Local            | High         |
| ConvGNN                                | [link](modeling/ConvGNNModel.ipynb)                          | Experimentation & implementation of Convolutional GNN model. Used in initial modeling of Flint through GNNs, potentially useful to understand GNNs + LSL data. | Colab            | Low          |
| GPModel                                | [link](modeling/GPModel.ipynb)                               | Implements multiple Gaussian Process models and tests against baseline XGBoost. GPs include: latitude & longitude, latitude, longitude & year built (spatiotemporal GP), and methods to derive model variances from the above models. Potentially useful to understand GPs for LSL data. | Colab            | Low          |
| GraphSAGE_GAT_Models                   | [link](modeling/GraphSAGE_GAT_Models.ipynb)                  | Implements a [GraphSage](http://snap.stanford.edu/graphsage/) and a [Graph Attention Network (GAT)](https://arxiv.org/abs/1710.10903). Note that these methods are derived from the vanilla GNNs implemented above and surpassed the ConvGNN in performance. Potentially useful to understand the performance of state-of-the-art graphical models for LSL data. | Colab            | Low          |
| StackingModel                          | [link](modeling/StackingModel.ipynb)                         | Implements [Stacked Generalization](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) model with a logistic metamodel across the baseline model and alternative models proposed throughout the project. Did not outperform diffusion alone. | Colab            | Low          |

### Discussion / Analysis

| **Notebook Title**          | **Link**                                           | **Description**                                              | **Run Location** | **Priority** |
| --------------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ---------------- | ------------ |
| Explore_Diffusion_Results   | [link](analysis/Explore_Diffusion_Results.ipynb)   | Performs modified ablation study of diffusion effects and discusses regularization of the XGBoost model with L1 & L2 regularization | Colab            | High         |
| Diffusion_Impact_Assessment | [link](analysis/Diffusion_Impact_Assessment.ipynb) | Reports the result of the "Diffusion Impact Assessment", which studies two key questions: (1) how did diffusion affect the ordering of all homes & uncertain homes especially?; (2) are there disparate impacts of the diffusion model?. The second question is answered via the Census Tract covariates used by the XGBoost model. | Colab            | High         |

### Examples

| **Notebook Title**             | **Link**                                              | **Description**                                              | **Run Location** | Priority |
| ------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ---------------- | -------- |
| precision-recall-plot          | [link](examples/precision-recall-plot.ipynb)          | Plots the usage of the precision & recall plotting tools     | Local            | Low      |
| flint_data_modularization_test | [link](examples/flint_data_modularization_test.ipynb) | Demonstrates use of modularized tooling for flint data samples. | Local            | Low      |

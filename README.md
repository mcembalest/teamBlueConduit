#Spatial Modeling for Lead Service Line Detection

###  [Harvard IACS](https://iacs.seas.harvard.edu/) Capstone Project , Fall 2021

### Industry Partner: [BlueConduit](https://www.blueconduit.com/)

Harvard IACS Team: Javiera Astudillo, Max Cembalest, Kevin Hare, and Dashiell Young-Saver; advised by Isaac Slavitt and Chris Tanner.

## Executive Summary

In this project, we investigate the potential for incorporating spatial information into BlueConduit's baseline machine learning model to predict the location of lead service lines. To do so, we investigate a number of approaches, from naïve inclusion of spatial features to Gaussian Processes and Graph Neural Networks. Ultimately, we show that graph-based nearest-neighbors diffusion provides superior performance, improving the hit rate curve for most neighborhood resolutions and reducing the average cost of replacement of a lead pipe. Further, we investigate possible mechanisms for these results such as providing regularization to the baseline model and locating homes with potentially inaccurate but otherwise predictive features. Finally, we evaluate all of our models at various spatial partitioning resolutions, in terms of both cross-validation as well as aggregation for calculation of the hit rate. This methodology results in overall worse performance across all models but more closely aligns with the real-world scenario of municipalities selecting small areas to investigate rather than working parcel-by-parcel.

## Helpful Links

- [Installation](#Installation--Data-Setup)
- [Repository Organization](#Repository-organization)
- [Broader Impact Assessment](#Broader-Impact-Assessment) ([File](docs/broader_impact_assessment.md))

### Documents
- [Technical Report](reports/Technical%20Reports/2021.12.15%20Harvard-BlueConduit%20Technical%20Report.pdf)
- Summary Blog Post ([In GitHub](reports/Misc/Blog%20Post%20(AC297r%2C%20BlueConduit%20Final%20Project).pdf)), ([Medium](https://medium.com/@youngsaver/using-spatial-information-to-detect-lead-pipes-73a1e68d5643))
### API Reference
- [Utilities](docs/bcs/utilities.md#API-Reference)
- [Modeling](docs/bcs/modeling.md)
- [Evaluation](docs/bcs/evaluation.md#API-Reference)
- [Distance Matrix / Data Processing](docs/bcs/distance_matrix.md#API-Reference)

## Broader Impact Assessment
We hope our work has a direct impact on health outcomes in Flint and other cities. We have shown that our diffusion model can help cities reduce the amount of time and money required to get lead out of the ground. This will, ultimately, create safer infrastructure and lower monetary burdens for cities and their residents.

However, we also investigated possible negative consequences that could arise from our diffusion model. One possible area of negative impact is disparate impact with respect to demographics such as age, race, and income.

Since our goal is to optimize the cost of replacement city-wide, certain neighborhoods are at risk of having their lead replacement unfairly prioritized at the expense of others. We investigated how our model changed the digging priority order across demographic factors. Specifically, we used Flint census tract data to investigate correlations between demographic variables and the changes in digging order produced by our model. We arrived at the following results:

**1. the available racial demographics of the census tract, calculated as the following: $\frac{\text{B_total_pop_black}}{\text{B_total_pop_black} + \text{B_total_pop_white}} \%$.** Our model seems to have no correlation (Spearman correlation -.03, with low-confidence p-value of 0.84). Note that `b_total_pop_black` and `b_total_pop_white` are measured at the [Census Tract](https://www2.census.gov/geo/pdfs/education/CensusTracts.pdf)-level.

**2.  the median age.** Our model has a slight negative correlation (Spearman correlation -0.2, with medium-confidence p-value of 0.23)

**3. the average value of** **`Residential_Building_Value`.** Our model has negative correlation (Spearman correlation -0.4, with high-confidence p-value of 0.01).

Overall, we found little impact with respect to age of residents and density of Black residents in neighborhoods. However, we have evidence to suggest our model prioritizes areas with lower residential value (i.e. lower property value neighborhoods). In other words, our model tends to bump up lower value homes in the dig order.

Note: The data underlying our project represents a city (Flint) with historically embedded racial discrimination in the form of housing disparities and redlining. So, we are not attempting to claim that the use of machine learning in this setting will be globally unbiased with respect to all protected variables. Rather, we are claiming that our diffusion method does not seem to *amplify* any bias on the basis of race or age, compared to BlueConduit’s baseline model. This assessment is specific to just one dataset: the ~27,000 homes in Flint. To improve on this impact assessment in the future, we would investigate how diffusion impacts dig orders in more cities. Another possible area of negative impact for future examination is whether our smoothed probabilities post-diffusion disrupt BlueConduit's longer term goal of having a calibrated model (e.g. of the homes predicted with 70% probability of lead, 70% of them actually end up having lead).

## Repository organization

This repository is loosely structured according to the principles laid out by the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template produced by [DrivenData](https://www.drivendata.org/). This includes a source repository for all code used in development and evaluation of the model frameworks presented, as well as a series of analysis notebooks for exploration and model results. As with a standard packaged Python project, we have developed separate documentation for the use of our tooling. Below is a set of helpful links:

- [`blue_conduit_spatial`](/blue_conduit_spatial): a Python package implementing the baseline and diffusion methods described in this project, as well as data loading and evaluation tooling.
- Documentation: Documentation for the `blue_conduit_spatial` package.
  - [Installation instructions](docs/installation.md) (also produced below)
  - [`blue_conduit_spatial` documentation](docs/blue_conduit_spatial.md)
- [Data](/data): For access to the data used in this project, please contact BlueConduit. All analyses can be reproduced once data is ported to this directory.
- [Notebooks](/notebooks): Notebooks that present (a) exploratory data analyses; (b) model evaluation results; (c) alternative models considered; (d) case studies of diffusion.
  - [Link](notebooks/README.md) to notebook index, describing each analytical notebook.

## Installation & Data Setup
To install the `blue_conduit_spatial` package, please follow the instructions below. Note that `gizmo ` is a proprietary package developed by BlueConduit. To obtain access to the package, please contact BlueConduit. Please note that all installs below are done using `pip` install, but if your interpreter calls to `pip3`, the commands can be easily substituted.

1. Update Python. Note that the partner library requires Python >= 3.7. This can be done via:

```shell
python --version
```

If you are in a virtual env with conda, you can do it like this:

```shell
conda update python
```

2. Install partner libraries. This requires downloading / git cloning the `gizmo` package.

```shell
cd gizmo
pip install -e .
```

2. Install requirements for `blue_conduit_spatial`. Note: must navigate back to `teamBlueConduit` directory.

```shell
pip install -r requirements.txt
```

3. Install `blue_conduit_spatial`. Execute following command in the `teamBlueConduit` directory.

```shell
pip install .
```

### Data folder structure

To reduce space locally, we have utilized a consistent structure of the data folders. Follow this [link](/data/README.md) to read documentation of the data folder structure required to reproduce the results from this project. All files can be replicated locally, though the distance matrices are > 5GB and thus were handled via Google Colab. Please email a Harvard IACS team member for access to cached predictions in Google Drive.

# Spatial Modeling for Lead Service Line Detection

###  [Harvard IACS](https://iacs.seas.harvard.edu/) Capstone Project , Fall 2021

### Industry Partner: [BlueConduit](https://www.blueconduit.com/)

Harvard IACS Team: Javiera Astudillo, Max Cembalest, Kevin Hare, and Dashiell Young-Saver; advised by Isaac Slavitt and Chris Tanner.

## Executive Summary

	#TODO

## Repository organization
This repository is loosely structured according to the principles laid out by the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template produced by DrivenData. This includes a source repository for all code used in development and evaluation of the model frameworks presented, as well as a series of analysis notebooks for exploration and model results. As with a standard packaged Python project, we have developed separate documentation for the use of our tooling. Below is a set of helpful links:

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

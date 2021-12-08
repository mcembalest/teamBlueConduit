# `blue_conduit_spatial` package documentation

At its core, `blue_conduit_spatial` is a set of tools that do four things:

1. Build a set of utility tools for repeatedly accessing the correct indexes / data for consistent and comparable models; especially important when evaluating model performance over time with visuals such as the hit rate curve, which require more than a single point estimate.
2. Data processing, in particular for the "road" / "street" distances between parcels in Flint.
3. Modeling, focused on the implementation of a custom `ServiceLineDiffusion` class which approximates the SKLearn functionality for fitting models and works in a model agnostic way; also implements various Gaussian-process and GNN-based models in a principled manner for reuse by BlueConduit.
4. Allow for customized and consistent evaluation.

These are organized into five modules, each with separate documentation:

1. [`blue_conduit_spatial.utilities`](bcs/utilities.md): includes tools for building and accessing data.
2. [`blue_conduit_spatial.distance_matrix`](bcs/distance_matrix.md): includes tools for setting up and building OSRM in AWS as well as objects for downloading and storing the street distance data.
3. [`blue_conduit_spatial.modeling`](bcs/modeling.md): includes tools for diffusion and GNN modeling.
4. [`blue_conduit_spatial.evaluation`](bcs/evaluation.md): includes tools for model evaluation such as the hit rate curve.
# Modeling

- [Description of running / execution](#Blue-Conduit-Baseline)
- [API Reference](#API-Reference)

## Blue Conduit Baseline

To fit the Blue Conduit baseline XGBoost models, we can run the following command. This requires the directory structure above, in particular having run or downloaded the `predictions` directory.

- Navigate to `blue_conduit_spatial/modeling`
- Execute `python blue_conduit_baseline.py`

Taken together, these commands will generate the `pred_probs_train.npz` and `pred_probs_test.npz` files. These correspond exactly to the indices described in `train_index.npz` and `test_index.npz`.

## API-Reference

### [Diffusion](../../blue_conduit_spatial/modeling/diffusion.py)

The `ServiceLineDiffusion` class provides an API similar to the `sk-learn` modeling API that fits diffusion for a user-specified number of iterations to a graph. This is the class structure used to reproduce the diffusion results which are the core of this project.

#### class `ServiceLineDiffusion`

Base class of diffusion process. Maintains all necessary components (e.g. which points are in train and which are in test, current predictions, etc.). Must be instantiated with graph and test set predictions to use. For more information on use of class, please see [`DiffusionModel`](../../notebooks/modeling/DiffusionModel.ipynb) notebook.

| **Arguments**   | **Type**       | **Status**     | Description                                                  |
| :-------------- | -------------- | -------------- | ------------------------------------------------------------ |
| `graph`         | `np.array`     | Required       | $(N, N)$ array holding the distances between each parcel.    |
| `train_indices` | `np.array`     | Required       | Sequence of indices in graph corresponding to the training set. |
| `test_indices`  | `np.array`     | Required       | Sequence of indices in graph corresponding to the test set.  |
| `Ytrain`        | `np.array`     | Required       | True lead values for the train set.                          |
| `Ytest`         | `np.array`     | Required       | True lead values for the test set.                           |
| `Ytrain_pred`   | `np.array`     | Required       | Baseline predictions for train set; if using e.g. a non-model, could be set to `0.5 * np.ones(Ytrain.shape[0])` or some other prior, e.g. `np.random.normal(0.5, scale=0.2, size=Ytrain.shape[0])`. |
| `Ytest_pred`    | `np.array`     | Required       | Similar to `Ytrain_pred` but corresponding to test set.      |
| `lam`           | `float`        | Optional; 0.5. | Lambda ($\lambda$) parameter governs self-weight in update step; higher $\lambda$ corresponds to more weight on self. |
| `lat_long_df`   | `pd.DataFrame` | Optional; None | If plotting, this will be used to localize points on the graph. Not necessary if not intending to plot. |

##### Methods

- `fit(n_iter=1, neighbor_fn=None, neighbor_params=None, distance_function=None, verbose=False)`:

  Fits the diffusion model by running `n_iter` update steps on graph.

  | **Arguments**       | **Type** | **Status**                                | Description                                                  |
  | :------------------ | -------- | ----------------------------------------- | ------------------------------------------------------------ |
  | `n_iter`            | `int`    | Optional; 1                               | Number of iterations (or update steps) to perform across graph. |
  | `neighbor_fn`       | `func`   | Optional; `graph_Kneighbors`              | A function to determine the nearest neighbors. If None, will use internal `graph_Kneighbors`, which wraps `KNearestNeighbors` from `sklearn`. Could update to be custom-wrapped version of Radial Nearest Neighbors or some other kernel function to determine neighbors. |
  | `neighbor_params`   | `dict`   | Optional; `{'graph': self.graph, 'K': 5}` | Dictionary of parameters to pass to the neighbor function. If none, will pass `self.graph` and K = 5. Note: *must* pass `self.graph` or some other method to externally determine neighbors. |
  | `distance_function` | `func`   | Optional; `diffusion_distance_weights`    | Function used to determine distance for weighted average update steps. If None, will use $1 / distance$ provided in the graph. |
  | `verbose`           | `bool`   | Optional; False                           | If True, will report the log-loss after each step; useful for debugging results. |

- `predict_proba(X, mode=None)`:

  Returns probabilities after lead values.
  
  Because this is a non-standard "model" and does not truly take a new X value, it uses the shape of X to determine whether to return the training or test predictions.
  
  | **Arguments** | **Type**     | **Status** | Description                                                  |
  | :------------ | ------------ | ---------- | ------------------------------------------------------------ |
  | `X`           | `array`-like | Required   | Array of training or test points, used only for its shape.   |
  | `mode`        | `str`        | Required   | One of None, 'train', 'test'. If none, will make decision based on X shape. |

- `diffusion_step(lead_vals)`

  Takes a single diffusion step across the whole graph

  | **Arguments** | **Type**   | **Status** | Description                                              |
  | :------------ | ---------- | ---------- | -------------------------------------------------------- |
  | `lead_vals`   | `np.array` | Required   | Array of current lead values to take update step across. |

- `_update_node(node_val, node_idx, weighted_avg_neighbor_lead, lam=0.5)`

  Defines update step for a single node. Can be overwritten, must return between [0,1].
  
  Note that overriding this method will produce the desired results for a kernel other than the weighted average by distance methodology.
  
  | **Arguments**                | **Type**        | **Status**    | Description                                                  |
  | :--------------------------- | --------------- | ------------- | ------------------------------------------------------------ |
  | `node_val`                   | `float`         | Required      | Current predicted probability for a given node.              |
  | `node_idx`                   | `np.array[int]` | Required      | Index of integers corresponding to the graph indexes of a particular parcel's neighbors. |
  | `weighted_avg_neighbor_lead` | `np.array`      | Required      | Array containing all current weighted average lead calculations. |
  | `lam`                        | `float`         | Optional; 0.5 | Share of weight to place on self vs. weighted average neighbors. |

- `graph_Kneighbors(graph, K)`

  Gets the nearest neighbors for each node in the graph.

  Note: This replaces 0 with 1e7 because `sklearn` `NearestNeighbors` treats 0 as a valid (close) distance. Since this is a non-connection in our graph, then want it to be a very small weight.

  | **Arguments** | **Type**   | **Status** | Description                                              |
  | :------------ | ---------- | ---------- | -------------------------------------------------------- |
  | `graph`       | `np.array` | Required   | (N, N) array of all current nodes / values are distances |
  | `K`           | `int`      | Required   | Number of neighbors                                      |

- `diffusion_distance_weight(distances)`

  Returns an arbitrary float distance as the inverse for weighting. Also adjusts 0 to be 1 such that a distance of zero maps to a weight of 1.

  

- `sqrt_distances(distances)`

  Calculates distance weights as the square root of the distances.

  Will tend to place more equal weight on further away neighbors since e.g. $\frac{1}{2 + 1} = 0.33$ and $\frac{1}{3 + 1} = 0.25$ are more spaced than $\frac{1}{\sqrt{2} + 1} = 0.41$ and $\frac{1}{\sqrt{3} + 1} = 0.36$.


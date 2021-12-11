from gnn import GNN
import numpy as np


adj = np.array([[1,1,1,0,0,1,0,1],
[1,1,1,1,0,0,1,0],
[1,1,1,0,1,1,0,0],
[0,1,0,1,0,0,0,1],
[0,0,1,0,1,0,1,1],
[1,0,1,0,0,1,1,0],
[0,1,0,0,1,1,1,1],
[1,0,0,1,1,0,1,1]])

features = np.array([
[0,0,0,1,0],
[0,1,0,1,0],
[0,0,0,1,1],
[0,1,1,0,1],
[1,1,1,0,1],
[1,0,1,0,1],
[0,1,1,0,0],
[1,1,1,1,0]])

labels = np.array([[0,1],
[0,1],
[0,1],
[1,0],
[1,0],
[0,1],
[1,0],
[0,1]])

train_mask = np.array([1,1,0,0,0,0,0,0])
val_mask = np.array([0,0,1,1,1,1,0,0])
test_mask = np.array([0,0,0,0,0,0,1,1])

my_gnn = GNN(adj, features, labels, train_mask, val_mask, test_mask, norm_adj=True)
my_gnn.train(100, 1e-2)
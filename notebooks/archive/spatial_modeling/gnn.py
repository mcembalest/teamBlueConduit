import numpy as np
import tensorflow as tf


# TODO:
# CAN THE ADJACENCY MATRIX ALSO TAKE WEIGHTED

def gnn_layer(fts, adj, transform, activation):
	seq_fts = transform(fts)
	ret_fts = tf.matmul(np.asmatrix(adj), np.asmatrix(seq_fts))
	return activation(ret_fts)

def masked_softmax_cross_entropy_loss(logits, labels, mask):
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	mask = tf.cast(mask, dtype=tf.float32)
	mask /= tf.reduce_mean(mask)
	loss *= mask
	return tf.reduce_mean(loss)

def masked_accuracy(logits, labels, mask):
	correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
	accuracy_all = tf.cast(correct_prediction, tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	mask /= tf.reduce_mean(mask)
	accuracy_all *= mask
	return tf.reduce_mean(accuracy_all)

class GNN:

	def __init__(self, adj, features, labels, train_mask, val_mask, test_mask, norm_adj = False):

		self.adj = adj.astype('float32') + np.eye(adj.shape[0]).astype('float32')
		self.features = features.astype('float32')
		self.labels = labels
		self.train_mask = train_mask
		self.val_mask = val_mask
		self.test_mask = test_mask
		if norm_adj:
			deg = tf.reduce_sum(self.adj, axis=-1)
			norm_deg = tf.linalg.diag(1.0 / tf.sqrt(deg))
			self.adj = np.asarray(tf.matmul(norm_deg, tf.matmul(np.asmatrix(self.adj), norm_deg)))

	def train(self, epochs, lr):

		optimizer = tf.keras.optimizers.Adam(lr)
		best_accuracy = 0.0
		for ep in range(epochs + 1):
			fts = tf.convert_to_tensor(self.features)
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(fts)
				lyr_1 = tf.keras.layers.Dense(50)
				lyr_2 = tf.keras.layers.Dense(self.labels.shape[1])

				hidden = gnn_layer(fts, self.adj, lyr_1, tf.nn.relu)
				logits = gnn_layer(hidden, self.adj, lyr_2, tf.identity)

			loss = masked_softmax_cross_entropy_loss(logits, self.labels, self.train_mask)
			variables = tape.watched_variables()
			grads = tape.gradient(loss, variables)
			optimizer.apply_gradients(zip(grads, variables))

			val_accuracy = masked_accuracy(logits, self.labels, self.val_mask)
			test_accuracy = masked_accuracy(logits, self.labels, self.test_mask)

			if val_accuracy > best_accuracy:
				best_accuracy = val_accuracy
				print('Epoch', ep, '| Train loss', loss.numpy(), '| Val acc', val_accuracy.numpy(), '| Test acc', test_accuracy.numpy())


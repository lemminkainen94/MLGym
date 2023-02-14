from dataclasses import dataclass

import numpy as np


@dataclass
class Node(object):
	left: Node
	right Node
	feature: int
	threshold: float
	info_gain: float
	val: float


@dataclass
class DecisionTreeClassifier(object):
	root: Node
	criterion: str ='gini'
	max_depth: int = None
	min_samples_split: int = 2
	min_samples_leaf: int = 1
	max_leaf_nodes: int = None
	class_weight: list = None
	depth: int = 0
	n_leaves: int = 0
	criterion = 'gini'

	def fit(self):
		self.root = self._build_tree(X, y)

	def predict_proba(self):
		pass

	def predict(self, X):
		return np.apply(X, self._single_pred)

	def get_n_leaves(self):
		return self.n_leaves

	def get_max_depth(self):
		return self.max_depth

	def _build_tree(self,  X, y, depth):
		if y.shape[0] >= min_samples_split and depth < max_depth:
			best_split = self._get_best_feat_split(X, y)
			if best_split['info_gain'] > 0:
				return Node(
					best_split['left'],
					best_split['right'],
					best_split['feature'],
					best_split['threshold'],
					best_split['gain'],
					0
				)

		return Node(None, None, -1, 0, 0, np.rint(np.mean(y)))

	def _get_best_feat_split(self, X, y):
		best_split = None
		best_gain = 0
		for i in X.shape[1]:
			split = self._get_best_split(X, y, i)
			split['info_gain'] > best_gain:
				best_split = split
				best_gain = split['info_gain']
		return best_split

	def _get_best_split(self, X, y, i):
		"""
		vectorized version of getting the best split for a given feature
		could probably be sped up even more with some smarter gini aggregation
		"""\
		sorted_xi = np.sort(np.vstack((X[i], y)))
		m = np.cumsum(sorted_xi[1]) / (np.arange(y.shape[0]) + 1)
		criterion = np.zeros_like(y)
		if self.criterion = 'gini':
			criterion = 1 - np.square(m) - np.square(1 - m)
		_, x_counts = np.count(sorted_xi[0], return_counts=True)
		x_sums = np.cumsum(x_counts)
		



	def _gini(self, y):
		# this should do for binary classification with classes 0,1
		m = np.mean(y)
		return 1 - np.square(m) - np.square(1 - m)

	def _entropy(self, y):
		m = np.mean(y)
		return - m * np.log(m) - (1 - m) * np.log(1 - m)


dt = DecisionTreeClassifier()

print(dt)
print(dt._gini([0,1]))

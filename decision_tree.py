from dataclasses import dataclass

import numpy as np


@dataclass
class DecisionTreeClassifier(object):
	criterion: str ='gini'
	max_depth: int = None
	min_samples_split: int = 2
	min_samples_leaf: int = 1
	max_leaf_nodes: int = None
	class_weight: list = None
	depth: int = 0
	n_leaves: int = 0

	def fit(self):
		pass

	def predict_proba(self):
		pass

	def predict(self):
		pass

	def get_n_leaves(self):
		return self.n_leaves

	def get_depth(self):
		return self.depth

	def _gini(self, y):
		# this should do for binary classification with classes 0,1
		m = np.mean(y)
		return 1 - np.square(m) - np.square(1 - m)

	def _entropy(self, X):
		pass


dt = DecisionTreeClassifier()

print(dt)
print(dt._gini([0,1]))

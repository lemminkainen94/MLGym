import numpy as np


EPS_CONST = 1e-9


class AdaBoost(object):
	def __init__(self, M):
		super().__init__()
		self.num_models = M
		self.base_models = []
		self.eps = np.zeros(M)
		self.alpha = np.zeros(M)

	def fit(self, X, y):
		n = X.shape[0]
		W = np.ones(n) / n

		for m in range(self.num_models):
			h_m, I_m = self._fit_base_model(W, X, y)
			self.base_models.append(h_m)

			self.eps[m] = np.sum(np.multiply(W, I_m)) / np.sum(W)
			print(self.eps[m])
			self.alpha[m] = np.log((1 - self.eps[m] + EPS_CONST) / self.eps[m])

			W = np.multiply(W, np.exp(self.alpha[m] * I_m))

	def _fit_base_model(self, clf, W, X, y):
		"""
		the base model should implement weighted loss function, 
		like sklearn classifiers with sample_weight param
		clf - classifier instance e.g. sklearn.tree.DecisionTreeClassifier
		"""
		h_m = clf.fit(X, y, sample_weight=W)
		preds = clf.predict(X)
		I_m = np.multiply(W, np.where(y == preds, 0, 1)) # preds mustn't be a soft vector
		return h_m, I_m


ada = AdaBoost(2)
ada.fit(np.ones(2), np.ones(2))
print(ada.alpha, ada.base_models)

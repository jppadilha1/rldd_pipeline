"""
Rede de Função de Base Radial (RBF) com treinamento em dois estágios,
compatível com scikit-learn (funciona em Pipeline e GridSearchCV).

Estágio 1 (não supervisionado): centros pelo k-means; larguras pela
  variância de cada agrupamento ("cluster") ou pela regra de Haykin
  sigma = d_max / sqrt(2k) ("haykin").
Estágio 2 (supervisionado): camada de saída linear ajustada por mínimos
  quadrados com regularização (equivalente à solução de convergência da
  regra delta generalizada / LMS, sem depender de taxa de aprendizado).
Ativação gaussiana: phi_j(x) = exp(-||x - c_j||^2 / (2 sigma_j^2)).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


class RedeRBF(ClassifierMixin, BaseEstimator):
    def __init__(self, n_centros=8, largura="cluster", ridge=1e-2, random_state=None):
        self.n_centros = n_centros
        self.largura = largura
        self.ridge = ridge
        self.random_state = random_state

    def _phi(self, X):
        d2 = ((X[:, None, :] - self.centros_[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * self.sigmas_[None, :] ** 2))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        k = int(min(self.n_centros, len(np.unique(X, axis=0))))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
        C = km.cluster_centers_
        if self.largura == "haykin":
            dc = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
            dmax = dc.max() if k > 1 else np.sqrt(X.var(0).sum())
            sig = np.full(k, dmax / np.sqrt(2 * k) if dmax > 0 else 1.0)
        else:
            sig = np.empty(k)
            for j in range(k):
                pts = X[km.labels_ == j]
                sig[j] = np.sqrt(((pts - C[j]) ** 2).sum(1).mean()) if len(pts) > 1 else 0.0
            pos = sig[sig > 1e-6]
            sig[sig <= 1e-6] = np.median(pos) if pos.size else 1.0
        self.centros_, self.sigmas_ = C, sig

        P = np.hstack([self._phi(X), np.ones((len(X), 1))])
        T = (y[:, None] == self.classes_[None, :]).astype(float)
        reg = self.ridge * np.eye(P.shape[1])
        reg[-1, -1] = 0.0  # não regulariza o bias
        self.W_ = np.linalg.solve(P.T @ P + reg, P.T @ T)
        return self

    def _saida(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([self._phi(X), np.ones((len(X), 1))]) @ self.W_

    def decision_function(self, X):
        o = self._saida(X)
        return o[:, 1] - o[:, 0] if len(self.classes_) == 2 else o

    def predict(self, X):
        return self.classes_[np.argmax(self._saida(X), axis=1)]

    def n_parametros(self):
        k, d = self.centros_.shape
        return k * d + k + self.W_.size

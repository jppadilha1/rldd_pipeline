"""
Redes neurais clássicas implementadas em NumPy, no estilo da disciplina
(Silva, Spatti e Flauzino, 2010; Haykin, 2009), compatíveis com scikit-learn
(funcionam em Pipeline e GridSearchCV).

Convenções comuns:
- Limiar como peso de uma entrada fixa x0 = -1.
- Pesos iniciais aleatórios uniformes em [0, 1] (semente em random_state).
- EQM = (1/p) * soma_k (1/2) * soma_j (d_j(k) - y_j(k))^2.
- Parada quando |EQM_atual - EQM_anterior| <= epsilon ou ao atingir max_epocas.
- Treinamento em lote (batch): os pesos são ajustados uma vez por época com o
  gradiente médio de todos os padrões. O Perceptron usa a regra de Hebb
  padrão a padrão (online), como no algoritmo original.

Depois do fit, cada modelo guarda:
  epocas_         número de épocas executadas
  eqm_final_      EQM no fim do treino (Perceptron: fração de erros)
  curva_          EQM (ou fração de erros) a cada época, começando na época 0
  motivo_parada_  "epsilon", "sem_erros", "max_epocas" ou "divergiu"
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


def _com_bias(X):
    return np.hstack([-np.ones((len(X), 1)), X])


def _alvos(y, classes, baixo, alto):
    if len(classes) == 2:
        return np.where(y == classes[1], alto, baixo).astype(float)[:, None]
    return np.where(y[:, None] == classes[None, :], alto, baixo).astype(float)


def _eqm(D, Y):
    return float(0.5 * np.mean(np.sum((D - Y) ** 2, axis=1)))


def _sigmoide(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))


def regra_delta_lote(P, D, eta, epsilon, max_epocas, W0):
    """Regra delta (LMS) em lote sobre P (já com a coluna de bias)."""
    W = W0.copy()
    p = len(P)
    eqm_ant = _eqm(D, P @ W)
    curva, motivo, ep = [eqm_ant], "max_epocas", 0
    for ep in range(1, max_epocas + 1):
        W_ant = W.copy()
        W += eta * P.T @ (D - P @ W) / p
        eqm = _eqm(D, P @ W)
        if not np.isfinite(eqm) or eqm > 1e6:
            W, motivo = W_ant, "divergiu"
            break
        curva.append(eqm)
        if abs(eqm - eqm_ant) <= epsilon:
            motivo = "epsilon"
            break
        eqm_ant = eqm
    return W, ep, curva, motivo


class _Base(ClassifierMixin, BaseEstimator):
    limiar_ = 0.0

    def _saida(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        o = self._saida(np.asarray(X, dtype=float))
        return o[:, 0] if o.shape[1] == 1 else o

    def predict(self, X):
        o = self._saida(np.asarray(X, dtype=float))
        if o.shape[1] == 1:
            return self.classes_[(o[:, 0] >= self.limiar_).astype(int)]
        return self.classes_[np.argmax(o, axis=1)]


class PerceptronClassico(_Base):
    """Perceptron de Rosenblatt (regra de Hebb), padrão a padrão.
    Várias classes: um Perceptron por classe (um contra todos)."""

    def __init__(self, eta=0.01, max_epocas=1000, random_state=None):
        self.eta = eta
        self.max_epocas = max_epocas
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        P, D = _com_bias(X), _alvos(y, self.classes_, -1.0, 1.0)
        W = rng.uniform(0, 1, (P.shape[1], D.shape[1]))
        sinal = lambda u: np.where(u >= 0, 1.0, -1.0)
        curva = [float(np.mean(np.any(sinal(P @ W) != D, axis=1)))]
        motivo, ep = "max_epocas", 0
        for ep in range(1, self.max_epocas + 1):
            for k in rng.permutation(len(P)):
                erro = D[k] - sinal(P[k] @ W)
                if np.any(erro != 0):
                    W += self.eta * np.outer(P[k], erro)
            taxa = float(np.mean(np.any(sinal(P @ W) != D, axis=1)))
            curva.append(taxa)
            if taxa == 0:
                motivo = "sem_erros"
                break
        self.W_, self.epocas_, self.curva_ = W, ep, curva
        self.eqm_final_, self.motivo_parada_ = curva[-1], motivo
        return self

    def _saida(self, X):
        return _com_bias(X) @ self.W_

    def n_parametros(self):
        return int(self.W_.size)


class AdalineClassico(_Base):
    """Adaline treinado pela regra delta (LMS) em lote."""

    def __init__(self, eta=0.01, epsilon=1e-6, max_epocas=10000, random_state=None):
        self.eta = eta
        self.epsilon = epsilon
        self.max_epocas = max_epocas
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        P, D = _com_bias(X), _alvos(y, self.classes_, -1.0, 1.0)
        W0 = rng.uniform(0, 1, (P.shape[1], D.shape[1]))
        self.W_, self.epocas_, self.curva_, self.motivo_parada_ = regra_delta_lote(
            P, D, self.eta, self.epsilon, self.max_epocas, W0)
        self.eqm_final_ = self.curva_[-1]
        return self

    def _saida(self, X):
        return _com_bias(X) @ self.W_

    def n_parametros(self):
        return int(self.W_.size)


class PMCClassico(_Base):
    """Perceptron Multicamadas com uma camada escondida, função logística em
    todos os neurônios e backpropagation (regra delta generalizada) em lote."""

    def __init__(self, n_ocultos=8, eta=0.5, epsilon=1e-6, max_epocas=10000, random_state=None):
        self.n_ocultos = n_ocultos
        self.eta = eta
        self.epsilon = epsilon
        self.max_epocas = max_epocas
        self.random_state = random_state

    def _propagar(self, X1, W1, W2):
        Y1 = _sigmoide(X1 @ W1.T)
        X2 = _com_bias(Y1)
        return Y1, X2, _sigmoide(X2 @ W2.T)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.limiar_ = 0.5
        rng = np.random.default_rng(self.random_state)
        X1, D = _com_bias(X), _alvos(y, self.classes_, 0.0, 1.0)
        p, h, o = len(X1), int(self.n_ocultos), D.shape[1]
        W1 = rng.uniform(0, 1, (h, X1.shape[1]))
        W2 = rng.uniform(0, 1, (o, h + 1))
        Y1, X2, Y2 = self._propagar(X1, W1, W2)
        eqm_ant = _eqm(D, Y2)
        curva, motivo, ep = [eqm_ant], "max_epocas", 0
        for ep in range(1, self.max_epocas + 1):
            d2 = (D - Y2) * Y2 * (1 - Y2)
            d1 = (d2 @ W2[:, 1:]) * Y1 * (1 - Y1)
            W2 = W2 + self.eta * d2.T @ X2 / p
            W1 = W1 + self.eta * d1.T @ X1 / p
            Y1, X2, Y2 = self._propagar(X1, W1, W2)
            eqm = _eqm(D, Y2)
            curva.append(eqm)
            if abs(eqm - eqm_ant) <= self.epsilon:
                motivo = "epsilon"
                break
            eqm_ant = eqm
        self.W1_, self.W2_ = W1, W2
        self.epocas_, self.curva_, self.motivo_parada_ = ep, curva, motivo
        self.eqm_final_ = curva[-1]
        return self

    def _saida(self, X):
        return self._propagar(_com_bias(X), self.W1_, self.W2_)[2]

    def n_parametros(self):
        return int(self.W1_.size + self.W2_.size)


class RBFClassica(_Base):
    """Rede RBF em dois estágios: centros pelo k-means e larguras pela
    variância de cada agrupamento ("cluster") ou por sigma = dmax/sqrt(2k)
    ("haykin"); camada de saída linear treinada pela regra delta em lote."""

    def __init__(self, n_centros=4, largura="cluster", eta=0.1, epsilon=1e-6,
                 max_epocas=10000, random_state=None):
        self.n_centros = n_centros
        self.largura = largura
        self.eta = eta
        self.epsilon = epsilon
        self.max_epocas = max_epocas
        self.random_state = random_state

    def _phi(self, X):
        d2 = ((X[:, None, :] - self.centros_[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * self.sigmas_[None, :] ** 2))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
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
        self.centros_, self.sigmas_, self.iter_kmeans_ = C, sig, int(km.n_iter_)
        rng = np.random.default_rng(self.random_state)
        P, D = _com_bias(self._phi(X)), _alvos(y, self.classes_, -1.0, 1.0)
        W0 = rng.uniform(0, 1, (P.shape[1], D.shape[1]))
        self.W_, self.epocas_, self.curva_, self.motivo_parada_ = regra_delta_lote(
            P, D, self.eta, self.epsilon, self.max_epocas, W0)
        self.eqm_final_ = self.curva_[-1]
        return self

    def _saida(self, X):
        return _com_bias(self._phi(X)) @ self.W_

    def n_parametros(self):
        k, d = self.centros_.shape
        return int(k * d + k + self.W_.size)

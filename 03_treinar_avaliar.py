"""
ETAPA 3 - Treinamento e avaliação: limiar fixo de EAR, Perceptron, Adaline,
PMC e RBF, com validação cruzada separada por participante.

Protocolo
- Validação externa: os 5 folds oficiais da UTA-RLDD (participantes de teste
  nunca aparecem no treino).
- Seleção de hiperparâmetros: GridSearchCV dentro do treino de cada fold, com
  GroupKFold por participante (3 partições), métrica = acurácia balanceada.
- Imputação (mediana) e padronização (z-score) ajustadas só no treino.
- Modelos estocásticos repetidos com N sementes; métricas = média das sementes
  em cada fold e depois média +- desvio entre folds.
- Decisão por vídeo: voto majoritário das janelas (comparável à "video
  accuracy" de Ghoddoosian et al., 2019).
- Teste de Wilcoxon pareado por participante (acurácia balanceada).
- Importância por permutação (PMC e RBF) nos folds de teste.

Tarefas: "binaria" (alerta x sonolento) e "tres" (alerta x baixa vigilância x
sonolento).

Exemplo:
  python 03_treinar_avaliar.py --atributos saida/atributos.csv --saida saida/resultados
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comum import ATRIBUTOS, PARAM
from rbf import RedeRBF

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def pipe(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()),
                     ("clf", clf)])


MODELOS = {
    "perceptron": (pipe(Perceptron(max_iter=1000, tol=1e-3)),
                   {"clf__eta0": [0.01, 0.1, 1.0]}),
    "adaline": (pipe(SGDClassifier(loss="squared_error", penalty=None, learning_rate="constant",
                                   max_iter=1000, tol=1e-4)),
                {"clf__eta0": [1e-4, 1e-3, 1e-2]}),
    "pmc": (pipe(MLPClassifier(activation="tanh", solver="lbfgs", max_iter=3000)),
            {"clf__hidden_layer_sizes": [(2,), (4,), (8,), (16,), (32,)],
             "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}),
    "rbf": (pipe(RedeRBF()),
            {"clf__n_centros": [2, 4, 8, 16, 32],
             "clf__largura": ["cluster", "haykin"],
             "clf__ridge": [1e-3, 1e-1, 10.0]}),
}


def carregar(caminho, tarefa):
    df = pd.read_csv(caminho, dtype={"participante": str})
    if tarefa == "binaria":
        df = df[df["rotulo"].isin([0, 10])].copy()
        df["y"] = (df["rotulo"] == 10).astype(int)
    else:
        df["y"] = df["rotulo"].map({0: 0, 5: 1, 10: 2})
    return df.reset_index(drop=True)


def escore(modelo, X, n_classes):
    clf = modelo
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)
            return p[:, 1] if n_classes == 2 else p
        except AttributeError:
            pass
    return clf.decision_function(X)


def metricas(y, yp, s, binaria):
    m = dict(acuracia=accuracy_score(y, yp), acuracia_bal=balanced_accuracy_score(y, yp))
    if binaria:
        m["f1_sonolento"] = f1_score(y, yp, pos_label=1, zero_division=0)
        m["sensibilidade"] = recall_score(y, yp, pos_label=1, zero_division=0)
        m["especificidade"] = recall_score(y, yp, pos_label=0, zero_division=0)
        try:
            m["auc"] = roc_auc_score(y, s) if s is not None and len(np.unique(y)) == 2 else np.nan
        except ValueError:
            m["auc"] = np.nan
    else:
        m["f1_macro"] = f1_score(y, yp, average="macro", zero_division=0)
    return m


def voto_video(pred):
    g = pred.groupby(["participante", "rotulo"])
    vid = g.agg(y=("y", "first"), yp=("yp", lambda v: np.bincount(v.astype(int)).argmax()))
    return accuracy_score(vid["y"], vid["yp"])


def rodar_tarefa(df, tarefa, sementes, saida):
    binaria = tarefa == "binaria"
    n_classes = df["y"].nunique()
    X, y, grupos = df[ATRIBUTOS], df["y"].to_numpy(), df["participante"].to_numpy()
    folds = sorted(df["fold"].unique())
    pred_linhas, hip_linhas, tempo_linhas, imp_linhas = [], [], [], []

    for fold in folds:
        te = df["fold"].to_numpy() == fold
        tr = ~te
        n_part_tr = len(np.unique(grupos[tr]))
        cv_int = GroupKFold(n_splits=min(3, n_part_tr))
        base = df.loc[te, ["fold", "participante", "rotulo", "janela", "y"]]

        # Baseline: limiar fixo de EAR (sem treinamento)
        if binaria:
            s = df.loc[te, "fech_max_fixo_s"].to_numpy()
            yp = (s >= PARAM["fixo_dur_s"]).astype(int)
            for sem in sementes:
                pred_linhas.append(base.assign(modelo="limiar_fixo_ear", semente=sem, yp=yp, escore=s))

        for nome, (est, grade) in MODELOS.items():
            gs = GridSearchCV(clone(est).set_params(clf__random_state=0), grade, cv=cv_int,
                              scoring="balanced_accuracy", n_jobs=-1, refit=False)
            gs.fit(X[tr], y[tr], groups=grupos[tr])
            melhor = gs.best_params_
            hip_linhas.append(dict(fold=fold, modelo=nome, cv_interna_acc_bal=gs.best_score_,
                                   **{k.replace("clf__", ""): str(v) for k, v in melhor.items()}))
            for sem in sementes:
                m = clone(est).set_params(**melhor, clf__random_state=sem)
                t0 = time.perf_counter()
                m.fit(X[tr], y[tr])
                t_fit = time.perf_counter() - t0
                yp = m.predict(X[te])
                sc = escore(m, X[te], n_classes)
                tempo_linhas.append(dict(fold=fold, modelo=nome, semente=sem, treino_s=t_fit,
                                         n_treino=int(tr.sum())))
                linha = base.assign(modelo=nome, semente=sem, yp=yp)
                linha["escore"] = sc if np.ndim(sc) == 1 else np.nan
                pred_linhas.append(linha)
                if nome in ("pmc", "rbf") and sem == sementes[0]:
                    pi = permutation_importance(m, X[te], y[te], scoring="balanced_accuracy",
                                                n_repeats=20, random_state=0, n_jobs=-1)
                    for a, mu, sd in zip(ATRIBUTOS, pi.importances_mean, pi.importances_std):
                        imp_linhas.append(dict(fold=fold, modelo=nome, atributo=a, importancia=mu, dp=sd))
        print(f"  [{tarefa}] fold {fold} concluído")

    pred = pd.concat(pred_linhas, ignore_index=True)
    pred.to_csv(saida / f"predicoes_{tarefa}.csv", index=False)
    pd.DataFrame(hip_linhas).to_csv(saida / f"hiperparametros_{tarefa}.csv", index=False)
    pd.DataFrame(tempo_linhas).to_csv(saida / f"tempo_treino_{tarefa}.csv", index=False)

    # Métricas por (modelo, semente, fold) -> média nas sementes -> média/dp nos folds
    regs = []
    for (mod, sem, fold), g in pred.groupby(["modelo", "semente", "fold"]):
        s = g["escore"].to_numpy() if g["escore"].notna().all() else None
        r = metricas(g["y"], g["yp"], s, binaria)
        r.update(modelo=mod, semente=sem, fold=fold, acuracia_video=voto_video(g))
        regs.append(r)
    met = pd.DataFrame(regs)
    por_fold = met.groupby(["modelo", "fold"]).mean(numeric_only=True).drop(columns="semente").reset_index()
    por_fold.to_csv(saida / f"metricas_por_fold_{tarefa}.csv", index=False, float_format="%.4f")
    resumo = por_fold.drop(columns="fold").groupby("modelo").agg(["mean", "std"])
    resumo.columns = [f"{a}_{b}".replace("_mean", "_media").replace("_std", "_dp") for a, b in resumo.columns]
    resumo.to_csv(saida / f"resumo_{tarefa}.csv", float_format="%.4f")

    # Por participante (para o teste pareado)
    pp_l = [dict(modelo=mod, semente=sem, participante=par,
                 acc_bal=balanced_accuracy_score(g["y"], g["yp"]))
            for (mod, sem, par), g in pred.groupby(["modelo", "semente", "participante"])]
    pp = (pd.DataFrame(pp_l).groupby(["modelo", "participante"])["acc_bal"].mean()
          .unstack(0))
    pp.to_csv(saida / f"metricas_por_participante_{tarefa}.csv", float_format="%.4f")
    testes = []
    for a in ["pmc", "rbf"]:
        for b in [c for c in pp.columns if c != a]:
            if (a, b) == ("rbf", "pmc"):
                continue
            d = pp[a] - pp[b]
            try:
                stat, p = wilcoxon(pp[a], pp[b], zero_method="zsplit")
            except ValueError:
                stat, p = np.nan, np.nan
            testes.append(dict(modelo_a=a, modelo_b=b, n_participantes=len(d),
                               diferenca_media=d.mean(), estatistica_W=stat, p_valor=p))
    pd.DataFrame(testes).to_csv(saida / f"testes_wilcoxon_{tarefa}.csv", index=False, float_format="%.4f")

    # Matrizes de confusão (soma dos folds, primeira semente)
    conf = {}
    for mod, g in pred[pred["semente"] == sementes[0]].groupby("modelo"):
        conf[mod] = confusion_matrix(g["y"], g["yp"], labels=sorted(df["y"].unique())).tolist()
    (saida / f"confusao_{tarefa}.json").write_text(json.dumps(conf, indent=2))

    if imp_linhas:
        imp = (pd.DataFrame(imp_linhas).groupby(["modelo", "atributo"])["importancia"]
               .agg(["mean", "std"]).reset_index().sort_values(["modelo", "mean"], ascending=[True, False]))
        imp.to_csv(saida / f"importancia_{tarefa}.csv", index=False, float_format="%.4f")

    print(f"\n=== Resumo ({tarefa}) ===")
    cols = [c for c in resumo.columns if c.split("_")[0] in ("acuracia", "f1", "auc", "sensibilidade")]
    print(resumo[cols].round(3).to_string())
    return resumo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atributos", default="saida/atributos.csv")
    ap.add_argument("--saida", default="saida/resultados")
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--tarefas", nargs="+", default=["binaria", "tres"])
    args = ap.parse_args()
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    sementes = list(range(args.sementes))
    t0 = time.time()
    for tarefa in args.tarefas:
        df = carregar(args.atributos, tarefa)
        print(f"\nTarefa {tarefa}: {len(df)} janelas, {df['participante'].nunique()} participantes, "
              f"folds {sorted(df['fold'].unique())}")
        rodar_tarefa(df, tarefa, sementes, saida)
    (saida / "parametros_atributos.json").write_text(json.dumps(PARAM, indent=2))
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min. Resultados em {saida.resolve()}")


if __name__ == "__main__":
    main()

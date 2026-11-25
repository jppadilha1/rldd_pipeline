"""
ETAPA 4 - Custo computacional (pedido do professor).

Mede, para Perceptron, Adaline, PMC e RBF treinados com todos os dados e com
os hiperparâmetros mais escolhidos na etapa 3:
  - número de parâmetros e multiplicações-acumulações (MACs) por inferência
  - tamanho em memória: pesos em float32 e arquivo serializado (joblib)
  - tempo de inferência de 1 janela (ms): via scikit-learn e via NumPy puro
    (forward implementado à mão, como seria num dispositivo embarcado)
  - tempo de treinamento
Mede também o custo das etapas anteriores do pipeline, para mostrar onde está
o gargalo: MediaPipe por quadro (lido dos .json da etapa 1) e cálculo dos
atributos de uma janela de 60 s.

Exemplo:
  python 04_custo_computacional.py --atributos saida/atributos.csv \
      --series saida/series --resultados saida/resultados
"""
import argparse
import ast
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from comum import ATRIBUTOS, PARAM, atributos_janela, preparar_video

sys.path.insert(0, str(Path(__file__).parent))
import importlib
etapa3 = importlib.import_module("03_treinar_avaliar")


def forward_numpy(m, nome):
    """Devolve (função de 1 amostra em float32, n_parametros, MACs).
    Reproduz à mão a inferência de cada rede, como num dispositivo embarcado."""
    imp = m.named_steps["imp"].statistics_.astype(np.float32)
    mu = m.named_steps["esc"].mean_.astype(np.float32)
    sd = m.named_steps["esc"].scale_.astype(np.float32)
    clf = m.named_steps["clf"]
    d = len(mu)
    pre = 2 * d  # média e desvio do z-score
    if nome in ("perceptron", "adaline"):
        W = clf.W_.astype(np.float32)[:, 0]  # W[0] é o limiar (entrada -1)

        def f(x):
            x = np.where(np.isnan(x), imp, x)
            return float(((x - mu) / sd) @ W[1:] - W[0]) >= 0
        return f, pre + clf.n_parametros(), d + 1
    if nome == "pmc":
        W1 = clf.W1_.astype(np.float32)
        W2 = clf.W2_.astype(np.float32)

        def f(x):
            x = np.where(np.isnan(x), imp, x)
            y1 = 1 / (1 + np.exp(-(W1[:, 1:] @ ((x - mu) / sd) - W1[:, 0])))
            y2 = 1 / (1 + np.exp(-(W2[:, 1:] @ y1 - W2[:, 0])))
            return float(y2[0]) >= 0.5
        return f, pre + clf.n_parametros(), W1.size + W2.size
    if nome == "rbf":
        C = clf.centros_.astype(np.float32)
        g = (1.0 / (2.0 * clf.sigmas_ ** 2)).astype(np.float32)
        W = clf.W_.astype(np.float32)[:, 0]

        def f(x):
            x = np.where(np.isnan(x), imp, x)
            z = (x - mu) / sd
            phi = np.exp(-(((z - C) ** 2).sum(1)) * g)
            return float(phi @ W[1:] - W[0]) >= 0
        k = C.shape[0]
        return f, pre + clf.n_parametros(), k * d + k + 1
    raise ValueError(nome)


def cronometrar(fn, n, aquecimento=200):
    for _ in range(aquecimento):
        fn()
    t = np.empty(n)
    for i in range(n):
        a = time.perf_counter_ns()
        fn()
        t[i] = time.perf_counter_ns() - a
    return np.median(t) / 1e6, np.percentile(t, 95) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atributos", default="saida/atributos.csv")
    ap.add_argument("--series", default="saida/series")
    ap.add_argument("--resultados", default="saida/resultados")
    ap.add_argument("--repeticoes", type=int, default=5000)
    args = ap.parse_args()
    res = Path(args.resultados)
    hip = pd.read_csv(res / "hiperparametros_binaria.csv")

    df = etapa3.carregar(args.atributos, "binaria")
    X, y = df[ATRIBUTOS], df["y"].to_numpy()
    x1_df = X.iloc[[0]]
    x1 = X.iloc[0].to_numpy(np.float32)

    linhas = []
    for nome, (est, _) in etapa3.MODELOS.items():
        params = etapa3.config_final(hip, nome)
        m = etapa3.clone(est).set_params(**params, clf__random_state=0)
        t0 = time.perf_counter()
        m.fit(X, y)
        t_fit = time.perf_counter() - t0
        f_np, n_par, macs = forward_numpy(m, nome)
        assert int(f_np(x1)) == int(m.predict(x1_df)[0]), "forward NumPy diverge do sklearn"
        buf = io.BytesIO()
        joblib.dump(m, buf)
        sk_med, sk_p95 = cronometrar(lambda: m.predict(x1_df), max(args.repeticoes // 5, 500))
        np_med, np_p95 = cronometrar(lambda: f_np(x1), args.repeticoes)
        linhas.append(dict(
            modelo=nome, hiperparametros=json.dumps({k.replace("clf__", ""): str(v) for k, v in params.items()}),
            n_parametros=n_par, macs_por_inferencia=macs,
            memoria_pesos_float32_kb=n_par * 4 / 1024, arquivo_joblib_kb=len(buf.getvalue()) / 1024,
            inferencia_sklearn_ms_mediana=sk_med, inferencia_sklearn_ms_p95=sk_p95,
            inferencia_numpy_ms_mediana=np_med, inferencia_numpy_ms_p95=np_p95,
            treino_s=t_fit, n_treino=len(X)))
        print(f"{nome:10s} params={n_par:5d}  float32={n_par * 4 / 1024:6.2f} KB  "
              f"sklearn={sk_med:.4f} ms  numpy={np_med:.4f} ms")
    pd.DataFrame(linhas).to_csv(res / "custo_computacional.csv", index=False, float_format="%.5f")

    # Custo das etapas de extração
    metas = [json.loads(p.read_text()) for p in Path(args.series).rglob("*.json")]
    metas = [m for m in metas if isinstance(m, dict) and "ms_mediapipe_mediana" in m]
    etapas = {}
    if metas:
        etapas["mediapipe_ms_quadro_mediana"] = float(np.median([m["ms_mediapipe_mediana"] for m in metas]))
        etapas["mediapipe_ms_quadro_p95"] = float(np.median([m["ms_mediapipe_p95"] for m in metas]))
        etapas["quadro_total_ms_mediana"] = float(np.median([m["ms_quadro_total_mediana"] for m in metas]))
        etapas["resolucao_processada"] = metas[0]["resolucao_processada"]
        etapas["fps_videos_mediana"] = float(np.median([m["fps"] for m in metas]))
        arq = next(Path(args.series).rglob("*.csv.gz"))
        serie = pd.read_csv(arq)
        trecho = serie[serie["t"] < PARAM["janela_s"]]

        def atr():
            prep = preparar_video(trecho)
            return atributos_janela(prep, 0.0, PARAM["janela_s"])
        med, p95 = cronometrar(atr, 200, aquecimento=10)
        etapas["atributos_janela60s_ms_mediana"] = med
        etapas["atributos_janela60s_ms_p95"] = p95
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        ram = None
    try:
        import mediapipe
        mp_v = mediapipe.__version__
    except Exception:
        mp_v = None
    ambiente = dict(sistema=platform.platform(), processador=platform.processor(),
                    nucleos_logicos=os.cpu_count(), ram_gb=ram, python=platform.python_version(),
                    numpy=np.__version__, sklearn=sklearn.__version__, pandas=pd.__version__,
                    mediapipe=mp_v, etapas_extracao=etapas)
    (res / "ambiente_e_custo_extracao.json").write_text(json.dumps(ambiente, indent=2, ensure_ascii=False))
    print(json.dumps(ambiente, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

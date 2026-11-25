"""
ETAPA 2 - Atributos interpretáveis por janela temporal.

Lê as séries da etapa 1 e, para cada vídeo, divide os 5 minutos em janelas de
60 s com passo de 30 s. Em cada janela calcula 14 atributos (PERCLOS, taxa e
duração de piscadas, amplitude e velocidade de reabertura, maior fechamento,
fechamentos longos, bocejos, abertura da boca e variabilidade da pose) mais
duas colunas usadas só pelo baseline de limiar fixo de EAR.

Saídas (pasta saida/):
  atributos.csv                    uma linha por janela
  resumo_atributos_por_classe.csv  média e desvio de cada atributo por classe
  qualidade_videos.csv             EAR de referência, % de rosto, nº de janelas

Exemplo:
  python 02_gerar_atributos.py --series saida/series --saida saida
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comum import ATRIBUTOS, NOMES_ROTULO, PARAM, janelas_do_video, preparar_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="saida/series")
    ap.add_argument("--saida", default="saida")
    args = ap.parse_args()

    arquivos = sorted(Path(args.series).rglob("*.csv.gz"))
    if not arquivos:
        raise SystemExit("Nenhuma série encontrada. Rode a etapa 1 antes.")

    linhas, qualidade = [], []
    for arq in arquivos:
        meta = json.loads(arq.with_suffix("").with_suffix(".json").read_text())
        serie = pd.read_csv(arq)
        jan = janelas_do_video(serie)
        prep = preparar_video(serie)
        validas = [j for j in jan if j["frac_valida"] >= PARAM["min_frac_valida"]]
        for j in validas:
            j.update(fold=meta["fold"], participante=str(meta["participante"]), rotulo=meta["rotulo"])
        linhas += validas
        qualidade.append(dict(fold=meta["fold"], participante=meta["participante"], rotulo=meta["rotulo"],
                              frac_face=meta["frac_face"], ear_ref=prep["ref"], rotacao=meta["rotacao"],
                              janelas_total=len(jan), janelas_validas=len(validas),
                              piscadas_por_min=len(prep["piscadas"]) / max(serie["t"].iloc[-1] / 60, 1e-9),
                              bocejos=len(prep["bocejos"]),
                              mar_p50=float(np.nanpercentile(serie["mar"], 50)),
                              mar_p99=float(np.nanpercentile(serie["mar"], 99))))

    df = pd.DataFrame(linhas)
    cols = ["fold", "participante", "rotulo", "janela", "t_ini", "frac_valida"] + ATRIBUTOS + \
           ["fech_max_fixo_s", "perclos_fixo"]
    df = df[cols].sort_values(["fold", "participante", "rotulo", "janela"])
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / "atributos.csv", index=False, float_format="%.6f")

    q = pd.DataFrame(qualidade).sort_values(["fold", "participante", "rotulo"])
    q.to_csv(saida / "qualidade_videos.csv", index=False, float_format="%.4f")

    resumo = df.groupby("rotulo")[ATRIBUTOS].agg(["mean", "std"]).T
    resumo.columns = [NOMES_ROTULO[c] for c in resumo.columns]
    resumo.to_csv(saida / "resumo_atributos_por_classe.csv", float_format="%.4f")

    print(f"Vídeos: {len(arquivos)} | participantes: {df['participante'].nunique()} | "
          f"janelas válidas: {len(df)}")
    print("Janelas por classe:", df["rotulo"].map(NOMES_ROTULO).value_counts().to_dict())
    print("\nMédia dos atributos por classe:")
    print(df.groupby(df["rotulo"].map(NOMES_ROTULO))[ATRIBUTOS].mean().T.round(3).to_string())
    ruins = q[q["frac_face"] < 0.7]
    if not ruins.empty:
        print("\nAVISO: vídeos com rosto detectado em menos de 70% dos quadros:")
        print(ruins[["fold", "participante", "rotulo", "frac_face", "rotacao"]].to_string(index=False))
    print(f"\nArquivos salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()

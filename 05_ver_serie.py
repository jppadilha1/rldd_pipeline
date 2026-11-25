"""
(Opcional) Inspeção visual de um vídeo processado: plota EAR normalizado, os
fechamentos detectados e o MAR, para conferir se os limiares fazem sentido.

Exemplo:
  python 05_ver_serie.py saida/series/fold1/01_10.csv.gz --de 0 --ate 60
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from comum import PARAM, preparar_video

ap = argparse.ArgumentParser()
ap.add_argument("serie")
ap.add_argument("--de", type=float, default=0)
ap.add_argument("--ate", type=float, default=60)
a = ap.parse_args()

s = pd.read_csv(a.serie)
p = preparar_video(s)
t = p["t"]
m = (t >= a.de) & (t < a.ate)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(t[m], p["ear_n"][m], lw=0.8, label="EAR / EAR olho aberto")
ax[0].axhline(PARAM["k_fechado"], color="r", ls="--", lw=0.8, label="limiar de fechamento")
for ev, cor in ((p["piscadas"], "tab:orange"), (p["longos"], "tab:red")):
    for e in ev:
        if a.de <= e["t"] < a.ate:
            ax[0].axvspan(e["t"], e["t"] + e["dur"], color=cor, alpha=0.3)
ax[0].set_ylabel("EAR normalizado")
ax[0].legend(loc="lower left", fontsize=8)
ax[1].plot(t[m], s["mar"].to_numpy()[m], lw=0.8, color="tab:green")
ax[1].axhline(PARAM["mar_limiar"], color="r", ls="--", lw=0.8)
ax[1].set_ylabel("MAR")
ax[1].set_xlabel("tempo (s)")
fig.suptitle(f"{a.serie} | EAR ref = {p['ref']:.3f} | piscadas: {len(p['piscadas'])} | "
             f"fech. longos: {len(p['longos'])} | bocejos: {len(p['bocejos'])}", fontsize=9)
fig.tight_layout()
saida = a.serie.replace(".csv.gz", f"_{int(a.de)}-{int(a.ate)}s.png")
fig.savefig(saida, dpi=150)
print("Figura salva em", saida)

"""
Código compartilhado do pipeline UTA-RLDD (PMC x RBF).

Contém: descoberta dos vídeos, índices dos pontos faciais do MediaPipe,
parâmetros de extração de atributos e as funções que transformam a série
temporal de um vídeo (EAR, MAR, pose) em atributos por janela.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Base de dados
# ---------------------------------------------------------------------------
# Na UTA-RLDD cada participante tem três vídeos nomeados pelo rótulo:
# 0 = alerta, 5 = baixa vigilância, 10 = sonolento.
ROTULOS = {"0": 0, "5": 5, "10": 10}
NOMES_ROTULO = {0: "alerta", 5: "baixa_vigilancia", 10: "sonolento"}
EXT_VIDEO = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".wmv", ".mts", ".3gp"}


def descobrir_videos(raiz):
    """Percorre a pasta da base e devolve um DataFrame com fold, participante,
    rótulo e caminho de cada vídeo. O fold é lido do nome de uma pasta
    ancestral do tipo 'Fold1_part1', 'Fold3', etc."""
    raiz = Path(raiz)
    itens = []
    for p in raiz.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in EXT_VIDEO or p.stem not in ROTULOS:
            continue
        fold = None
        for parte in p.relative_to(raiz).parts[:-1]:
            m = re.match(r"(?i)fold[\s_-]*(\d+)", parte)
            if m:
                fold = int(m.group(1))
        itens.append(dict(fold=fold, participante=str(p.parent.name),
                          rotulo=ROTULOS[p.stem], caminho=str(p)))
    cols = ["fold", "participante", "rotulo", "caminho"]
    df = pd.DataFrame(itens, columns=cols)
    return df.sort_values(["fold", "participante", "rotulo"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pontos faciais (MediaPipe Face Landmarker, 478 pontos)
# ---------------------------------------------------------------------------
# Ordem p1..p6 do EAR de Soukupová e Čech (2016):
# p1 e p4 = cantos; (p2,p6) e (p3,p5) = pares verticais.
OLHO_DIR = [33, 160, 158, 133, 153, 144]
OLHO_ESQ = [362, 385, 387, 263, 373, 380]
# MAR pelo contorno interno dos lábios: média de 3 distâncias verticais
# dividida pela largura da boca.
BOCA_CANTOS = (78, 308)
BOCA_VERT = [(82, 87), (13, 14), (312, 317)]


def _d(a, b):
    return float(np.linalg.norm(a - b))


def calc_ear(pts, idx):
    p = pts[idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-9)


def calc_mar(pts):
    largura = _d(pts[BOCA_CANTOS[0]], pts[BOCA_CANTOS[1]])
    vert = np.mean([_d(pts[a], pts[b]) for a, b in BOCA_VERT])
    return vert / (largura + 1e-9)


def angulos_euler(matriz):
    """Converte a matriz de transformação facial 4x4 em (pitch, yaw, roll)
    em graus."""
    R = np.asarray(matriz, dtype=float)[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


# ---------------------------------------------------------------------------
# Parâmetros dos atributos (documentados no artigo)
# ---------------------------------------------------------------------------
PARAM = dict(
    janela_s=60.0,          # tamanho da janela temporal
    passo_s=30.0,           # deslocamento entre janelas (50% de sobreposição)
    min_frac_valida=0.70,   # janela descartada se o rosto aparece em <70% dos quadros
    ref_percentil=90,       # EAR de olho aberto = percentil 90 do EAR do vídeo
    k_fechado=0.70,         # olho fechado se EAR < 0,70 x EAR de olho aberto
    piscada_min_s=0.06,     # duração mínima de uma piscada
    piscada_max_s=0.50,     # acima disso é "fechamento longo"
    mar_limiar=0.50,        # boca muito aberta
    bocejo_min_s=1.5,       # duração mínima de um bocejo
    ear_fixo=0.21,          # limiar fixo usado no baseline
    fixo_dur_s=1.0,         # baseline: alarme se olho fechado (EAR<0,21) por >= 1 s
)

ATRIBUTOS = [
    "ear_media", "ear_dp", "perclos",
    "taxa_piscadas", "dur_piscada_media", "amp_piscada_media", "vel_abertura_media",
    "fech_max_s", "taxa_fech_longos",
    "taxa_bocejos", "frac_boca_aberta",
    "pitch_dp", "yaw_dp", "roll_dp",
]


def _corridas(mask):
    """Índices (início, fim) inclusivos das sequências de True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    dm = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    ini = np.where(dm == 1)[0]
    fim = np.where(dm == -1)[0] - 1
    return list(zip(ini, fim))


def dp_circular(graus):
    g = np.asarray(graus, dtype=float)
    g = g[~np.isnan(g)]
    if g.size < 2:
        return np.nan
    a = np.radians(g)
    media = np.angle(np.mean(np.exp(1j * a)))
    desvio = np.angle(np.exp(1j * (a - media)))
    return float(np.degrees(np.std(desvio)))


def preparar_video(serie, param=PARAM):
    """Recebe a série de um vídeo (colunas t, face, ear, mar, pitch, yaw, roll)
    e devolve um dicionário com sinais filtrados e eventos detectados
    (piscadas, fechamentos longos, bocejos, fechamentos pelo limiar fixo)."""
    s = serie.sort_values("t").reset_index(drop=True)
    t = s["t"].to_numpy(float)
    valido = s["face"].to_numpy(int) == 1
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30

    ear = s["ear"].where(valido)
    ear_f = ear.rolling(3, center=True, min_periods=1).median().to_numpy(float)
    mar = s["mar"].where(valido).rolling(3, center=True, min_periods=1).median().to_numpy(float)

    ear_validos = ear_f[valido & ~np.isnan(ear_f)]
    ref = float(np.percentile(ear_validos, param["ref_percentil"])) if ear_validos.size else np.nan

    fechado = valido & (ear_f < param["k_fechado"] * ref)
    fechado_fixo = valido & (ear_f < param["ear_fixo"])
    boca_aberta = valido & (mar > param["mar_limiar"])

    piscadas, longos = [], []
    for i, f in _corridas(fechado):
        dur = t[f] - t[i] + dt
        if dur < param["piscada_min_s"]:
            continue
        i_min = i + int(np.nanargmin(ear_f[i:f + 1]))
        amp = (ref - ear_f[i_min]) / ref
        t_abre = t[f] - t[i_min] + dt
        ev = dict(t=t[i], dur=dur, amp=amp, vel=amp / t_abre)
        (piscadas if dur <= param["piscada_max_s"] else longos).append(ev)

    bocejos = [dict(t=t[i], dur=t[f] - t[i] + dt) for i, f in _corridas(boca_aberta)
               if t[f] - t[i] + dt >= param["bocejo_min_s"]]
    fixos = [dict(t=t[i], dur=t[f] - t[i] + dt) for i, f in _corridas(fechado_fixo)]

    return dict(t=t, dt=dt, valido=valido, ear_n=ear_f / ref, ref=ref, fechado=fechado,
                fechado_fixo=fechado_fixo, boca_aberta=boca_aberta,
                pitch=s["pitch"].where(valido).to_numpy(float),
                yaw=s["yaw"].where(valido).to_numpy(float),
                roll=s["roll"].where(valido).to_numpy(float),
                piscadas=piscadas, longos=longos, bocejos=bocejos, fixos=fixos)


def _eventos_em(evs, t0, t1):
    return [e for e in evs if t0 <= e["t"] < t1]


def atributos_janela(prep, t0, t1, param=PARAM):
    """Atributos interpretáveis de uma janela [t0, t1)."""
    t = prep["t"]
    sel = (t >= t0) & (t < t1)
    n = int(sel.sum())
    val = sel & prep["valido"]
    frac_valida = val.sum() / n if n else 0.0
    minutos = max(val.sum() * prep["dt"] / 60.0, 1e-9)

    pisc = _eventos_em(prep["piscadas"], t0, t1)
    longos = _eventos_em(prep["longos"], t0, t1)
    fech = pisc + longos
    fixos = _eventos_em(prep["fixos"], t0, t1)
    bocejos = _eventos_em(prep["bocejos"], t0, t1)
    ear_n = prep["ear_n"][val]

    def media(lst, chave):
        return float(np.mean([e[chave] for e in lst])) if lst else np.nan

    return dict(
        t_ini=t0, frac_valida=frac_valida,
        ear_media=float(np.nanmean(ear_n)) if ear_n.size else np.nan,
        ear_dp=float(np.nanstd(ear_n)) if ear_n.size else np.nan,
        perclos=float(prep["fechado"][val].mean()) if val.any() else np.nan,
        taxa_piscadas=len(pisc) / minutos,
        dur_piscada_media=media(pisc, "dur"),
        amp_piscada_media=media(pisc, "amp"),
        vel_abertura_media=media(pisc, "vel"),
        fech_max_s=max([e["dur"] for e in fech], default=0.0),
        taxa_fech_longos=len(longos) / minutos,
        taxa_bocejos=len(bocejos) / minutos,
        frac_boca_aberta=float(prep["boca_aberta"][val].mean()) if val.any() else np.nan,
        pitch_dp=dp_circular(prep["pitch"][val]),
        yaw_dp=dp_circular(prep["yaw"][val]),
        roll_dp=dp_circular(prep["roll"][val]),
        # colunas do baseline (não entram nas redes)
        fech_max_fixo_s=max([e["dur"] for e in fixos], default=0.0),
        perclos_fixo=float(prep["fechado_fixo"][val].mean()) if val.any() else np.nan,
    )


def janelas_do_video(serie, param=PARAM):
    prep = preparar_video(serie, param)
    t = prep["t"]
    if t.size == 0:
        return []
    linhas = []
    t0, idx = float(t[0]), 0
    while t0 + param["janela_s"] <= t[-1] + prep["dt"] + 1e-6:
        a = atributos_janela(prep, t0, t0 + param["janela_s"], param)
        a["janela"] = idx
        linhas.append(a)
        t0 += param["passo_s"]
        idx += 1
    return linhas

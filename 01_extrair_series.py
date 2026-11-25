"""
ETAPA 1 - Extração das séries temporais (EAR, MAR, pose da cabeça) dos vídeos.

É a etapa pesada. Cada vídeo é lido quadro a quadro (nunca inteiro na memória),
o MediaPipe Face Landmarker localiza os 478 pontos do rosto e salvamos, por
quadro: tempo, rosto detectado, EAR (olho esquerdo, direito e médio), MAR e
pitch/yaw/roll. Também mede o tempo de inferência do MediaPipe por quadro,
usado na tabela de custo computacional.

Pode ser executada em lotes (ex.: um fold por vez): vídeos já processados são
pulados, então dá para baixar o Fold 1, rodar, apagar os vídeos, baixar o
Fold 2, rodar de novo, e assim por diante.

Exemplos:
  python 01_extrair_series.py --raiz D:/RLDD --listar
  python 01_extrair_series.py --raiz D:/RLDD --por-fold 3
  python 01_extrair_series.py --raiz D:/RLDD --por-fold 3 --processos 4
  python 01_extrair_series.py --raiz D:/RLDD --todos
"""
import argparse
import json
import os
import platform
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from comum import (OLHO_DIR, OLHO_ESQ, angulos_euler, calc_ear, calc_mar,
                   descobrir_videos)

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
ROTACOES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def garantir_modelo(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        print(f"Baixando o modelo do MediaPipe para {caminho} ...")
        urllib.request.urlretrieve(URL_MODELO, caminho)
    return caminho


def criar_landmarker(caminho_modelo, modo_video=True):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(caminho_modelo)),
        running_mode=vision.RunningMode.VIDEO if modo_video else vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def preparar_quadro(frame, rot, largura_max):
    import mediapipe as mp
    if rot:
        frame = cv2.rotate(frame, ROTACOES[rot])
    h, w = frame.shape[:2]
    if w > largura_max:
        frame = cv2.resize(frame, (largura_max, int(round(h * largura_max / w))),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), rgb.shape


def escolher_rotacao(caminho, modelo, inicio_s, duracao_s, largura_max, n=24):
    """Alguns vídeos de celular vêm deitados. Testa 0/90/180/270 graus em
    quadros amostrados e fica com a rotação que mais detecta rosto."""
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    ini = int(inicio_s * fps)
    fim = int(min(total - 1, (inicio_s + duracao_s) * fps)) if total else ini + int(duracao_s * fps)
    alvos = np.linspace(ini, max(ini, fim), n).astype(int)
    quadros = []
    for f in alvos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            quadros.append(fr)
    cap.release()
    if not quadros:
        return 0, 0.0
    lm = criar_landmarker(modelo, modo_video=False)
    taxas = {}
    for rot in (0, 90, 270, 180):
        det = sum(1 for fr in quadros
                  if lm.detect(preparar_quadro(fr, rot, largura_max)[0]).face_landmarks)
        taxas[rot] = det / len(quadros)
        if rot == 0 and taxas[0] >= 0.8:
            break
    lm.close()
    melhor = max(taxas, key=taxas.get)
    return melhor, taxas[melhor]


def processar_video(item, modelo, saida, inicio_s, duracao_s, largura_max):
    caminho = item["caminho"]
    destino = Path(saida) / f"fold{item['fold']}" / f"{item['participante']}_{item['rotulo']}.csv.gz"
    meta_path = destino.with_suffix("").with_suffix(".json")
    if destino.exists() and meta_path.exists():
        return f"pulado (já existe): {destino.name}"
    destino.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        return f"ERRO ao abrir {caminho}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur_video = total / fps if total else float("nan")
    if str(inicio_s) == "centro":
        # trecho central: evita o início (ajuste da câmera, posicionamento) e o
        # fim (gesto de encerrar a gravação) sem supor como a sonolência evolui
        ini = max(0.0, (dur_video - duracao_s) / 2) if np.isfinite(dur_video) else 0.0
    else:
        ini = float(inicio_s)
        if np.isfinite(dur_video) and dur_video < ini + duracao_s:
            ini = max(0.0, dur_video - duracao_s)  # vídeo curto: pega o final
    cap.release()

    rot, taxa_rot = escolher_rotacao(caminho, modelo, ini, duracao_s, largura_max)

    cap = cv2.VideoCapture(caminho)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ini * fps)))
    lm = criar_landmarker(modelo, modo_video=True)

    linhas, tempos_inf, tempos_tot = [], [], []
    t_msec, idx = [], 0
    n_max = int(round(duracao_s * fps))
    forma = None
    while idx < n_max:
        a0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        img, forma = preparar_quadro(frame, rot, largura_max)
        h, w = forma[:2]
        ts = int(round(idx * 1000.0 / fps))  # monotônico, exigido pelo modo VIDEO
        a1 = time.perf_counter()
        res = lm.detect_for_video(img, ts)
        tempos_inf.append((time.perf_counter() - a1) * 1000)
        if res.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            e_d, e_e = calc_ear(pts, OLHO_DIR), calc_ear(pts, OLHO_ESQ)
            pitch, yaw, roll = (angulos_euler(res.facial_transformation_matrixes[0])
                                if res.facial_transformation_matrixes else (np.nan,) * 3)
            linhas.append((idx, 1, e_e, e_d, (e_e + e_d) / 2, calc_mar(pts), pitch, yaw, roll))
        else:
            linhas.append((idx, 0) + (np.nan,) * 7)
        tempos_tot.append((time.perf_counter() - a0) * 1000)
        idx += 1
    cap.release()
    lm.close()

    if not linhas:
        return f"ERRO: nenhum quadro lido em {caminho}"

    df = pd.DataFrame(linhas, columns=["quadro", "face", "ear_esq", "ear_dir", "ear",
                                       "mar", "pitch", "yaw", "roll"])
    # Tempo: usa o carimbo do contêiner se for crescente; senão, quadro/fps.
    tm = np.asarray(t_msec, dtype=float)
    if len(tm) > 2 and np.all(np.diff(tm) > 0):
        df.insert(1, "t", (tm - tm[0]) / 1000.0)
        fonte_t = "pos_msec"
    else:
        df.insert(1, "t", df["quadro"] / fps)
        fonte_t = "quadro/fps"

    tmp = destino.with_name(destino.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.5f", compression="gzip")
    os.replace(tmp, destino)
    meta = dict(
        fold=int(item["fold"]) if item["fold"] is not None else None,
        participante=item["participante"], rotulo=int(item["rotulo"]),
        arquivo=Path(caminho).name, fps=fps, duracao_video_s=dur_video,
        inicio_s=ini, duracao_s=float(df["t"].iloc[-1]), quadros=len(df),
        frac_face=float(df["face"].mean()), rotacao=rot, taxa_rotacao=taxa_rot,
        resolucao_processada=[int(forma[1]), int(forma[0])], fonte_tempo=fonte_t,
        ms_mediapipe_mediana=float(np.median(tempos_inf)),
        ms_mediapipe_p95=float(np.percentile(tempos_inf, 95)),
        ms_quadro_total_mediana=float(np.median(tempos_tot)),
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return (f"ok: fold{item['fold']} {item['participante']}_{item['rotulo']} | "
            f"{len(df)} quadros | rosto {meta['frac_face']:.0%} | rot {rot} | "
            f"{meta['ms_mediapipe_mediana']:.1f} ms/quadro")


def selecionar(videos, args):
    completos = (videos.groupby(["fold", "participante"])["rotulo"].nunique()
                 .reset_index().query("rotulo == 3"))
    if args.todos:
        return completos[["fold", "participante"]]
    if args.participantes:
        return completos[completos["participante"].isin(args.participantes)][["fold", "participante"]]
    sel_path = Path(args.saida) / "selecao_participantes.json"
    selecao = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    rng = random.Random(args.semente)
    for fold, grupo in completos.groupby("fold"):
        chave = str(fold)
        if chave in selecao:
            continue
        candidatos = sorted(grupo["participante"])
        rng_f = random.Random(f"{args.semente}-{fold}")
        selecao[chave] = sorted(rng_f.sample(candidatos, min(args.por_fold, len(candidatos))))
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(json.dumps(selecao, indent=2))
    pares = [(int(f), p) for f, ps in selecao.items() for p in ps]
    return pd.DataFrame(pares, columns=["fold", "participante"]).merge(
        completos[["fold", "participante"]], on=["fold", "participante"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", required=True, help="pasta que contém as pastas Fold*_part*")
    ap.add_argument("--saida", default="saida/series")
    ap.add_argument("--modelo", default="face_landmarker.task")
    ap.add_argument("--inicio", default="centro",
                    help="'centro' (padrão: trecho central do vídeo) ou segundos a pular no início")
    ap.add_argument("--duracao", type=float, default=300.0, help="segundos processados por vídeo (padrão 300)")
    ap.add_argument("--largura-max", type=int, default=640)
    ap.add_argument("--por-fold", type=int, default=3, help="participantes sorteados por fold")
    ap.add_argument("--participantes", nargs="*", help="lista explícita de IDs de participantes")
    ap.add_argument("--todos", action="store_true", help="processa todos os participantes encontrados")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--listar", action="store_true", help="só lista o que foi encontrado e sai")
    args = ap.parse_args()

    videos = descobrir_videos(args.raiz)
    if videos.empty:
        sys.exit("Nenhum vídeo encontrado. Confira --raiz (esperado: .../Fold1_part1/01/0.mov etc.).")
    if videos["fold"].isna().any():
        print("AVISO: vídeos sem fold identificado (pasta Fold* não encontrada no caminho):")
        print(videos[videos["fold"].isna()].head())
        videos = videos.dropna(subset=["fold"])
    videos["fold"] = videos["fold"].astype(int)

    resumo = videos.groupby("fold").agg(participantes=("participante", "nunique"), videos=("caminho", "size"))
    print("Encontrado:\n", resumo.to_string())
    if args.listar:
        return

    alvo = selecionar(videos, args)
    fila = videos.merge(alvo, on=["fold", "participante"]).to_dict("records")
    print(f"\nParticipantes selecionados: {len(alvo)} | vídeos na fila: {len(fila)}")
    print(alvo.groupby("fold")["participante"].apply(list).to_string())

    modelo = garantir_modelo(args.modelo)
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kw = dict(modelo=str(modelo), saida=args.saida, inicio_s=args.inicio,
              duracao_s=args.duracao, largura_max=args.largura_max)
    if args.processos <= 1:
        for i, it in enumerate(fila, 1):
            print(f"[{i}/{len(fila)}]", processar_video(it, **kw), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.processos) as ex:
            futs = [ex.submit(processar_video, it, **kw) for it in fila]
            for i, f in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(fila)}]", f.result(), flush=True)
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min. Séries em: {args.saida}")
    print(f"Máquina: {platform.platform()} | {platform.processor()} | {os.cpu_count()} núcleos")


if __name__ == "__main__":
    main()

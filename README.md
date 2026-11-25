# Pipeline PMC × RBF na UTA-RLDD

Detecção de sonolência com atributos faciais interpretáveis.
Avaliação experimental comparativa de modelos leves, com validação separada por participante.

## Instalação

Python 3.10, 3.11 ou 3.12 (64 bits).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Na primeira execução, a etapa 1 baixa sozinha o modelo `face_landmarker.task`, que tem cerca de 4 MB.
Se o download falhar, baixe manualmente pelo link `URL_MODELO` em `01_extrair_series.py` e coloque o arquivo nesta pasta.

## Organização da base

Depois de descompactar, a estrutura esperada é parecida com esta:

```
D:/RLDD/
  Fold1_part1/01/0.mov   5.mov   10.mov
  Fold1_part1/02/...
  Fold1_part2/...
  Fold2_part1/...
```

Não importa a extensão (.mov, .mp4, .MOV, .m4v). O script identifica o fold pelo nome da pasta `FoldN...`.
Para conferir o que ele encontrou sem processar nada:

```bash
python 01_extrair_series.py --raiz D:/RLDD --listar
```

## Ordem de execução

| Etapa | Comando | Pesado? |
|---|---|---|
| 1. Séries EAR/MAR/pose | `python 01_extrair_series.py --raiz D:/RLDD --por-fold 3 --processos 4` | **Sim**, várias horas |
| 2. Atributos por janela | `python 02_gerar_atributos.py` | Segundos |
| 3. Treino e avaliação | `python 03_treinar_avaliar.py` | Alguns minutos |
| 4. Custo computacional | `python 04_custo_computacional.py` | Menos de 1 min |
| 5. (opcional) Gráfico de um vídeo | `python 05_ver_serie.py saida/series/fold1/XX_10.csv.gz` | Segundos |

### Processamento em lotes (um fold por vez)

A base completa ocupa cerca de 111 GB, mas não é preciso ter tudo no disco ao mesmo tempo:

1. Baixe e descompacte só o Fold 1 (part1 e part2).
2. Rode a etapa 1. Ela sorteia 3 participantes desse fold com semente fixa e grava a escolha em `saida/series/selecao_participantes.json`.
3. Confira se saíram os arquivos `saida/series/fold1/*.csv.gz` e depois apague os vídeos.
4. Repita os passos para os Folds 2 a 5. Vídeos já processados são pulados, e se o processo for interrompido basta rodar de novo.

As etapas 2 a 4 só rodam depois que os 5 folds estiverem extraídos.

### Faça um teste rápido antes

Rode com um único participante e gere o gráfico dele:

```bash
python 01_extrair_series.py --raiz D:/RLDD --participantes 01
python 05_ver_serie.py saida/series/fold1/01_10.csv.gz --de 0 --ate 60
```

Abra o PNG e confira duas coisas: as piscadas aparecem como vales que cruzam a linha vermelha, e o rosto foi detectado em quase todo o vídeo (o percentual aparece no terminal).

## Parâmetros principais (em `comum.py`, dicionário `PARAM`)

- Trecho por vídeo: os 300 s centrais de cada vídeo (minutos 2,5 a 7,5 num vídeo de 10 min), evitando os artefatos de início e fim da gravação. Pode ser alterado com `--inicio` e `--duracao`.
- Janelas: 60 s com passo de 30 s. Janelas em que o rosto aparece em menos de 70% dos quadros são descartadas.
- Olho fechado: EAR abaixo de 0,70 × o EAR de olho aberto, que é o percentil 90 do próprio vídeo. Esse limiar adaptativo não usa nenhum rótulo.
- Piscada: fechamento de 0,06 a 0,5 s. Acima de 0,5 s conta como fechamento longo.
- Bocejo: MAR acima de 0,5 por pelo menos 1,5 s.
- Baseline de limiar fixo: alarme quando o EAR fica abaixo de 0,21 por pelo menos 1 s.

# -*- coding: utf-8 -*-
"""
LIMPEZA DOS DADOS COLETADOS DO YOUTUBE (TCC Aviator)

Reproduz, em um único script, todas as etapas de limpeza aplicadas à coleta
bruta feita pela API do YouTube, na mesma ordem em que foram feitas.

ENTRADAS (pasta Dados; só são lidas, nunca alteradas):
  videos.csv                         vídeos coletados pela API (457)
  comentarios.csv                    comentários coletados pela API (47.367 após remover duplicatas)
  videos_removidos_fora_do_tema.csv  37 vídeos removidos na revisão manual

SAÍDAS:
  videos_filtrados.csv               base final de vídeos
  comentarios_filtrados.csv          base final de comentários
  comentarios_spam_removidos.csv     comentários removidos como spam (para documentação)
"""

import sys
from pathlib import Path

import pandas as pd

# ----------------------------- configuração ---------------------------
DATA_INICIO = "2023-01-01"       # ano anterior à sanção da Lei 14.790/2023
MIN_COMENTARIOS_POR_VIDEO = 2    # vídeos com 0-1 comentário útil são removidos
PADRAO_PROMO = (r"(?:t\.me/|wa\.me/|https?://|www\.|bit\.ly|cadastr|clique aqui|"
                r"link na bio|meu link|c[oó]digo de indica[cç][aã]o)")

PASTA = Path(__file__).resolve().parent / "Dados"
COLUNAS_COMENTARIOS = ["comentario_id", "video_id", "autor", "autor_canal_id",
                       "texto", "likes", "data"]


def passo(titulo, n_videos, n_comentarios):
    print(f"  {titulo:<58} {n_videos:>5} vídeos {n_comentarios:>7} comentários")


for nome in ("videos.csv", "comentarios.csv", "videos_removidos_fora_do_tema.csv"):
    if not (PASTA / nome).exists():
        sys.exit(f"ERRO: arquivo não encontrado: {nome}")

# ------------------------ 0. leitura dos arquivos brutos --------------
videos = pd.read_csv(PASTA / "videos.csv", encoding="utf-8-sig", dtype=str)
# o comentarios.csv bruto pode estar sem cabeçalho (como foi coletado)
cab = (PASTA / "comentarios.csv").open(encoding="utf-8-sig").readline()
tem_cabecalho = cab.startswith("comentario_id")
com = pd.read_csv(PASTA / "comentarios.csv", encoding="utf-8-sig", dtype=str,
                  keep_default_na=False,
                  header=0 if tem_cabecalho else None,
                  names=None if tem_cabecalho else COLUNAS_COMENTARIOS)
com["likes"] = pd.to_numeric(com["likes"], errors="coerce")

print("=" * 90)
print("FUNIL DE LIMPEZA")
print("=" * 90)
passo("0. Coleta bruta (arquivos da API)", len(videos), len(com))

# ------------------- 1. duplicatas de ID + corte de período -----------
com = com.drop_duplicates(subset=["comentario_id"])
passo("1a. Remoção de comentario_id duplicado", len(videos), len(com))

videos["data_publicacao"] = pd.to_datetime(videos["data_publicacao"], utc=True)
videos = videos[videos["data_publicacao"] >= pd.Timestamp(DATA_INICIO, tz="UTC")]
com = com[com["video_id"].isin(set(videos["video_id"]))]
passo(f"1b. Só vídeos publicados desde {DATA_INICIO}", len(videos), len(com))

# ------------------- 2. vídeos fora do tema (revisão manual) ----------
fora = set(pd.read_csv(PASTA / "videos_removidos_fora_do_tema.csv", dtype=str)["video_id"])
videos = videos[~videos["video_id"].isin(fora)]
com = com[com["video_id"].isin(set(videos["video_id"]))]
passo("2. Remoção de vídeos fora do tema (revisão manual)", len(videos), len(com))

# ------------------- 3. spam / propaganda / bot -----------------------
texto = com["texto"].astype(str)
mask_promo = texto.str.contains(PADRAO_PROMO, case=False, na=False, regex=True)
repete_em_videos = com.groupby(["autor_canal_id", "texto"])["video_id"].transform("nunique")
mask_bot = repete_em_videos > 1
mask_spam = mask_promo | mask_bot
com[mask_spam].to_csv(PASTA / "comentarios_spam_removidos.csv", index=False, encoding="utf-8-sig")
com = com[~mask_spam]
passo("3. Remoção de spam (links/promoção ou texto repetido por conta)", len(videos), len(com))

# ------------------- 4. vídeos com quase nenhum comentário ------------
por_video = com.groupby("video_id").size()
manter = set(por_video[por_video >= MIN_COMENTARIOS_POR_VIDEO].index)
videos = videos[videos["video_id"].isin(manter)]
com = com[com["video_id"].isin(manter)]
passo(f"4. Remoção de vídeos com menos de {MIN_COMENTARIOS_POR_VIDEO} comentários", len(videos), len(com))

# ------------------------------- saída --------------------------------
videos.to_csv(PASTA / "videos_filtrados.csv", index=False, encoding="utf-8-sig")
com.to_csv(PASTA / "comentarios_filtrados.csv", index=False, encoding="utf-8-sig")
print(f"\nBASE FINAL: {len(videos)} vídeos, {len(com)} comentários")
print("Salvos: videos_filtrados.csv, comentarios_filtrados.csv, comentarios_spam_removidos.csv")

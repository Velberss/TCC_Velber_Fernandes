from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
import os
import pandas as pd
import time
from pathlib import Path
from langdetect import detect, LangDetectException

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("Configure a variável de ambiente YOUTUBE_API_KEY antes de executar.")

pasta = Path("Dados")
pasta.mkdir(exist_ok=True)

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

PALAVRAS_CHAVE = [
    "jogo do aviator",
    "aviator bet",
    "crash game apostas brasil",
    "como ganhar no aviator",
    "estratégia aviator",
    "bug aviator",
    "bet aviator",
    "dinheiro facil aviator",
]


ARQUIVO_PROGRESSO = pasta / "videos_ja_coletados.txt"

def carregar_progresso():
    """Retorna o conjunto de video_ids já coletados."""
    if ARQUIVO_PROGRESSO.exists():
        with open(ARQUIVO_PROGRESSO, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def salvar_progresso(video_id):
    with open(ARQUIVO_PROGRESSO, "a") as f:
        f.write(video_id + "\n")


def is_portugues(texto):
    try:
        return detect(texto) == "pt"
    except LangDetectException:
        return False

DATA_INICIO_COLETA = "2023-01-01"  # marco: ano anterior à sanção da Lei 14.790/2023

def buscar_videos(termo, max_resultados=100, data_inicio=DATA_INICIO_COLETA):

    videos = []
    next_page = None

    # Filtro de data: só vídeos publicados a partir do marco definido
    published_after = f"{data_inicio}T00:00:00Z"

    while len(videos) < max_resultados:

        request = youtube.search().list(
            q=termo,
            part="snippet",
            type="video",
            regionCode="BR",
            relevanceLanguage="pt",
            publishedAfter=published_after,   # <-- filtro adicionado aqui
            maxResults=min(100, max_resultados - len(videos)),
            pageToken=next_page
        )

        response = request.execute()

        for item in response["items"]:
            videos.append({
                "video_id": item["id"]["videoId"],
                "titulo": item["snippet"]["title"],
                "canal": item["snippet"]["channelTitle"],
                "canal_id": item["snippet"]["channelId"],
                "data_publicacao": item["snippet"]["publishedAt"],
                "palavra_chave": termo
            })

        next_page = response.get("nextPageToken")

        if not next_page:
            break

        time.sleep(0.5)

    return videos

def coletar_comentarios(video_id):

    comentarios = []
    next_page = None
    
    while True:

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page
        )

        response = request.execute()

        for item in response["items"]:

            comentario = item["snippet"]["topLevelComment"]["snippet"]
            texto = comentario["textDisplay"]

            # Filtra apenas comentários em português
            if not is_portugues(texto):
                continue

            comentarios.append({
                "comentario_id":
                    item["snippet"]["topLevelComment"]["id"],

                "video_id": video_id,

                "autor":
                    comentario["authorDisplayName"],

                "autor_canal_id":
                    comentario.get("authorChannelId", {}).get("value", ""),

                "texto": texto,

                "likes":
                    comentario["likeCount"],

                "data":
                    comentario["publishedAt"]
            })

        next_page = response.get("nextPageToken")

        if not next_page:
            break

        time.sleep(0.2)

    return comentarios


todos_videos = []

print("Buscando vídeos...")

for termo in PALAVRAS_CHAVE:

    print(f"  Pesquisando: {termo}")

    try:
        videos = buscar_videos(termo)
        todos_videos.extend(videos)
        print(f"  -> {len(videos)} vídeos encontrados")

    except Exception as e:
        print(f"  Erro na busca '{termo}': {e}")

    time.sleep(1)

# Remove duplicados
videos_df = pd.DataFrame(todos_videos).drop_duplicates(subset=["video_id"])

print(f"\nTotal de vídeos únicos: {len(videos_df)}")

videos_df.to_csv(
    pasta / "videos.csv",
    index=False,
    encoding="utf-8-sig"
)

ja_coletados = carregar_progresso()

# Filtra só os vídeos que ainda não foram processados
videos_pendentes = videos_df[~videos_df["video_id"].isin(ja_coletados)]

print(f"Vídeos já processados anteriormente: {len(ja_coletados)}")
print(f"Vídeos a processar agora: {len(videos_pendentes)}")
print("\nColetando comentários...\n")

primeiro_arquivo = not (pasta / "comentarios.csv").exists()
total_coletados = 0

for i, (_, video) in enumerate(videos_pendentes.iterrows()):

    video_id = video["video_id"]
    titulo_curto = video["titulo"][:50]

    print(f"[{i+1}/{len(videos_pendentes)}] {titulo_curto}")

    try:
        comentarios = coletar_comentarios(video_id)
        total_coletados += len(comentarios)
        print(f"  -> {len(comentarios)} comentários em português")

        if comentarios:
            df_parcial = pd.DataFrame(comentarios)
            df_parcial.to_csv(
                pasta / "comentarios.csv",
                mode="a",
                header=primeiro_arquivo,
                index=False,
                encoding="utf-8-sig"
            )
            primeiro_arquivo = False

    except Exception as e:
        codigo = str(e)
        if "commentsDisabled" in codigo:
            print(f"  -> Comentários desativados neste vídeo, pulando.")
        elif "quotaExceeded" in codigo:
            print("\n COTA DIÁRIA ESGOTADA! Rode novamente amanhã.")
            print(f"   Progresso salvo. Retome a partir do vídeo {i+1}.")
            break
        else:
            print(f"  -> Erro: {e}")

    # Salva progresso após cada vídeo
    salvar_progresso(video_id)

    # Relatório parcial a cada 50 vídeos
    if (i + 1) % 50 == 0:
        print(f"\n--- Parcial: {i+1} vídeos, {total_coletados} comentários ---\n")

print("\n" + "=" * 50)
print("COLETA FINALIZADA")
print("=" * 50)

if (pasta / "comentarios.csv").exists():
    df_final = pd.read_csv(pasta / "comentarios.csv")
    df_final = df_final.drop_duplicates(subset=["comentario_id"])
    df_final.to_csv(pasta / "comentarios.csv", index=False, encoding="utf-8-sig")

    print(f"Total de comentários: {len(df_final)}")
    print(f"Usuários únicos:      {df_final['autor_canal_id'].nunique()}")
    print(f"Vídeos com comentários: {df_final['video_id'].nunique()}")
    print(f"\nArquivos salvos em: {pasta}/")
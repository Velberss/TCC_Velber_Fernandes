from pathlib import Path

import joblib
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

PASTA_MODELOS = RAIZ_PROJETO / "resultados" / "modelos_finais"

CAMINHO_TFIDF = PASTA_MODELOS / "tfidf" / "modelo_tfidf.joblib"
CAMINHO_ROBERTA = PASTA_MODELOS / "roberta"

NOME_BASE_MANUAL = "base_tcc_final_1100.csv"
NOME_BASE_COMPLETA = "comentarios_filtrados.csv"

CANDIDATOS_BASE_MANUAL = [
    RAIZ_PROJETO / NOME_BASE_MANUAL,
    RAIZ_PROJETO / "data" / NOME_BASE_MANUAL,
    RAIZ_PROJETO / "dados" / NOME_BASE_MANUAL,
    RAIZ_PROJETO / "dataset" / NOME_BASE_MANUAL,
]

CANDIDATOS_BASE_COMPLETA = [
    RAIZ_PROJETO / NOME_BASE_COMPLETA,
    RAIZ_PROJETO / "data" / NOME_BASE_COMPLETA,
    RAIZ_PROJETO / "dados" / NOME_BASE_COMPLETA,
    RAIZ_PROJETO / "dataset" / NOME_BASE_COMPLETA,
]

PASTA_SAIDA = RAIZ_PROJETO / "resultados" / "classificacao_completa"

# Mesma ordem de classes utilizada no treinamento final.
ROTULOS = ["Contrário", "Favorável", "Neutro"]

COLUNA_ID_MANUAL = "youtube_comment_id"
COLUNA_ID_COMPLETA = "comentario_id"

COLUNA_TEXTO_MANUAL = "comment_text"
COLUNA_TEXTO_COMPLETA = "texto"
COLUNA_ROTULO_MANUAL = "rotulo_final"

COMPRIMENTO_MAXIMO = 128
TAMANHO_LOTE_ROBERTA = 16


# ============================================================
# LOCALIZAÇÃO DOS ARQUIVOS
# ============================================================

def localizar_arquivo(candidatos, nome):
    for caminho in candidatos:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        f"Não encontrei o arquivo '{nome}'.\n"
        "Verifique se ele está na raiz do projeto ou em uma "
        "das pastas: data, dados ou dataset."
    )


# ============================================================
# LEITURA E VALIDAÇÃO DAS BASES
# ============================================================

def carregar_bases():
    caminho_manual = localizar_arquivo(
        CANDIDATOS_BASE_MANUAL,
        NOME_BASE_MANUAL,
    )

    caminho_completa = localizar_arquivo(
        CANDIDATOS_BASE_COMPLETA,
        NOME_BASE_COMPLETA,
    )

    manual = pd.read_csv(
        caminho_manual,
        encoding="utf-8-sig",
        dtype={COLUNA_ID_MANUAL: str},
    )

    completa = pd.read_csv(
        caminho_completa,
        encoding="utf-8-sig",
        dtype={COLUNA_ID_COMPLETA: str},
    )

    colunas_manuais = {
        COLUNA_ID_MANUAL,
        COLUNA_ROTULO_MANUAL,
        COLUNA_TEXTO_MANUAL,
    }

    colunas_completas = {
        COLUNA_ID_COMPLETA,
        COLUNA_TEXTO_COMPLETA,
    }

    faltantes_manual = colunas_manuais - set(manual.columns)
    faltantes_completa = colunas_completas - set(completa.columns)

    if faltantes_manual:
        raise ValueError(
            f"Colunas ausentes na base manual: {sorted(faltantes_manual)}"
        )

    if faltantes_completa:
        raise ValueError(
            "Colunas ausentes na base completa: "
            f"{sorted(faltantes_completa)}"
        )

    # Normaliza IDs para facilitar a comparação.
    manual[COLUNA_ID_MANUAL] = (
        manual[COLUNA_ID_MANUAL].fillna("").astype(str).str.strip()
    )

    completa[COLUNA_ID_COMPLETA] = (
        completa[COLUNA_ID_COMPLETA].fillna("").astype(str).str.strip()
    )

    # Remove espaços excedentes dos textos.
    manual[COLUNA_TEXTO_MANUAL] = (
        manual[COLUNA_TEXTO_MANUAL].fillna("").astype(str).str.strip()
    )

    completa[COLUNA_TEXTO_COMPLETA] = (
        completa[COLUNA_TEXTO_COMPLETA].fillna("").astype(str).str.strip()
    )

    manual[COLUNA_ROTULO_MANUAL] = (
        manual[COLUNA_ROTULO_MANUAL].fillna("").astype(str).str.strip()
    )

    # Verifica se há IDs duplicados na base manual.
    ids_manuais_validos = manual[
        manual[COLUNA_ID_MANUAL] != ""
    ][COLUNA_ID_MANUAL]

    if ids_manuais_validos.duplicated().any():
        raise ValueError(
            "A base manual contém IDs de comentários duplicados. "
            "Verifique a base antes de continuar."
        )

    # Verifica IDs duplicados na base completa.
    ids_completos_validos = completa[
        completa[COLUNA_ID_COMPLETA] != ""
    ][COLUNA_ID_COMPLETA]

    if ids_completos_validos.duplicated().any():
        raise ValueError(
            "A base completa contém IDs de comentários duplicados. "
            "Verifique a base antes de continuar."
        )

    # Mantém apenas rótulos manuais válidos para identificar
    # comentários que já foram classificados.
    manual_valida = manual[
        manual[COLUNA_ID_MANUAL].ne("")
        & manual[COLUNA_ROTULO_MANUAL].isin(ROTULOS)
    ].copy()

    ids_manuais = set(manual_valida[COLUNA_ID_MANUAL])

    # Identifica os comentários da base completa que já possuem
    # classificação manual.
    completa["classificacao_manual"] = completa[
        COLUNA_ID_COMPLETA
    ].map(
        manual_valida.set_index(COLUNA_ID_MANUAL)[COLUNA_ROTULO_MANUAL]
    )

    completa["status_classificacao"] = "Pendente"

    completa.loc[
        completa[COLUNA_ID_COMPLETA].isin(ids_manuais),
        "status_classificacao",
    ] = "Classificado manualmente"

    # Um comentário só pode ser classificado automaticamente
    # se possuir ID e texto.
    elegiveis = (
        completa[COLUNA_ID_COMPLETA].ne("")
        & completa[COLUNA_TEXTO_COMPLETA].ne("")
        & ~completa[COLUNA_ID_COMPLETA].isin(ids_manuais)
    )

    indices_pendentes = completa.index[elegiveis].tolist()

    print("=" * 65)
    print("VALIDAÇÃO DAS BASES")
    print("=" * 65)
    print(f"Base manual: {caminho_manual}")
    print(f"Base completa: {caminho_completa}")
    print(f"Comentários na base completa: {len(completa)}")
    print(f"Comentários com classificação manual correspondente: "
          f"{completa['status_classificacao'].eq('Classificado manualmente').sum()}")
    print(f"Comentários elegíveis para classificação automática: "
          f"{len(indices_pendentes)}")

    return completa, indices_pendentes


# ============================================================
# CLASSIFICAÇÃO COM TF-IDF
# ============================================================

def classificar_tfidf(textos):
    print("\nCarregando modelo TF-IDF...")

    if not CAMINHO_TFIDF.exists():
        raise FileNotFoundError(
            f"Modelo TF-IDF não encontrado: {CAMINHO_TFIDF}\n"
            "Execute primeiro o script treinar_modelos_finais.py."
        )

    modelo = joblib.load(CAMINHO_TFIDF)

    previsoes = modelo.predict(textos)
    probabilidades = modelo.predict_proba(textos)

    classes_modelo = list(modelo.classes_)

    resultados = []

    for rotulo, probabilidades_item in zip(previsoes, probabilidades):
        indice_maior = probabilidades_item.argmax()

        resultados.append({
            "predicao_tfidf": str(rotulo),
            "confianca_tfidf": float(probabilidades_item[indice_maior]),
        })

    print(f"TF-IDF: {len(resultados)} comentários classificados.")

    return resultados


# ============================================================
# CLASSIFICAÇÃO COM ROBERTA
# ============================================================

def classificar_roberta(textos):
    print("\nCarregando modelo RoBERTa...")

    if not CAMINHO_ROBERTA.exists():
        raise FileNotFoundError(
            f"Modelo RoBERTa não encontrado: {CAMINHO_ROBERTA}\n"
            "Execute primeiro o script treinar_modelos_finais.py."
        )

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo de inferência: {dispositivo}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(CAMINHO_ROBERTA)
    )

    modelo = AutoModelForSequenceClassification.from_pretrained(
        str(CAMINHO_ROBERTA)
    )

    modelo.to(dispositivo)
    modelo.eval()

    resultados = []

    for inicio in range(0, len(textos), TAMANHO_LOTE_ROBERTA):
        lote_textos = textos[
            inicio:inicio + TAMANHO_LOTE_ROBERTA
        ]

        entradas = tokenizer(
            lote_textos,
            padding=True,
            truncation=True,
            max_length=COMPRIMENTO_MAXIMO,
            return_tensors="pt",
        )

        entradas = {
            chave: valor.to(dispositivo)
            for chave, valor in entradas.items()
        }

        with torch.no_grad():
            saidas = modelo(**entradas)
            probabilidades = torch.softmax(
                saidas.logits,
                dim=-1,
            )

        probabilidades = probabilidades.cpu()

        for probabilidades_item in probabilidades:
            indice = int(probabilidades_item.argmax().item())

            resultados.append({
                "predicao_roberta": ROTULOS[indice],
                "confianca_roberta": float(
                    probabilidades_item[indice].item()
                ),
            })

        processados = min(
            inicio + len(lote_textos),
            len(textos),
        )

        print(
            f"RoBERTa: {processados}/{len(textos)} comentários processados.",
            end="\r",
            flush=True,
        )

    print(f"\nRoBERTa: {len(resultados)} comentários classificados.")

    return resultados


# ============================================================
# EXECUÇÃO DA CLASSIFICAÇÃO
# ============================================================

def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    completa, indices_pendentes = carregar_bases()

    # Prepara as colunas de saída.
    completa["predicao_tfidf"] = pd.NA
    completa["confianca_tfidf"] = pd.NA
    completa["predicao_roberta"] = pd.NA
    completa["confianca_roberta"] = pd.NA
    completa["concordancia_modelos"] = pd.NA

    if not indices_pendentes:
        print(
            "\nNão há comentários elegíveis para classificação automática."
        )

        caminho_consolidado = (
            PASTA_SAIDA / "comentarios_classificados_consolidado.csv"
        )

        completa.to_csv(
            caminho_consolidado,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Arquivo salvo em: {caminho_consolidado}")
        return

    textos = completa.loc[
        indices_pendentes,
        COLUNA_TEXTO_COMPLETA,
    ].tolist()

    # Classificação pelos dois modelos.
    resultados_tfidf = classificar_tfidf(textos)
    resultados_roberta = classificar_roberta(textos)

    df_tfidf = pd.DataFrame(
        resultados_tfidf,
        index=indices_pendentes,
    )

    df_roberta = pd.DataFrame(
        resultados_roberta,
        index=indices_pendentes,
    )

    # Incorpora previsões à base completa.
    for coluna in df_tfidf.columns:
        completa.loc[indices_pendentes, coluna] = df_tfidf[coluna]

    for coluna in df_roberta.columns:
        completa.loc[indices_pendentes, coluna] = df_roberta[coluna]

    completa.loc[
        indices_pendentes,
        "concordancia_modelos",
    ] = (
        completa.loc[indices_pendentes, "predicao_tfidf"]
        == completa.loc[indices_pendentes, "predicao_roberta"]
    ).map({
        True: "Sim",
        False: "Não",
    })

    completa.loc[
        indices_pendentes,
        "status_classificacao",
    ] = "Classificado automaticamente"

    # Mantém os comentários sem ID ou sem texto identificados.
    indices_sem_id = completa.index[
        completa[COLUNA_ID_COMPLETA].eq("")
    ]

    completa.loc[
        indices_sem_id,
        "status_classificacao",
    ] = "Não classificado: sem ID"

    indices_sem_texto = completa.index[
        completa[COLUNA_TEXTO_COMPLETA].eq("")
        & completa[COLUNA_ID_COMPLETA].ne("")
        & ~completa[COLUNA_ID_COMPLETA].isin(
            completa.loc[
                completa["status_classificacao"].eq(
                    "Classificado manualmente"
                ),
                COLUNA_ID_COMPLETA,
            ]
        )
    ]

    completa.loc[
        indices_sem_texto,
        "status_classificacao",
    ] = "Não classificado: sem texto"

    # Arquivo apenas com as previsões automáticas.
    automaticos = completa.loc[
        indices_pendentes
    ].copy()

    caminho_automaticos = (
        PASTA_SAIDA / "comentarios_classificados_automaticamente.csv"
    )

    automaticos.to_csv(
        caminho_automaticos,
        index=False,
        encoding="utf-8-sig",
    )

    # Arquivo consolidado com comentários manuais e automáticos.
    caminho_consolidado = (
        PASTA_SAIDA / "comentarios_classificados_consolidado.csv"
    )

    completa.to_csv(
        caminho_consolidado,
        index=False,
        encoding="utf-8-sig",
    )

    # Resumo da classificação.
    resumo = pd.DataFrame({
        "indicador": [
            "Total de comentários na base",
            "Classificados manualmente",
            "Classificados automaticamente",
            "Concordância entre TF-IDF e RoBERTa",
            "Discordância entre TF-IDF e RoBERTa",
        ],
        "quantidade": [
            len(completa),
            int(
                completa["status_classificacao"]
                .eq("Classificado manualmente")
                .sum()
            ),
            len(automaticos),
            int(
                automaticos["concordancia_modelos"]
                .eq("Sim")
                .sum()
            ),
            int(
                automaticos["concordancia_modelos"]
                .eq("Não")
                .sum()
            ),
        ],
    })

    caminho_resumo = PASTA_SAIDA / "resumo_classificacao.csv"

    resumo.to_csv(
        caminho_resumo,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("CLASSIFICAÇÃO CONCLUÍDA")
    print("=" * 65)

    print(f"\n{resumo.to_string(index=False)}")

    print("\nArquivos gerados:")
    print(f"- Previsões automáticas: {caminho_automaticos}")
    print(f"- Base consolidada: {caminho_consolidado}")
    print(f"- Resumo: {caminho_resumo}")


if __name__ == "__main__":
    main()

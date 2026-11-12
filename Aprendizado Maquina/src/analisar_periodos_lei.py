from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "resultados/classificacao_completa/"
    "comentarios_classificados_consolidado.csv"
)

PASTA_SAIDA = Path("resultados/analise_temporal")

DATA_CORTE = pd.Timestamp("2023-12-29")

COLUNA_DATA = "data"

COLUNAS_MODELOS = {
    "TF-IDF": "predicao_tfidf",
    "RoBERTa": "predicao_roberta",
}

COLUNA_MANUAL = "classificacao_manual"

ROTULOS_VALIDOS = [
    "Favorável",
    "Contrário",
    "Neutro",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar_rotulo(valor):
    """
    Padroniza os rótulos para evitar diferenças causadas
    por espaços ou letras maiúsculas/minúsculas.
    """
    if pd.isna(valor):
        return None

    valor = str(valor).strip().casefold()

    rotulos = {
        "favorável": "Favorável",
        "favoravel": "Favorável",
        "contrário": "Contrário",
        "contrario": "Contrário",
        "neutro": "Neutro",
    }

    return rotulos.get(valor)


def classificar_periodo(data):
    """
    Separa os comentários usando 29/12/2023 como data de corte.
    """
    if pd.isna(data):
        return "Data inválida"

    if data < DATA_CORTE:
        return "Antes da lei"

    return "Depois da lei"


def gerar_distribuicao(df, coluna_rotulo, nome_modelo):
    """
    Calcula a distribuição dos rótulos considerando apenas
    os comentários que possuem classificação válida na coluna.
    """
    dados = df.copy()

    dados["rotulo"] = dados[coluna_rotulo].apply(
        normalizar_rotulo
    )

    # Considera somente os três rótulos válidos.
    dados = dados[dados["rotulo"].isin(ROTULOS_VALIDOS)]

    resultados = []

    for periodo in ["Antes da lei", "Depois da lei"]:
        grupo = dados[dados["periodo"] == periodo]

        total_classificados = len(grupo)

        for rotulo in ROTULOS_VALIDOS:
            quantidade = (grupo["rotulo"] == rotulo).sum()

            percentual = (
                quantidade / total_classificados * 100
                if total_classificados > 0
                else 0
            )

            resultados.append({
                "modelo": nome_modelo,
                "periodo": periodo,
                "rotulo": rotulo,
                "quantidade": int(quantidade),
                "total_classificados": total_classificados,
                "percentual": round(percentual, 2),
            })

    return pd.DataFrame(resultados)


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("=" * 65)
    print("ANÁLISE TEMPORAL DOS COMENTÁRIOS")
    print("=" * 65)

    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {ARQUIVO_ENTRADA}"
        )

    # Lê o arquivo original sem modificá-lo.
    df = pd.read_csv(
        ARQUIVO_ENTRADA,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(f"\nTotal de linhas lidas: {len(df)}")
    print(f"Colunas encontradas: {list(df.columns)}")

    # Verifica as colunas necessárias.
    colunas_necessarias = [
        COLUNA_DATA,
        COLUNA_MANUAL,
        *COLUNAS_MODELOS.values(),
    ]

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas necessárias não encontradas: {faltantes}"
        )

    # Converte as datas e registra quais são válidas.
    df[COLUNA_DATA] = pd.to_datetime(
        df[COLUNA_DATA],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    # Separa os comentários por período.
    df["periodo"] = df[COLUNA_DATA].apply(classificar_periodo)

    # Cria a pasta de saída.
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. QUANTIDADE TOTAL POR PERÍODO
    # --------------------------------------------------------

    total_validas = df[df["periodo"] != "Data inválida"]

    quantidade_periodos = (
        total_validas.groupby("periodo")
        .size()
        .reindex(["Antes da lei", "Depois da lei"], fill_value=0)
        .rename("quantidade")
        .reset_index()
    )

    total_com_datas_validas = quantidade_periodos["quantidade"].sum()

    quantidade_periodos["percentual_base"] = (
        quantidade_periodos["quantidade"]
        / total_com_datas_validas
        * 100
    ).round(2)

    quantidade_periodos.to_csv(
        PASTA_SAIDA / "quantidade_por_periodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nQuantidade de comentários por período:")
    print(quantidade_periodos.to_string(index=False))

    # --------------------------------------------------------
    # 2. DISTRIBUIÇÃO DAS PREVISÕES AUTOMÁTICAS
    # --------------------------------------------------------

    distribuicoes_modelos = []

    for nome_modelo, coluna in COLUNAS_MODELOS.items():
        distribuicao = gerar_distribuicao(
            df=df,
            coluna_rotulo=coluna,
            nome_modelo=nome_modelo,
        )

        distribuicoes_modelos.append(distribuicao)

    distribuicao_automatica = pd.concat(
        distribuicoes_modelos,
        ignore_index=True,
    )

    distribuicao_automatica.to_csv(
        PASTA_SAIDA / "distribuicao_rotulos_por_periodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDistribuição das previsões automáticas:")
    print(distribuicao_automatica.to_string(index=False))

    # --------------------------------------------------------
    # 3. DISTRIBUIÇÃO DAS CLASSIFICAÇÕES MANUAIS
    # --------------------------------------------------------

    distribuicao_manual = gerar_distribuicao(
        df=df,
        coluna_rotulo=COLUNA_MANUAL,
        nome_modelo="Classificação manual",
    )

    distribuicao_manual.to_csv(
        PASTA_SAIDA / "distribuicao_manual_por_periodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDistribuição das classificações manuais:")
    print(distribuicao_manual.to_string(index=False))

    # --------------------------------------------------------
    # 4. CONFERÊNCIA DAS DATAS E COBERTURA DAS CLASSIFICAÇÕES
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 4. CONCORDÂNCIA ENTRE TF-IDF E ROBERTA POR PERÍODO
    # --------------------------------------------------------

    coluna_tfidf = COLUNAS_MODELOS["TF-IDF"]
    coluna_roberta = COLUNAS_MODELOS["RoBERTa"]

    # Normaliza as previsões dos dois modelos.
    df["rotulo_tfidf"] = df[coluna_tfidf].apply(
        normalizar_rotulo
    )

    df["rotulo_roberta"] = df[coluna_roberta].apply(
        normalizar_rotulo
    )

    # Considera apenas comentários com previsões válidas
    # nos dois modelos.
    previsoes_validas = df[
        df["rotulo_tfidf"].isin(ROTULOS_VALIDOS)
        & df["rotulo_roberta"].isin(ROTULOS_VALIDOS)
        & df["periodo"].isin(["Antes da lei", "Depois da lei"])
    ].copy()

    # Verifica se os modelos concordam em cada comentário.
    previsoes_validas["concordam"] = (
        previsoes_validas["rotulo_tfidf"]
        == previsoes_validas["rotulo_roberta"]
    )

    resultados_concordancia = []

    for periodo in ["Antes da lei", "Depois da lei"]:

        grupo = previsoes_validas[
            previsoes_validas["periodo"] == periodo
        ]

        total = len(grupo)
        concordantes = int(grupo["concordam"].sum())
        divergentes = total - concordantes

        percentual_concordancia = (
            concordantes / total * 100 if total else 0
        )

        percentual_divergencia = (
            divergentes / total * 100 if total else 0
        )

        resultados_concordancia.append({
            "periodo": periodo,
            "total_com_duas_previsoes": total,
            "concordantes": concordantes,
            "divergentes": divergentes,
            "percentual_concordancia": round(
                percentual_concordancia, 2
            ),
            "percentual_divergencia": round(
                percentual_divergencia, 2
            ),
        })

    tabela_concordancia = pd.DataFrame(
        resultados_concordancia
    )

    tabela_concordancia.to_csv(
        PASTA_SAIDA / "concordancia_por_periodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nConcordância entre TF-IDF e RoBERTa por período:")
    print(tabela_concordancia.to_string(index=False))

    # --------------------------------------------------------
    # 5. DETALHAMENTO DAS DIVERGÊNCIAS
    # --------------------------------------------------------

    divergencias = previsoes_validas[
        ~previsoes_validas["concordam"]
    ].copy()

    divergencias["comparacao"] = (
        divergencias["rotulo_tfidf"]
        + " -> "
        + divergencias["rotulo_roberta"]
    )

    tabela_divergencias = (
        divergencias
        .groupby(["periodo", "comparacao"])
        .size()
        .reset_index(name="quantidade")
    )

    # Calcula o percentual de cada tipo de divergência
    # em relação ao total de divergências daquele período.
    if not tabela_divergencias.empty:
        tabela_divergencias["total_divergencias_periodo"] = (
            tabela_divergencias
            .groupby("periodo")["quantidade"]
            .transform("sum")
        )

        tabela_divergencias["percentual"] = (
            tabela_divergencias["quantidade"]
            / tabela_divergencias["total_divergencias_periodo"]
            * 100
        ).round(2)

    tabela_divergencias.to_csv(
        PASTA_SAIDA / "detalhamento_divergencias_por_periodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDetalhamento das divergências:")
    print(tabela_divergencias.to_string(index=False))

if __name__ == "__main__":
    main()

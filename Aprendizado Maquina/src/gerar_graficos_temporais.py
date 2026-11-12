from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

PASTA_RESULTADOS = (
    PASTA_PROJETO
    / "resultados"
    / "analise_temporal"
)

PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"

# Cria a pasta de gráficos caso ainda não exista.
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def carregar_csv(nome_arquivo):
    """
    Carrega um CSV da pasta de resultados.
    """

    caminho = PASTA_RESULTADOS / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    return pd.read_csv(
        caminho,
        encoding="utf-8-sig",
    )


def salvar_grafico(nome_arquivo):
    """
    Ajusta o layout e salva o gráfico em PNG com 300 DPI.
    """

    caminho = PASTA_GRAFICOS / nome_arquivo

    plt.tight_layout()

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo: {caminho}")


def adicionar_rotulos_barras(ax, formato="%.2f%%"):
    """
    Adiciona os valores numéricos às barras de um gráfico.
    """

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=formato,
            padding=3,
        )


# ============================================================
# GRÁFICO 1 — QUANTIDADE DE COMENTÁRIOS POR PERÍODO
# ============================================================

def grafico_quantidade_por_periodo():
    """
    Apresenta a quantidade e o percentual de comentários
    em cada período analisado.
    """

    df = carregar_csv(
        "quantidade_por_periodo.csv"
    )

    # Mantém a ordem cronológica dos períodos.
    ordem_periodos = [
        "Antes da lei",
        "Depois da lei",
    ]

    df["periodo"] = pd.Categorical(
        df["periodo"],
        categories=ordem_periodos,
        ordered=True,
    )

    df = df.sort_values("periodo")

    plt.figure(figsize=(10, 6))

    barras = plt.bar(
        df["periodo"].astype(str),
        df["quantidade"],
    )

    plt.title(
        "Quantidade de comentários por período",
        fontsize=14,
    )

    plt.xlabel("Período")
    plt.ylabel("Quantidade de comentários")

    # Mostra quantidade e percentual acima de cada barra.
    rotulos = [
        (
            f"{quantidade:,.0f}".replace(",", ".")
            + f"\n({percentual:.2f}%)"
        )
        for quantidade, percentual in zip(
            df["quantidade"],
            df["percentual_base"],
        )
    ]

    plt.bar_label(
        barras,
        labels=rotulos,
        padding=4,
    )

    salvar_grafico(
        "grafico_01_quantidade_por_periodo.png"
    )


# ============================================================
# GRÁFICO 2 — DISTRIBUIÇÃO DAS CLASSIFICAÇÕES DO TF-IDF
# ============================================================

def grafico_distribuicao_tfidf():
    """
    Compara a distribuição percentual das classificações
    automáticas do TF-IDF entre os períodos.
    """

    df = carregar_csv(
        "distribuicao_rotulos_por_periodo.csv"
    )

    # Seleciona somente os resultados do TF-IDF.
    df = df[
        df["modelo"]
        .astype(str)
        .str.upper()
        == "TF-IDF"
    ].copy()

    if df.empty:
        raise ValueError(
            "Não foram encontradas linhas do modelo TF-IDF."
        )

    tabela = df.pivot(
        index="rotulo",
        columns="periodo",
        values="percentual",
    )

    ordem_rotulos = [
        "Favorável",
        "Contrário",
        "Neutro",
    ]

    tabela = tabela.reindex(
        ordem_rotulos
    )

    ordem_periodos = [
        "Antes da lei",
        "Depois da lei",
    ]

    tabela = tabela[
        [
            periodo
            for periodo in ordem_periodos
            if periodo in tabela.columns
        ]
    ]

    ax = tabela.plot(
        kind="bar",
        figsize=(11, 7),
    )

    ax.set_title(
        "Distribuição das classificações automáticas "
        "por período — TF-IDF",
        fontsize=13,
    )

    ax.set_xlabel("Classificação prevista")
    ax.set_ylabel("Percentual dos comentários (%)")

    ax.legend(
        title="Período",
    )

    ax.set_xticklabels(
        tabela.index,
        rotation=0,
    )

    adicionar_rotulos_barras(
        ax,
        formato="%.2f%%",
    )

    salvar_grafico(
        "grafico_02_distribuicao_tfidf.png"
    )


# ============================================================
# GRÁFICO 3 — DISTRIBUIÇÃO DAS CLASSIFICAÇÕES DO ROBERTA
# ============================================================

def grafico_distribuicao_roberta():
    """
    Compara a distribuição percentual das classificações
    automáticas do RoBERTa entre os períodos.
    """

    df = carregar_csv(
        "distribuicao_rotulos_por_periodo.csv"
    )

    # Seleciona somente os resultados do RoBERTa.
    df = df[
        df["modelo"]
        .astype(str)
        .str.upper()
        .str.contains("ROBERTA")
    ].copy()

    if df.empty:
        raise ValueError(
            "Não foram encontradas linhas do modelo RoBERTa."
        )

    tabela = df.pivot(
        index="rotulo",
        columns="periodo",
        values="percentual",
    )

    ordem_rotulos = [
        "Favorável",
        "Contrário",
        "Neutro",
    ]

    tabela = tabela.reindex(
        ordem_rotulos
    )

    ordem_periodos = [
        "Antes da lei",
        "Depois da lei",
    ]

    tabela = tabela[
        [
            periodo
            for periodo in ordem_periodos
            if periodo in tabela.columns
        ]
    ]

    ax = tabela.plot(
        kind="bar",
        figsize=(11, 7),
    )

    ax.set_title(
        "Distribuição das classificações automáticas "
        "por período — RoBERTa",
        fontsize=13,
    )

    ax.set_xlabel("Classificação prevista")
    ax.set_ylabel("Percentual dos comentários (%)")

    ax.legend(
        title="Período",
    )

    ax.set_xticklabels(
        tabela.index,
        rotation=0,
    )

    adicionar_rotulos_barras(
        ax,
        formato="%.2f%%",
    )

    salvar_grafico(
        "grafico_03_distribuicao_roberta.png"
    )


# ============================================================
# GRÁFICO 4 — CONCORDÂNCIA ENTRE TF-IDF E ROBERTA
# ============================================================

def grafico_concordancia():
    """
    Compara os percentuais de concordância e divergência
    entre TF-IDF e RoBERTa em cada período.
    """

    df = carregar_csv(
        "concordancia_por_periodo.csv"
    )

    ordem_periodos = [
        "Antes da lei",
        "Depois da lei",
    ]

    df["periodo"] = pd.Categorical(
        df["periodo"],
        categories=ordem_periodos,
        ordered=True,
    )

    df = df.sort_values("periodo")

    plt.figure(figsize=(10, 6))

    x = list(range(len(df)))

    largura = 0.36

    barras_concordancia = plt.bar(
        [
            i - largura / 2
            for i in x
        ],
        df["percentual_concordancia"],
        width=largura,
        label="Concordância",
    )

    barras_divergencia = plt.bar(
        [
            i + largura / 2
            for i in x
        ],
        df["percentual_divergencia"],
        width=largura,
        label="Divergência",
    )

    plt.xticks(
        x,
        df["periodo"].astype(str),
    )

    plt.title(
        "Concordância entre TF-IDF e RoBERTa por período",
        fontsize=14,
    )

    plt.xlabel("Período")
    plt.ylabel("Comentários (%)")

    plt.ylim(0, 100)

    plt.legend(
        loc="upper right",
    )

    plt.bar_label(
        barras_concordancia,
        fmt="%.2f%%",
        padding=3,
    )

    plt.bar_label(
        barras_divergencia,
        fmt="%.2f%%",
        padding=3,
    )

    salvar_grafico(
        "grafico_04_concordancia.png"
    )


# ============================================================
# GRÁFICO 5 — TIPOS DE DIVERGÊNCIA ENTRE OS MODELOS
# ============================================================

def grafico_divergencias():
    """
    Apresenta a distribuição percentual dos tipos de
    divergência entre TF-IDF e RoBERTa em cada período.

    Os percentuais são calculados sobre o total de
    divergências do respectivo período.
    """

    df = carregar_csv(
        "detalhamento_divergencias_por_periodo.csv"
    )

    if df.empty:
        raise ValueError(
            "O arquivo de divergências está vazio."
        )

    tabela = df.pivot(
        index="comparacao",
        columns="periodo",
        values="percentual",
    ).fillna(0)

    # Ordena as categorias pela sequência definida.
    ordem_divergencias = [
        "Neutro -> Contrário",
        "Contrário -> Neutro",
        "Neutro -> Favorável",
        "Favorável -> Neutro",
        "Favorável -> Contrário",
        "Contrário -> Favorável",
    ]

    categorias_existentes = [
        categoria
        for categoria in ordem_divergencias
        if categoria in tabela.index
    ]

    tabela = tabela.reindex(
        categorias_existentes
    )

    ordem_periodos = [
        "Antes da lei",
        "Depois da lei",
    ]

    tabela = tabela[
        [
            periodo
            for periodo in ordem_periodos
            if periodo in tabela.columns
        ]
    ]

    ax = tabela.plot(
        kind="barh",
        figsize=(12, 8),
    )

    ax.set_title(
        "Tipos de divergência entre TF-IDF e RoBERTa por período",
        fontsize=13,
    )

    ax.set_xlabel(
        "Percentual das divergências (%)"
    )

    ax.set_ylabel(
        "TF-IDF → RoBERTa"
    )

    ax.legend(
        title="Período",
    )

    adicionar_rotulos_barras(
        ax,
        formato="%.2f%%",
    )

    salvar_grafico(
        "grafico_05_divergencias.png"
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    """
    Executa a geração de todos os gráficos temporais.
    """

    print("=" * 65)
    print("GERAÇÃO DOS GRÁFICOS DA ANÁLISE TEMPORAL")
    print("=" * 65)

    print(
        f"\nPasta de entrada:\n{PASTA_RESULTADOS}"
    )

    print(
        f"\nPasta de saída:\n{PASTA_GRAFICOS}\n"
    )

    grafico_quantidade_por_periodo()

    grafico_distribuicao_tfidf()

    grafico_distribuicao_roberta()

    grafico_concordancia()

    grafico_divergencias()

    print("\n" + "=" * 65)
    print("TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO.")
    print("=" * 65)

    print(
        f"\nArquivos salvos em:\n{PASTA_GRAFICOS}"
    )


if __name__ == "__main__":
    main()

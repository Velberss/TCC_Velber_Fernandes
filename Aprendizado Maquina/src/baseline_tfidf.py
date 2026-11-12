"""
Baseline do TCC: TF-IDF + Regressão Logística
Base final: 1.100 comentários; validação cruzada com folds predefinidos.

Execute a partir da raiz do projeto:
    python src/baseline_tfidf.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
ARQUIVO_BASE = ROOT / "dados" / "base_tcc_final_1100.csv"
ARQUIVO_FOLDS = ROOT / "dados" / "folds_1100.csv"
PASTA_SAIDA = ROOT / "resultados" / "baseline"
ARQUIVO_METRICAS = PASTA_SAIDA / "metrics_baseline_1100.json"
ARQUIVO_PREDICOES = PASTA_SAIDA / "predicoes_baseline_1100.csv"

SEED = 42
N_FOLDS = 5
ROTULOS = ["Neutro", "Favorável", "Contrário"]
ID_COL = "youtube_comment_id"


def main():
    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(f"Base não encontrada: {ARQUIVO_BASE}")
    if not ARQUIVO_FOLDS.exists():
        raise FileNotFoundError(
            f"Arquivo de folds não encontrado: {ARQUIVO_FOLDS}\n"
            "Coloque folds_1100.csv dentro da pasta dados/."
        )

    df = pd.read_csv(ARQUIVO_BASE, encoding="utf-8-sig")
    folds = pd.read_csv(ARQUIVO_FOLDS, encoding="utf-8-sig")

    colunas_base = {ID_COL, "comment_text", "rotulo_final"}
    colunas_folds = {ID_COL, "fold"}
    if not colunas_base.issubset(df.columns):
        raise ValueError(
            f"Colunas ausentes na base: {colunas_base - set(df.columns)}. "
            f"Encontradas: {list(df.columns)}"
        )
    if not colunas_folds.issubset(folds.columns):
        raise ValueError(
            f"Colunas ausentes nos folds: {colunas_folds - set(folds.columns)}. "
            f"Encontradas: {list(folds.columns)}"
        )
    if df[ID_COL].duplicated().any() or folds[ID_COL].duplicated().any():
        raise ValueError("Há identificadores de comentário duplicados.")
    if len(df) != 1100 or len(folds) != 1100:
        raise ValueError(f"Esperados 1.100 registros na base e nos folds; encontrados {len(df)} e {len(folds)}.")
    if set(df[ID_COL]) != set(folds[ID_COL]):
        raise ValueError("Os identificadores da base e dos folds não correspondem.")
    if df[["comment_text", "rotulo_final"]].isna().any().any():
        raise ValueError("Há textos ou rótulos vazios.")
    if set(df["rotulo_final"].unique()) != set(ROTULOS):
        raise ValueError(f"Rótulos inesperados: {set(df['rotulo_final'].unique())}")
    if set(folds["fold"].unique()) != set(range(N_FOLDS)):
        raise ValueError(f"Os folds devem ser 0, 1, 2, 3 e 4; encontrados {sorted(folds['fold'].unique())}.")

    df = df.merge(folds[[ID_COL, "fold"]], on=ID_COL, how="left", validate="one_to_one")
    df["fold"] = df["fold"].astype(int)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TF-IDF + REGRESSÃO LOGÍSTICA — BASE DE 1.100 COMENTÁRIOS")
    print("=" * 65)
    print(f"Base: {ARQUIVO_BASE}")
    print(f"Folds: {ARQUIVO_FOLDS}")
    print(f"Distribuição geral:\n{df['rotulo_final'].value_counts().reindex(ROTULOS, fill_value=0)}")
    print("\nDistribuição por fold:")
    print(pd.crosstab(df["fold"], df["rotulo_final"]).reindex(columns=ROTULOS, fill_value=0))

    resultados = []
    predicoes = []

    for fold_id in range(N_FOLDS):
        treino = df[df["fold"] != fold_id]
        teste = df[df["fold"] == fold_id]
        modelo = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
            ),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        )
        # O TF-IDF é ajustado dentro de cada fold, usando apenas o treino.
        modelo.fit(treino["comment_text"].astype(str), treino["rotulo_final"])
        pred = modelo.predict(teste["comment_text"].astype(str))
        real = teste["rotulo_final"].to_numpy()

        metricas = {
            "fold": fold_id,
            "macro_f1": f1_score(real, pred, average="macro", labels=ROTULOS, zero_division=0),
            "precisao_macro": precision_score(real, pred, average="macro", labels=ROTULOS, zero_division=0),
            "acuracia": accuracy_score(real, pred),
        }
        for rotulo in ROTULOS:
            metricas[f"f1_{rotulo}"] = f1_score(real, pred, labels=[rotulo], average="macro", zero_division=0)
        resultados.append(metricas)
        predicoes.append(pd.DataFrame({
            ID_COL: teste[ID_COL].to_numpy(),
            "real": real,
            "predito": pred,
            "fold": fold_id,
        }))
        print(f"Fold {fold_id + 1}/{N_FOLDS} | Macro-F1: {metricas['macro_f1']:.4f} | Acurácia: {metricas['acuracia']:.4f}")

    res = pd.DataFrame(resultados)
    pred_all = pd.concat(predicoes, ignore_index=True)
    pred_all.to_csv(ARQUIVO_PREDICOES, index=False, encoding="utf-8-sig")

    resumo = {}
    metricas_cols = ["macro_f1", "precisao_macro", "acuracia"] + [f"f1_{r}" for r in ROTULOS]
    print("\nRESULTADO MÉDIO (média ± desvio-padrão entre folds)")
    for coluna in metricas_cols:
        media = float(res[coluna].mean())
        desvio = float(res[coluna].std(ddof=1))
        resumo[coluna] = {"media": round(media, 4), "desvio": round(desvio, 4)}
        print(f"{coluna:>20}: {media:.4f} ± {desvio:.4f}")

    cm = confusion_matrix(pred_all["real"], pred_all["predito"], labels=ROTULOS)
    matriz = pd.DataFrame(cm, index=ROTULOS, columns=ROTULOS)
    print("\nMatriz de confusão (linhas = real; colunas = predito):")
    print(matriz.to_string())

    with open(ARQUIVO_METRICAS, "w", encoding="utf-8") as f:
        json.dump({
            "modelo": "TF-IDF + Regressão Logística",
            "base": str(ARQUIVO_BASE.relative_to(ROOT)),
            "folds": str(ARQUIVO_FOLDS.relative_to(ROOT)),
            "quantidade_comentarios": len(df),
            "quantidade_folds": N_FOLDS,
            "seed": SEED,
            "resumo": resumo,
            "por_fold": res.round(4).to_dict(orient="records"),
            "matriz_confusao": cm.tolist(),
            "ordem_rotulos": ROTULOS,
        }, f, ensure_ascii=False, indent=2)

    print("\nArquivos salvos:")
    print(f"- {ARQUIVO_METRICAS}")
    print(f"- {ARQUIVO_PREDICOES}")


if __name__ == "__main__":
    main()

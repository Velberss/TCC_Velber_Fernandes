from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ["Favorável", "Contrário"]
MODEL_NAME = "pt_core_news_sm"

# Stopwords funcionais + expressões conversacionais genéricas observadas em comentários.
# Negadores são preservados intencionalmente, pois alteram o sentido do enunciado.
CUSTOM_STOPWORDS = {
    "pra", "pro", "pros", "pras", "ta", "tá", "vc", "vcs", "cê", "ce",
    "né", "ne", "kkk", "kkkk", "kkkkk", "haha", "hahaha", "rs", "blz",
    "mano", "irmao", "irmão", "cara", "galera", "pessoal", "aqui", "ali",
    "agora", "hoje", "ontem", "video", "vídeo", "youtube", "canal", "comentario",
    "comentário", "comentarios", "comentários", "link", "www", "http", "https",
}
NEGATIONS = {"não", "nao", "nunca", "nem", "sem", "jamais", "tampouco"}
# Classes gramaticais com maior conteúdo lexical para esta análise.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

# Termos removidos somente das nuvens de palavras.
# Eles continuam preservados na base processada e nas métricas linguísticas.
CLOUD_STOPWORDS = {
    "não", "nao", "bla", "blá",
    "coisa", "coisas", "tipo", "tipos",
    "alguém", "alguem", "alguéns",
}

# Expressões genéricas que não devem dominar a visualização.
# Não inclua aqui termos temáticos como "dinheiro", "ganhar" ou "perder"
# sem antes verificar sua relevância nos comentários.
CLOUD_MAX_TERM_CHARS = 32



def normalize_text(value: object) -> str:
    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w.-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def locate_input(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV de entrada não encontrado: {path}")
        return path
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "dados" / "comentarios_classificados_consolidado.csv",
        here.parent.parent / "02_Aprendizado_de_Maquina" / "resultados" / "classificacao_completa" / "comentarios_classificados_consolidado.csv",
        here.parent.parent / "Aprendizado Maquina" / "resultados" / "classificacao_completa" / "comentarios_classificados_consolidado.csv",
        here.parent.parent / "comentarios_classificados_consolidado.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Não encontrei o CSV consolidado. Informe o caminho com --input."
    )


def tokenize_with_spacy(text: str, nlp, stopwords: set[str]) -> tuple[list[str], list[str]]:
    doc = nlp(normalize_text(text))
    lemmas: list[str] = []
    content_original: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct or token.like_url or token.pos_ == "SYM":
            continue
        surface = token.text.lower().strip()
        lemma = (token.lemma_ or token.text).lower().strip()
        lemma = re.sub(r"[^a-záàâãéêíóôõúüç]", "", lemma)
        if not lemma:
            continue
        # Guarda as negações mesmo quando o modelo as classifica como palavra funcional.
        if surface in NEGATIONS or lemma in NEGATIONS:
            lemmas.append("não" if surface == "nao" or lemma == "nao" else (lemma if lemma in NEGATIONS else surface))
            content_original.append(surface)
            continue
        if token.pos_ not in CONTENT_POS:
            continue
        if token.is_stop or surface in stopwords or lemma in stopwords:
            continue
        if len(lemma) < 2:
            continue
        lemmas.append(lemma)
        content_original.append(surface)
    return lemmas, content_original


def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else float("nan")


def mattr(tokens: list[str], window: int = 10) -> float:
    if len(tokens) < window:
        return float("nan")
    vals = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(vals))


def save_csv(frame: pd.DataFrame, folder: Path, filename: str) -> None:
    frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")


def is_useful_cloud_term(term: str) -> bool:
    """Filtra termos apenas para as nuvens, sem alterar os dados analíticos."""
    parts = term.lower().split()
    if not parts or len(term) > CLOUD_MAX_TERM_CHARS:
        return False
    if any(part in CLOUD_STOPWORDS for part in parts):
        return False
    # Evita termos compostos formados quase só por palavras muito genéricas.
    if all(part in {"jogo", "pessoa", "casa", "vida"} for part in parts):
        return False
    return True


def build_exclusive_cloud_scores(
    score_by_label: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Garante que cada expressão exata apareça em no máximo uma nuvem.
    A expressão fica no grupo em que apresenta maior peso discriminativo.
    """
    exclusive = {label: {} for label in LABELS}
    all_terms = set().union(*(set(scores) for scores in score_by_label.values()))

    for term in all_terms:
        if not is_useful_cloud_term(term):
            continue
        fav_score = score_by_label.get("Favorável", {}).get(term, 0.0)
        con_score = score_by_label.get("Contrário", {}).get(term, 0.0)
        if fav_score > con_score and fav_score > 0:
            exclusive["Favorável"][term] = fav_score
        elif con_score > fav_score and con_score > 0:
            exclusive["Contrário"][term] = con_score

    return exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise linguística com spaCy para comentários Favoráveis e Contrários.")
    parser.add_argument("--input", help="Caminho do CSV consolidado.")
    parser.add_argument("--output", help="Pasta de saída. Padrão: resultados_spacy ao lado da pasta src.")
    parser.add_argument("--max-words", type=int, default=40, help="Máximo de termos em cada nuvem (padrão: 40).")
    args = parser.parse_args()

    try:
        nlp = spacy.load(MODEL_NAME, disable=["parser", "ner"])
    except OSError as exc:
        raise SystemExit(
            f"Modelo spaCy '{MODEL_NAME}' não encontrado. Instale com: python -m spacy download {MODEL_NAME}"
        ) from exc

    input_path = locate_input(args.input)
    project_root = Path(__file__).resolve().parent.parent
    output = Path(args.output).expanduser().resolve() if args.output else project_root / "resultados_spacy"
    charts = output / "graficos"
    output.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"texto", "classificacao_manual"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colunas ausentes: {required - set(df.columns)}")
    base = df[df["classificacao_manual"].isin(LABELS)].copy()
    base["texto"] = base["texto"].fillna("").astype(str)
    counts = base["classificacao_manual"].value_counts().reindex(LABELS, fill_value=0)
    if (counts == 0).any():
        raise ValueError("A base precisa conter comentários manuais dos dois posicionamentos.")

    stopwords = set(nlp.Defaults.stop_words) | CUSTOM_STOPWORDS
    stopwords -= NEGATIONS

    print("Modelo spaCy:", MODEL_NAME)
    print("Amostra manual:")
    print(counts.to_string())
    print("Processando lemas e classes gramaticais...")

    tokenized = [tokenize_with_spacy(text, nlp, stopwords)[0] for text in base["texto"]]
    base["lemas_lexicais"] = tokenized
    base["texto_lexical"] = base["lemas_lexicais"].apply(lambda x: " ".join(x))
    base["quantidade_palavras_original"] = base["texto"].apply(lambda x: len(re.findall(r"\b\w+\b", normalize_text(x))))
    base["quantidade_caracteres"] = base["texto"].str.len()
    base["quantidade_lemas_lexicais"] = base["lemas_lexicais"].apply(len)
    base["lemas_unicos"] = base["lemas_lexicais"].apply(lambda x: len(set(x)))
    base["ttr_lexical"] = base["lemas_lexicais"].apply(ttr)
    base["mattr_lexical_10"] = base["lemas_lexicais"].apply(mattr)

    summary = []
    for label in LABELS:
        group = base[base["classificacao_manual"] == label]
        all_lemmas = [lemma for seq in group["lemas_lexicais"] for lemma in seq]
        summary.append({
            "posicionamento": label,
            "quantidade_comentarios": len(group),
            "media_palavras_originais": round(group["quantidade_palavras_original"].mean(), 3),
            "media_caracteres": round(group["quantidade_caracteres"].mean(), 3),
            "media_lemas_lexicais": round(group["quantidade_lemas_lexicais"].mean(), 3),
            "media_lemas_unicos_por_comentario": round(group["lemas_unicos"].mean(), 3),
            "ttr_lexical_medio": round(group["ttr_lexical"].mean(), 4),
            "mattr_lexical_10_medio": round(group["mattr_lexical_10"].mean(), 4),
            "vocabulario_distinto_lemas": len(set(all_lemmas)),
            "comentarios_com_10_ou_mais_lemas": int(group["mattr_lexical_10"].notna().sum()),
        })
    summary_df = pd.DataFrame(summary)
    save_csv(summary_df, output, "resumo_linguistico_spacy.csv")

    # TF-IDF em lemas; unigramas e bigramas permitem destacar expressões curtas.
    vectorizer = TfidfVectorizer(
        lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.90,
        max_features=15000, sublinear_tf=True,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]{2,}\b",
    )
    matrix = vectorizer.fit_transform(base["texto_lexical"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    labels = base["classificacao_manual"].to_numpy()
    idx_fav = np.where(labels == "Favorável")[0]
    idx_con = np.where(labels == "Contrário")[0]
    mean_fav = np.asarray(matrix[idx_fav].mean(axis=0)).ravel()
    mean_con = np.asarray(matrix[idx_con].mean(axis=0)).ravel()

    # Peso discriminativo: média TF-IDF no grupo menos média no grupo oposto.
    # Evita dar destaque automático a termos frequentes nos dois posicionamentos.
    term_rows = []
    score_by_label: dict[str, dict[str, float]] = {}
    for label, own, other in [("Favorável", mean_fav, mean_con), ("Contrário", mean_con, mean_fav)]:
        scores = np.maximum(own - other, 0)
        order = np.argsort(scores)[::-1]
        score_by_label[label] = {}
        for pos in order:
            if scores[pos] <= 0:
                continue
            term = feature_names[pos]
            score = float(scores[pos])
            score_by_label[label][term] = score
            term_rows.append({
                "posicionamento": label,
                "termo_ou_expressao": term,
                "tfidf_medio_grupo": round(float(own[pos]), 6),
                "tfidf_medio_grupo_oposto": round(float(other[pos]), 6),
                "diferenca_discriminativa": round(score, 6),
            })
    terms_df = pd.DataFrame(term_rows)
    save_csv(terms_df, output, "termos_discriminativos_spacy.csv")

    # Similaridade lexical em lemas processados pelo spaCy.
    similarities = cosine_similarity(matrix, dense_output=True)
    def within_pairs(indices: np.ndarray) -> np.ndarray:
        sub = similarities[np.ix_(indices, indices)]
        return sub[np.triu_indices_from(sub, k=1)]
    pairs_fav = within_pairs(idx_fav)
    pairs_con = within_pairs(idx_con)
    pairs_cross = similarities[np.ix_(idx_fav, idx_con)].ravel()
    sim_rows = []
    for name, values in [("Favorável x Favorável", pairs_fav), ("Contrário x Contrário", pairs_con), ("Favorável x Contrário", pairs_cross)]:
        sim_rows.append({
            "comparacao": name, "quantidade_pares": len(values),
            "similaridade_media": round(float(np.mean(values)), 6),
            "similaridade_mediana": round(float(np.median(values)), 6),
            "percentil_90": round(float(np.percentile(values, 90)), 6),
            "pares_com_similaridade_ge_080_pct": round(float(np.mean(values >= .80) * 100), 3),
        })
    save_csv(pd.DataFrame(sim_rows), output, "similaridade_lexical_spacy.csv")

    # Nuvens: filtragem exclusiva para visualização.
    # Negadores como "não" continuam nas métricas e na base processada,
    # mas são retirados das nuvens por não funcionarem bem como termos isolados.
    # Cada expressão exata é atribuída somente ao grupo em que é mais discriminativa.
    cloud_scores = build_exclusive_cloud_scores(score_by_label)
    cloud_term_rows = []

    for label, filename in [
        ("Favorável", "nuvem_spacy_favoravel.png"),
        ("Contrário", "nuvem_spacy_contrario.png"),
    ]:
        scores = cloud_scores[label]
        top_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)[:args.max_words]
        )

        for term, score in top_scores.items():
            cloud_term_rows.append({
                "posicionamento": label,
                "termo_ou_expressao": term,
                "peso_discriminativo": round(float(score), 6),
            })

        if not top_scores:
            print(f"Aviso: nenhum termo passou pelos filtros da nuvem para {label}.")
            continue

        cloud = WordCloud(
            width=2000, height=1200, background_color="white", colormap="viridis",
            max_words=args.max_words, collocations=False, prefer_horizontal=0.92,
            min_font_size=12, relative_scaling=0.45, random_state=42,
        ).generate_from_frequencies(top_scores)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Termos mais característicos — {label}", fontsize=20, pad=18)
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    save_csv(pd.DataFrame(cloud_term_rows), output, "termos_nuvens_filtrados_spacy.csv")

    # Gráficos de extensão e diversidade.
    for column, title, ylabel, filename in [
        ("media_palavras_originais", "Extensão média dos comentários", "Palavras por comentário", "grafico_extensao_palavras_spacy.png"),
        ("media_caracteres", "Extensão média em caracteres", "Caracteres por comentário", "grafico_extensao_caracteres_spacy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(summary_df["posicionamento"], summary_df[column])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, summary_df[column]):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center")
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Exporta a base processada para auditoria e reprodutibilidade.
    export_cols = [c for c in ["comentario_id", "video_id", "texto", "classificacao_manual",
                               "quantidade_palavras_original", "quantidade_caracteres",
                               "quantidade_lemas_lexicais", "lemas_unicos", "ttr_lexical",
                               "mattr_lexical_10", "texto_lexical"] if c in base.columns]
    audit = base[export_cols].copy()
    audit["texto_lexical"] = audit["texto_lexical"].apply(lambda x: x.split())
    audit["texto_lexical"] = audit["texto_lexical"].apply(lambda x: " ".join(x))
    save_csv(audit, output, "base_processada_spacy.csv")

    print("\nRESUMO LINGUÍSTICO COM spaCy")
    print(summary_df.to_string(index=False))
    print("\nTERMOS DISCRIMINATIVOS (10 primeiros por grupo)")
    print(terms_df.groupby("posicionamento").head(10).to_string(index=False))
    print("\nSIMILARIDADE LEXICAL")
    print(pd.DataFrame(sim_rows).to_string(index=False))
    print(f"\nCSV de entrada: {input_path}")
    print(f"Resultados: {output}")
    print(f"Gráficos: {charts}")
    print("Observação: resultados descritivos da amostra manual; não representam automaticamente todo o corpus.")


if __name__ == "__main__":
    main()

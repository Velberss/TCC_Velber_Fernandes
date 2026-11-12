from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import Dataset


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

NOME_BASE = "base_tcc_final_1100.csv"

CANDIDATOS_BASE = [
    RAIZ_PROJETO / NOME_BASE,
    RAIZ_PROJETO / "data" / NOME_BASE,
    RAIZ_PROJETO / "dados" / NOME_BASE,
    RAIZ_PROJETO / "dataset" / NOME_BASE,
]

PASTA_SAIDA = RAIZ_PROJETO / "resultados" / "modelos_finais"

MODELO_ROBERTA = "eduagarcia/RoBERTaLexPT-base"

COLUNA_TEXTO = "comment_text"
COLUNA_ROTULO = "rotulo_final"

ROTULOS = ["Contrário", "Favorável", "Neutro"]
ROTULO2ID = {rotulo: i for i, rotulo in enumerate(ROTULOS)}
ID2ROTULO = {i: rotulo for rotulo, i in ROTULO2ID.items()}

SEED = 42
EPOCAS = 6
TAMANHO_LOTE = 1
ACUMULACAO_GRADIENTE = 4
COMPRIMENTO_MAXIMO = 128


# ============================================================
# LOCALIZAÇÃO E VALIDAÇÃO DA BASE
# ============================================================

def localizar_base():
    for caminho in CANDIDATOS_BASE:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        f"Não encontrei '{NOME_BASE}'.\n"
        "Coloque o CSV na raiz do projeto ou nas pastas "
        "'data', 'dados' ou 'dataset'."
    )


def carregar_base():
    caminho = localizar_base()

    # O arquivo deve estar em UTF-8.
    df = pd.read_csv(caminho, encoding="utf-8-sig")

    colunas_necessarias = {COLUNA_TEXTO, COLUNA_ROTULO}
    faltantes = colunas_necessarias - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Colunas ausentes na base: {sorted(faltantes)}"
        )

    df = df[[COLUNA_TEXTO, COLUNA_ROTULO]].copy()

    df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str).str.strip()
    df[COLUNA_ROTULO] = df[COLUNA_ROTULO].astype(str).str.strip()

    df = df[
        (df[COLUNA_TEXTO] != "")
        & (df[COLUNA_ROTULO].isin(ROTULOS))
    ].copy()

    if df.empty:
        raise ValueError("A base ficou vazia após a validação.")

    distribuicao = df[COLUNA_ROTULO].value_counts()

    if set(distribuicao.index) != set(ROTULOS):
        raise ValueError(
            "A base precisa conter as três classes: "
            "Contrário, Favorável e Neutro."
        )

    print(f"Base utilizada: {caminho}")
    print(f"Total de comentários: {len(df)}")
    print("\nDistribuição dos rótulos:")
    print(distribuicao.reindex(ROTULOS))

    return df


# ============================================================
# MODELO 1 — TF-IDF + REGRESSÃO LOGÍSTICA
# ============================================================

def treinar_tfidf(df):
    print("\n" + "=" * 65)
    print("TREINAMENTO DO MODELO TF-IDF + REGRESSÃO LOGÍSTICA")
    print("=" * 65)

    modelo = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
            ),
        ),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        ),
    ])

    modelo.fit(
        df[COLUNA_TEXTO],
        df[COLUNA_ROTULO],
    )

    pasta_modelo = PASTA_SAIDA / "tfidf"
    pasta_modelo.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_modelo / "modelo_tfidf.joblib"
    joblib.dump(modelo, caminho_modelo)

    print("Modelo TF-IDF treinado com sucesso.")
    print(f"Modelo salvo em: {caminho_modelo}")


# ============================================================
# DATASET PARA O ROBERTA
# ============================================================

class DatasetComentarios(Dataset):
    def __init__(self, textos, rotulos, tokenizer):
        self.textos = list(textos)
        self.rotulos = [
            ROTULO2ID[rotulo] for rotulo in rotulos
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, indice):
        item = self.tokenizer(
            self.textos[indice],
            truncation=True,
            max_length=COMPRIMENTO_MAXIMO,
        )

        item["labels"] = self.rotulos[indice]

        return item


# ============================================================
# TRAINER COM PESOS POR CLASSE
# ============================================================

class TrainerComPeso(Trainer):
    def __init__(self, *args, pesos=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos = pesos

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = inputs.copy()
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        funcao_perda = nn.CrossEntropyLoss(
            weight=self.pesos.to(outputs.logits.device)
        )

        loss = funcao_perda(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# MODELO 2 — ROBERTALexPT
# ============================================================

def treinar_roberta(df):
    print("\n" + "=" * 65)
    print("TREINAMENTO DO MODELO ROBERTALexPT")
    print("=" * 65)

    set_seed(SEED)

    textos = df[COLUNA_TEXTO].tolist()
    rotulos = df[COLUNA_ROTULO].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODELO_ROBERTA)

    dataset = DatasetComentarios(
        textos=textos,
        rotulos=rotulos,
        tokenizer=tokenizer,
    )

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_ROBERTA,
        num_labels=len(ROTULOS),
        label2id=ROTULO2ID,
        id2label=ID2ROTULO,
    )

    # Pesos calculados a partir da distribuição da base.
    rotulos_numericos = [ROTULO2ID[r] for r in rotulos]

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(ROTULOS)),
        y=rotulos_numericos,
    )

    pesos = torch.tensor(pesos, dtype=torch.float)

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU não detectada. O treinamento será feito na CPU.")

    pasta_modelo = PASTA_SAIDA / "roberta"
    pasta_temporaria = PASTA_SAIDA / "temporario_roberta"

    pasta_modelo.mkdir(parents=True, exist_ok=True)
    pasta_temporaria.mkdir(parents=True, exist_ok=True)

    argumentos = TrainingArguments(
        output_dir=str(pasta_temporaria),
        num_train_epochs=EPOCAS,
        per_device_train_batch_size=TAMANHO_LOTE,
        gradient_accumulation_steps=ACUMULACAO_GRADIENTE,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = TrainerComPeso(
        model=modelo,
        args=argumentos,
        train_dataset=dataset,
        data_collator=collator,
        pesos=pesos,
    )

    print(f"Quantidade de exemplos de treinamento: {len(dataset)}")
    print(f"Épocas: {EPOCAS}")
    print(f"Comprimento máximo dos textos: {COMPRIMENTO_MAXIMO}")

    trainer.train()

    # Salva o modelo e o tokenizador para a classificação posterior.
    trainer.save_model(str(pasta_modelo))
    tokenizer.save_pretrained(str(pasta_modelo))

    print("Modelo RoBERTa treinado com sucesso.")
    print(f"Modelo e tokenizador salvos em: {pasta_modelo}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    set_seed(SEED)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = carregar_base()

    print("\nOs modelos serão treinados usando toda a base validada.")
    print("Os resultados anteriores de validação cruzada serão mantidos.")

    treinar_tfidf(df)
    treinar_roberta(df)

    print("\n" + "=" * 65)
    print("TREINAMENTO FINALIZADO")
    print("=" * 65)
    print(f"Arquivos salvos em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()

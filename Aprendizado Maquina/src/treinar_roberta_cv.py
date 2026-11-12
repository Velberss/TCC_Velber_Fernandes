"""
Fine-tuning do RoBERTa em português com folds predefinidos (base final de 1.100 comentários).

Execute a partir da raiz do projeto:
    python src/treinar_roberta_cv.py

O teste rápido executa somente o fold 0 por uma época e não salva os arquivos finais.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
ARQUIVO_BASE = ROOT / "dados" / "base_tcc_final_1100.csv"
ARQUIVO_FOLDS = ROOT / "dados" / "folds_1100.csv"
PASTA_SAIDA = ROOT / "resultados" / "roberta"
PASTA_TEMP = ROOT / "resultados" / "roberta" / "temporarios"

ID_COL = "youtube_comment_id"
ROTULOS = ["Neutro", "Favorável", "Contrário"]
ROTULO2ID = {rotulo: i for i, rotulo in enumerate(ROTULOS)}
ID2ROTULO = {i: rotulo for rotulo, i in ROTULO2ID.items()}
SEED = 42
N_FOLDS = 5

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="eduagarcia/RoBERTaLexPT-base")
parser.add_argument("--epocas", type=int, default=6)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch", type=int, default=1, help="Lote pequeno para limitar uso de VRAM.")
parser.add_argument("--acumulacao", type=int, default=4, help="Acumulação de gradiente.")
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--teste_rapido", action="store_true", help="Executa somente o fold 0 por uma época.")
args = parser.parse_args()

ARQUIVO_METRICAS = PASTA_SAIDA / "metrics_roberta_cv_1100.json"
ARQUIVO_PREDICOES = PASTA_SAIDA / "predicoes_roberta_cv_1100.csv"


def validar_dados():
    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(f"Base não encontrada: {ARQUIVO_BASE}")
    if not ARQUIVO_FOLDS.exists():
        raise FileNotFoundError(
            f"Folds não encontrados: {ARQUIVO_FOLDS}. Coloque folds_1100.csv em dados/."
        )

    df = pd.read_csv(ARQUIVO_BASE, encoding="utf-8-sig")
    folds = pd.read_csv(ARQUIVO_FOLDS, encoding="utf-8-sig")
    colunas_base = {ID_COL, "comment_text", "rotulo_final"}
    colunas_folds = {ID_COL, "fold"}

    if not colunas_base.issubset(df.columns):
        raise ValueError(f"Colunas ausentes na base: {colunas_base - set(df.columns)}")
    if not colunas_folds.issubset(folds.columns):
        raise ValueError(f"Colunas ausentes nos folds: {colunas_folds - set(folds.columns)}")
    if df[ID_COL].duplicated().any() or folds[ID_COL].duplicated().any():
        raise ValueError("Há identificadores de comentário duplicados.")
    if len(df) != 1100 or len(folds) != 1100:
        raise ValueError(f"Esperados 1.100 registros; base={len(df)}, folds={len(folds)}.")
    if set(df[ID_COL]) != set(folds[ID_COL]):
        raise ValueError("Os identificadores da base e dos folds não correspondem.")
    if df[["comment_text", "rotulo_final"]].isna().any().any():
        raise ValueError("Há comentários sem texto ou rótulo.")
    if set(df["rotulo_final"].unique()) != set(ROTULOS):
        raise ValueError(f"Rótulos inesperados: {set(df['rotulo_final'].unique())}")
    if set(folds["fold"].unique()) != set(range(N_FOLDS)):
        raise ValueError(f"Os folds devem ser 0 a 4; encontrados {sorted(folds['fold'].unique())}.")

    df = df.merge(folds[[ID_COL, "fold"]], on=ID_COL, how="left", validate="one_to_one")
    df["fold"] = df["fold"].astype(int)
    df["label"] = df["rotulo_final"].map(ROTULO2ID)
    df["comment_text"] = df["comment_text"].astype(str)
    return df


class Comentarios(torch.utils.data.Dataset):
    def __init__(self, textos, labels, tokenizer, max_len):
        self.enc = tokenizer(list(textos), truncation=True, max_length=max_len)
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {chave: valores[i] for chave, valores in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class TrainerComPeso(Trainer):
    def __init__(self, *args, pesos=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos = pesos

    # Assinatura compatível com versões recentes do Transformers.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.pesos.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    PASTA_TEMP.mkdir(parents=True, exist_ok=True)
    df = validar_dados()

    print("=" * 68)
    print("ROBERTA — BASE FINAL DE 1.100 COMENTÁRIOS")
    print("=" * 68)
    print(f"Base: {ARQUIVO_BASE}")
    print(f"Folds: {ARQUIVO_FOLDS}")
    print(f"Dispositivo: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Distribuição dos rótulos:\n{df['rotulo_final'].value_counts().reindex(ROTULOS, fill_value=0)}")
    print("\nDistribuição por fold:")
    print(pd.crosstab(df["fold"], df["rotulo_final"]).reindex(columns=ROTULOS, fill_value=0))

    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    folds_para_rodar = [0] if args.teste_rapido else list(range(N_FOLDS))
    epocas = 1 if args.teste_rapido else args.epocas
    resultados, predicoes = [], []

    for fold_id in folds_para_rodar:
        print(f"\n===== Fold {fold_id + 1}/{N_FOLDS} =====")
        set_seed(SEED + fold_id)
        treino = df[df["fold"] != fold_id].copy()
        teste = df[df["fold"] == fold_id].copy()
        if len(treino) != 880 or len(teste) != 220:
            raise ValueError(f"Tamanhos inesperados no fold {fold_id}: treino={len(treino)}, validação={len(teste)}.")

        pesos = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(ROTULOS)),
            y=treino["label"].to_numpy(),
        )
        print("Pesos das classes:", {ID2ROTULO[i]: round(float(pesos[i]), 4) for i in range(len(ROTULOS))})

        model = AutoModelForSequenceClassification.from_pretrained(
            args.modelo,
            num_labels=len(ROTULOS),
            id2label=ID2ROTULO,
            label2id=ROTULO2ID,
        )
        # Reduz o consumo de memória; batch padrão 1 e max_len 128.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir=str(PASTA_TEMP / f"fold_{fold_id}"),
            num_train_epochs=epocas,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.acumulacao,
            weight_decay=0.01,
            warmup_steps=0,
            lr_scheduler_type="linear",
            save_strategy="no",
            report_to="none",
            logging_steps=25,
            seed=SEED + fold_id,
            fp16=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = TrainerComPeso(
            model=model,
            args=training_args,
            train_dataset=Comentarios(treino["comment_text"], treino["label"], tokenizer, args.max_len),
            data_collator=collator,
            pesos=torch.tensor(pesos, dtype=torch.float),
        )
        trainer.train()
        saida = trainer.predict(
            Comentarios(teste["comment_text"], teste["label"], tokenizer, args.max_len)
        )
        pred = saida.predictions.argmax(axis=-1)
        real = teste["label"].to_numpy()
        metricas = {
            "fold": fold_id,
            "macro_f1": f1_score(real, pred, average="macro", labels=list(range(len(ROTULOS))), zero_division=0),
            "precisao_macro": precision_score(real, pred, average="macro", labels=list(range(len(ROTULOS))), zero_division=0),
            "acuracia": accuracy_score(real, pred),
        }
        for i, rotulo in enumerate(ROTULOS):
            metricas[f"f1_{rotulo}"] = f1_score(real, pred, labels=[i], average="macro", zero_division=0)
        resultados.append(metricas)
        predicoes.append(pd.DataFrame({
            ID_COL: teste[ID_COL].to_numpy(),
            "real": [ID2ROTULO[int(i)] for i in real],
            "predito": [ID2ROTULO[int(i)] for i in pred],
            "fold": fold_id,
        }))
        print(f"Fold concluído | Macro-F1: {metricas['macro_f1']:.4f} | Acurácia: {metricas['acuracia']:.4f}")

        del trainer, model, saida
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = pd.DataFrame(resultados)
    pred_all = pd.concat(predicoes, ignore_index=True)
    cm = confusion_matrix(pred_all["real"], pred_all["predito"], labels=ROTULOS)

    print("\nMÉTRICAS")
    resumo = {}
    for coluna in [c for c in res.columns if c != "fold"]:
        media = float(res[coluna].mean())
        desvio = float(res[coluna].std(ddof=1)) if len(res) > 1 else float("nan")
        resumo[coluna] = {
            "media": round(media, 4),
            "desvio": None if np.isnan(desvio) else round(desvio, 4),
        }
        print(f"{coluna:>20}: {media:.4f}" + (f" ± {desvio:.4f}" if not np.isnan(desvio) else " (um fold; sem desvio)"))

    print("\nMatriz de confusão (linhas = real; colunas = predito):")
    print(pd.DataFrame(cm, index=ROTULOS, columns=ROTULOS).to_string())

    if not args.teste_rapido:
        pred_all.to_csv(ARQUIVO_PREDICOES, index=False, encoding="utf-8-sig")
        with open(ARQUIVO_METRICAS, "w", encoding="utf-8") as f:
            json.dump({
                "modelo": args.modelo,
                "base": str(ARQUIVO_BASE.relative_to(ROOT)),
                "arquivo_folds": str(ARQUIVO_FOLDS.relative_to(ROOT)),
                "quantidade_comentarios": len(df),
                "quantidade_folds": N_FOLDS,
                "epocas": args.epocas,
                "learning_rate": args.lr,
                "batch": args.batch,
                "acumulacao": args.acumulacao,
                "max_len": args.max_len,
                "seed": SEED,
                "resumo": resumo,
                "por_fold": res.round(4).to_dict(orient="records"),
                "matriz_confusao": cm.tolist(),
                "ordem_rotulos": ROTULOS,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nArquivos salvos:\n- {ARQUIVO_METRICAS}\n- {ARQUIVO_PREDICOES}")
    else:
        print("\nTeste rápido concluído; arquivos finais não foram sobrescritos.")


if __name__ == "__main__":
    main()

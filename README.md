# TCC — Classificação e Caracterização em Comentários sobre Apostas Online: O Caso do Jogo Aviator no YouTube

Este repositório reúne o código desenvolvido para o Trabalho de Conclusão de Curso (TCC), com foco na coleta, limpeza, análise linguística e classificação automatizada de comentários em português sobre o jogo Aviator, extraídos do YouTube.

A proposta do projeto é investigar como os comentários se distribuem entre posições favoráveis, contrárias e neutras, além de comparar o discurso antes e depois da sanção da Lei 14.790/2023.

## Objetivo do projeto

- Coletar comentários sobre o tema "Aviator" e jogos de apostas no YouTube;
- Filtrar e limpar comentários ruidosos, spam e textos fora do tema;
- Analisar padrões linguísticos e vocabulário por posicionamento ideológico;
- Treinar modelos de classificação para identificar opiniões em comentários;
- Comparar a evolução do discurso antes e depois da lei.

## Estrutura do repositório

```text
Codigo TCC/
├── README.md
├── Analise de Padrões Linguisticos/
│   └── src/
│       └── analise_padroes_linguisticos_spacy.py
├── Aprendizado Maquina/
│   ├── requirements.txt
│   ├── src/
│   │   ├── analisar_periodos_lei.py
│   │   ├── baseline_tfidf.py
│   │   ├── classificar_comentarios_restantes.py
│   │   ├── gerar_graficos_temporais.py
│   │   ├── treinar_modelos_finais.py
│   │   ├── treinar_roberta_cv.py
│   │   └── ...
├── Coleta e Processamento/
│   ├── main.py
│   ├── limpeza_dados.py
│   └── Dados/
└── ...
```

## Fluxo do projeto

### 1) Coleta dos dados
Arquivo principal:
- `Coleta e Processamento/main.py`

Esse script:
- acessa a API do YouTube;
- busca vídeos por palavras-chave relacionadas ao Aviator;
- coleta comentários em português;
- salva os dados em arquivos CSV dentro da pasta `Dados/`.

Antes de executar, defina a variável de ambiente:

```bash
export YOUTUBE_API_KEY="sua_chave_aqui"
```

No Windows PowerShell:

```powershell
$env:YOUTUBE_API_KEY="sua_chave_aqui"
```

### 2) Limpeza e filtro dos dados
Arquivo principal:
- `Coleta e Processamento/limpeza_dados.py`

Esse script faz a limpeza da base bruta, removendo:
- duplicatas;
- vídeos fora do tema;
- comentários de spam ou propagandas;
- vídeos com poucos comentários;
- dados fora do período de análise.

### 3) Análise linguística
Arquivo principal:
- `Analise de Padrões Linguisticos/src/analise_padroes_linguisticos_spacy.py`

Esse script:
- carrega o modelo spaCy (`pt_core_news_sm`);
- normaliza e lematiza os comentários;
- analisa vocabulário, TF-IDF e padrões de linguagem por classe (Favorável/Contrário);
- gera nuvens de palavras e métricas linguísticas.

### 4) Modelos de classificação
Pasta:
- `Aprendizado Maquina/src/`

Scripts relevantes:
- `baseline_tfidf.py` — baseline com TF-IDF + Regressão Logística;
- `treinar_roberta_cv.py` — treinamento com validação cruzada do modelo RoBERTa;
- `treinar_modelos_finais.py` — treinamento do modelo final e salvamento dos artefatos;
- `classificar_comentarios_restantes.py` — classificação automatizada dos comentários restantes;
- `gerar_graficos_temporais.py` — geração de gráficos para comparação de períodos;
- `analisar_periodos_lei.py` — análise temporal antes e depois da Lei 14.790/2023.

## Requisitos

O projeto depende de bibliotecas Python para processamento de texto, aprendizado de máquina e coleta de dados.

### Dependências principais

```bash
pip install -r "Aprendizado Maquina/requirements.txt"
```

Dependências extras, caso ainda não estejam instaladas:

```bash
pip install google-api-python-client langdetect spacy wordcloud
python -m spacy download pt_core_news_sm
```

## Ambiente recomendado

- Python 3.10+
- Ambiente virtual (`venv` ou `conda`) recomendado
- GPU opcional para acelerar os treinos do RoBERTa

## Como executar

### Etapa 1: coletar os dados
```bash
cd "Coleta e Processamento"
python main.py
```

### Etapa 2: limpar a base
```bash
cd "Coleta e Processamento"
python limpeza_dados.py
```

### Etapa 3: executar a análise linguística
```bash
cd "Analise de Padrões Linguisticos/src"
python analise_padroes_linguisticos_spacy.py --input <caminho_do_csv>
```

### Etapa 4: treinar os modelos
```bash
cd "Aprendizado Maquina/src"
python baseline_tfidf.py
python treinar_modelos_finais.py
```

## Saídas esperadas

O projeto gera arquivos em pastas como:

- `Dados/` — dados brutos e processados
- `resultados/` — métricas, predições, gráficos e comparações temporais
- `resultados_spacy/` — análise linguística e nuvens de palavras
- `resultados/modelos_finais/` — modelos treinados e artefatos salvos

## Observações importantes

- O código foi estruturado em módulos, separando coleta, limpeza, análise textual e aprendizado de máquina.
- Alguns scripts dependem de arquivos CSV de entrada específicos, gerados em etapas anteriores do pipeline.
- O projeto foi desenvolvido em português e é orientado a análise de discursos em mídia social.
- A coleta de dados exige uma chave válida da API do YouTube.

## Próximos passos sugeridos

- automatizar o pipeline completo em um único script executável;
- documentar melhor os arquivos de entrada e saída de cada etapa;
- adicionar testes unitários para validação das funções de limpeza;
- criar uma interface de visualização dos resultados.

## Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do TCC. Verifique com a instituição ou orientador a política de uso e compartilhamento do código e dos dados.

## Contato

Se necessário, ajuste este README conforme a estrutura final do seu TCC e os nomes exatos dos arquivos gerados no seu ambiente local.

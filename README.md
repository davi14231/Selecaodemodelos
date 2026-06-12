# Seleção de Modelos - IRIS e SRAG Pediátrico

Este repositório contém a atividade de seleção de modelos. Ele possui duas partes:

- seleção de modelos no conjunto clássico **IRIS**;
- seleção de modelos no conjunto de dados do grupo, **SRAG pediátrico**.

Foram avaliados os seguintes modelos:

- K-NN
- Naive Bayes
- Regressão Logística
- Árvore de Decisão
- Rede Neural MLP
- SVM

A seleção de hiperparâmetros foi feita com `RandomizedSearchCV`, usando validação cruzada com 5 folds e acurácia como métrica principal. Depois da busca, cada modelo foi avaliado em um conjunto de teste separado com 20% dos dados.

## Estrutura

```text
iris-model-selection/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── main.py
│   └── srag_model_selection.py
├── results/
│   ├── resultados_modelos.csv
│   └── fronteira_decisao.png
└── docs/
    └── relatorio_selecao_modelos_iris_docs.docx
```

## Como executar

Crie um ambiente virtual, instale as dependências e rode o script desejado:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux ou Mac, a ativação do ambiente virtual fica assim:

```bash
source .venv/bin/activate
```

## Executar o IRIS

```bash
python src/main.py
```

Saídas geradas:

- `results/resultados_modelos.csv`
- `results/fronteira_decisao.png`

## Executar o SRAG pediátrico

Para rodar a parte do conjunto de dados do grupo, coloque o arquivo `INFLUD19-23-03-2026.csv` dentro da pasta `data/`.

Depois execute:

```bash
python src/srag_model_selection.py
```

Também é possível passar o caminho do CSV manualmente:

```bash
python src/srag_model_selection.py --data "C:\caminho\para\INFLUD19-23-03-2026.csv"
```

Saídas geradas:

- `results/srag_resultados_modelos.csv`
- `results/srag_matriz_confusao_melhor_modelo.png`

## Variável alvo do SRAG

No conjunto SRAG, foi criada e usada a coluna `UTI_BIN` como variável alvo, a partir da coluna original `UTI`:

- `0`: paciente não foi para UTI;
- `1`: paciente foi para UTI.

A coluna `UTI` original não foi usada como atributo de entrada, pois ela é diretamente relacionada ao alvo e causaria vazamento de informação.

Também foram evitadas colunas claramente posteriores ao desfecho, como datas de entrada ou saída da UTI e evolução final.

O script também aplica os filtros usados na base do grupo:

- idade até 12 anos;
- notificação em Pernambuco (`SG_UF_NOT = PE`);
- remoção de registros com UTI desconhecida ou ignorada.

## Resultados obtidos no IRIS

| Modelo | Acurácia CV | Acurácia Teste | Precisão Teste | Recall Teste |
|---|---:|---:|---:|---:|
| Regressão Logística | 0.9667 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 0.9667 | 0.9667 | 0.9697 | 0.9667 |
| K-NN | 0.9583 | 0.9667 | 0.9697 | 0.9667 |
| Naive Bayes | 0.9583 | 0.9667 | 0.9697 | 0.9667 |
| Árvore de Decisão | 0.9417 | 0.9333 | 0.9333 | 0.9333 |
| Rede Neural | 0.9167 | 0.8333 | 0.8498 | 0.8333 |

O melhor modelo no IRIS foi a **Regressão Logística**, que obteve 100% de acurácia, precisão e recall no conjunto de teste.

## Observação sobre o CSV do SRAG

O CSV bruto não foi incluído no Git por padrão porque é um conjunto de dados de saúde. Caso o repositório seja privado e o professor permita, o arquivo pode ser colocado manualmente em `data/`.

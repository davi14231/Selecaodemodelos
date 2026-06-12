import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon, loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_ITER = 25
CV = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "INFLUD19-23-03-2026.csv"

TARGET = "UTI_BIN"
PEDIATRIC_MAX_AGE = 12
FILTER_NOTIFICATION_UF = "PE"

NUMERIC_FEATURES = [
    "IDADE_ANOS",
]

CATEGORICAL_FEATURES_BASE = [
    "CS_SEXO",
    "REGIAO",
    "CS_RACA",
    "CS_ZONA",
    "NOSOCOMIAL",
    "AVE_SUINO",
    "FEBRE",
    "TOSSE",
    "GARGANTA",
    "DISPNEIA",
    "DESC_RESP",
    "SATURACAO",
    "DIARREIA",
    "VOMITO",
    "FATOR_RISC",
    "CARDIOPATI",
    "HEMATOLOGI",
    "SIND_DOWN",
    "HEPATICA",
    "ASMA",
    "DIABETES",
    "NEUROLOGIC",
    "PNEUMOPATI",
    "IMUNODEPRE",
    "RENAL",
    "OBESIDADE",
    "OUT_MORBI",
    "VACINA",
    "ANTIVIRAL",
    "RAIOX_RES",
    "PCR_RESUL",
    "RES_AN",
]

RMR_MUNICIPALITY_CODES = {
    260105,  # Araçoiaba
    260290,  # Cabo de Santo Agostinho
    260345,  # Camaragibe
    260680,  # Igarassu
    260720,  # Ipojuca
    260760,  # Ilha de Itamaracá
    260775,  # Itapissuma
    260790,  # Jaboatão dos Guararapes
    260960,  # Olinda
    261070,  # Paulista
    261160,  # Recife
    261370,  # São Lourenço da Mata
}


def criar_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def carregar_dados(caminho_csv):
    caminho = Path(caminho_csv)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            f"Coloque o CSV em {DEFAULT_RAW_DATA_PATH} ou passe o caminho com --data."
        )

    return pd.read_csv(caminho, sep=None, engine="python")


def adicionar_idade_anos(df):
    if "IDADE_ANOS" in df.columns:
        return df

    if "NU_IDADE_N" not in df.columns or "TP_IDADE" not in df.columns:
        raise ValueError("Não foi possível criar IDADE_ANOS. Colunas NU_IDADE_N e TP_IDADE não encontradas.")

    df = df.copy()
    df["IDADE_ANOS"] = np.nan
    df.loc[df["TP_IDADE"] == 1, "IDADE_ANOS"] = df.loc[df["TP_IDADE"] == 1, "NU_IDADE_N"] / 365
    df.loc[df["TP_IDADE"] == 2, "IDADE_ANOS"] = df.loc[df["TP_IDADE"] == 2, "NU_IDADE_N"] / 12
    df.loc[df["TP_IDADE"] == 3, "IDADE_ANOS"] = df.loc[df["TP_IDADE"] == 3, "NU_IDADE_N"]

    return df


def adicionar_uti_bin(df):
    if TARGET in df.columns:
        return df

    if "UTI" not in df.columns:
        raise ValueError("Não foi possível criar UTI_BIN. Coluna UTI não encontrada.")

    df = df.copy()
    df[TARGET] = np.nan
    df.loc[df["UTI"] == 1, TARGET] = 1
    df.loc[df["UTI"] == 2, TARGET] = 0

    return df


def adicionar_regiao(df):
    if "REGIAO" in df.columns:
        return df

    df = df.copy()

    if "CO_MUN_RES" in df.columns:
        municipio = pd.to_numeric(df["CO_MUN_RES"], errors="coerce")
        df["REGIAO"] = np.where(municipio.isin(RMR_MUNICIPALITY_CODES), "RMR", "Interior de PE")
        if "SG_UF" in df.columns:
            df.loc[df["SG_UF"].notna() & (df["SG_UF"] != "PE"), "REGIAO"] = "Fora de PE"
    else:
        df["REGIAO"] = "Não informado"

    return df


def filtrar_base_de_estudo(df):
    dados = adicionar_idade_anos(df)
    dados = adicionar_uti_bin(dados)
    dados = adicionar_regiao(dados)

    dados = dados[dados["IDADE_ANOS"] <= PEDIATRIC_MAX_AGE].copy()

    if "SG_UF_NOT" in dados.columns:
        dados = dados[dados["SG_UF_NOT"] == FILTER_NOTIFICATION_UF].copy()

    dados = dados[dados[TARGET].isin([0, 1])].copy()

    return dados


def preparar_dados(df):
    dados = filtrar_base_de_estudo(df)

    categorical_features = [coluna for coluna in CATEGORICAL_FEATURES_BASE if coluna in dados.columns]
    numeric_features = [coluna for coluna in NUMERIC_FEATURES if coluna in dados.columns]
    colunas = numeric_features + categorical_features + [TARGET]
    colunas_faltando = [coluna for coluna in [TARGET] if coluna not in dados.columns]

    if colunas_faltando:
        raise ValueError(f"Colunas não encontradas no CSV: {colunas_faltando}")

    dados = dados[colunas].copy()

    # No SIVEP/SRAG, o código 9 costuma representar ignorado ou sem informação.
    dados[categorical_features] = dados[categorical_features].replace(9, np.nan)
    dados = dados.dropna(subset=[TARGET])

    X = dados[numeric_features + categorical_features]
    y = dados[TARGET].astype(int)

    return X, y, numeric_features, categorical_features


def criar_preprocessador(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot", criar_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def criar_pipeline(modelo, numeric_features, categorical_features):
    return Pipeline([
        ("preprocessador", criar_preprocessador(numeric_features, categorical_features)),
        ("modelo", modelo),
    ])


def limpar_parametros(valor):
    if isinstance(valor, dict):
        return {chave: limpar_parametros(item) for chave, item in valor.items()}
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def avaliar_modelo(nome, modelo, parametros, X_treino, X_teste, y_treino, y_teste, n_iter=N_ITER):
    if parametros is None:
        scores_cv = cross_val_score(modelo, X_treino, y_treino, cv=CV, scoring="accuracy")
        melhor_cv = float(scores_cv.mean())
        modelo.fit(X_treino, y_treino)
        melhor_modelo = modelo
        melhores_parametros = "sem busca de hiperparâmetros"
    else:
        busca = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=parametros,
            n_iter=n_iter,
            scoring="accuracy",
            cv=CV,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        busca.fit(X_treino, y_treino)
        melhor_cv = float(busca.best_score_)
        melhor_modelo = busca.best_estimator_
        melhores_parametros = limpar_parametros(busca.best_params_)

    previsoes = melhor_modelo.predict(X_teste)

    return melhor_modelo, {
        "modelo": nome,
        "acuracia_cv": melhor_cv,
        "acuracia_teste": accuracy_score(y_teste, previsoes),
        "precisao_teste": precision_score(y_teste, previsoes, average="macro", zero_division=0),
        "recall_teste": recall_score(y_teste, previsoes, average="macro", zero_division=0),
        "melhores_parametros": str(melhores_parametros),
        "matriz_confusao": confusion_matrix(y_teste, previsoes),
    }


def criar_modelos(numeric_features, categorical_features):
    return {
        "K-NN": {
            "modelo": criar_pipeline(KNeighborsClassifier(), numeric_features, categorical_features),
            "parametros": {
                "modelo__n_neighbors": randint(1, 50),
            },
            "n_iter": N_ITER,
        },
        "Naive Bayes": {
            "modelo": criar_pipeline(GaussianNB(), numeric_features, categorical_features),
            "parametros": None,
            "n_iter": N_ITER,
        },
        "Regressão Logística": {
            "modelo": criar_pipeline(
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                numeric_features,
                categorical_features,
            ),
            "parametros": {
                "modelo__C": uniform(loc=0.001, scale=10**8),
            },
            "n_iter": N_ITER,
        },
        "Árvore de Decisão": {
            "modelo": criar_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), numeric_features, categorical_features),
            "parametros": {
                "modelo__min_samples_split": uniform(loc=0.01, scale=0.99),
                "modelo__min_samples_leaf": uniform(loc=0.01, scale=0.99),
                "modelo__max_leaf_nodes": randint(2, 1000),
                "modelo__max_depth": randint(1, 1000),
            },
            "n_iter": N_ITER,
        },
        "Rede Neural": {
            "modelo": criar_pipeline(
                MLPClassifier(
                    solver="sgd",
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                ),
                numeric_features,
                categorical_features,
            ),
            "parametros": {
                "modelo__hidden_layer_sizes": [(10,), (25,), (50,), (75,), (100,), (50, 25)],
                "modelo__activation": ["identity", "logistic", "tanh"],
                "modelo__learning_rate_init": expon(loc=1e-6, scale=0.1),
                "modelo__max_iter": randint(100, 1000),
                "modelo__momentum": uniform(loc=0.5, scale=0.4999),
            },
            "n_iter": 15,
        },
        "SVM": {
            "modelo": criar_pipeline(SVC(random_state=RANDOM_STATE), numeric_features, categorical_features),
            "parametros": {
                "modelo__C": loguniform(1e-5, 1e5),
                "modelo__kernel": ["poly", "rbf", "sigmoid"],
                "modelo__degree": randint(2, 5),
                "modelo__gamma": loguniform(1e-5, 1e5),
            },
            "n_iter": 15,
        },
    }


def salvar_resultados(resultados):
    tabela = pd.DataFrame(resultados)
    tabela_sem_matriz = tabela.drop(columns=["matriz_confusao"])
    tabela_sem_matriz = tabela_sem_matriz.sort_values(
        by=["acuracia_teste", "precisao_teste", "recall_teste", "acuracia_cv"],
        ascending=False,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    caminho = RESULTS_DIR / "srag_resultados_modelos.csv"
    tabela_sem_matriz.to_csv(caminho, index=False)

    return tabela_sem_matriz, caminho


def salvar_matriz_confusao(matriz, nome_modelo):
    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Não UTI", "UTI"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusão - {nome_modelo}")

    RESULTS_DIR.mkdir(exist_ok=True)
    caminho = RESULTS_DIR / "srag_matriz_confusao_melhor_modelo.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()

    return caminho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_RAW_DATA_PATH, help="Caminho para o CSV bruto do conjunto SRAG.")
    args = parser.parse_args()

    df = carregar_dados(args.data)
    X, y, numeric_features, categorical_features = preparar_dados(df)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Base SRAG pediátrico")
    print("Formato original:", df.shape)
    print(f"Filtro usado: idade <= {PEDIATRIC_MAX_AGE} e SG_UF_NOT = {FILTER_NOTIFICATION_UF}")
    print("Treino:", X_treino.shape)
    print("Teste:", X_teste.shape)
    print("Atributos numéricos:", numeric_features)
    print("Quantidade de atributos categóricos:", len(categorical_features))
    print("Distribuição do alvo:")
    print(y.value_counts().sort_index())

    modelos = criar_modelos(numeric_features, categorical_features)
    melhores_modelos = {}
    resultados = []

    for nome, configuracao in modelos.items():
        print(f"\nTreinando modelo: {nome}")
        melhor_modelo, resultado = avaliar_modelo(
            nome=nome,
            modelo=configuracao["modelo"],
            parametros=configuracao["parametros"],
            X_treino=X_treino,
            X_teste=X_teste,
            y_treino=y_treino,
            y_teste=y_teste,
            n_iter=configuracao["n_iter"],
        )

        melhores_modelos[nome] = melhor_modelo
        resultados.append(resultado)

        print("Acurácia CV:", round(resultado["acuracia_cv"], 4))
        print("Acurácia teste:", round(resultado["acuracia_teste"], 4))
        print("Precisão teste:", round(resultado["precisao_teste"], 4))
        print("Recall teste:", round(resultado["recall_teste"], 4))
        print("Matriz de confusão:")
        print(resultado["matriz_confusao"])

    tabela, caminho_csv = salvar_resultados(resultados)
    nome_melhor = tabela.iloc[0]["modelo"]
    matriz_melhor = next(item["matriz_confusao"] for item in resultados if item["modelo"] == nome_melhor)
    caminho_matriz = salvar_matriz_confusao(matriz_melhor, nome_melhor)

    print("\nResultados finais:")
    print(tabela[["modelo", "acuracia_cv", "acuracia_teste", "precisao_teste", "recall_teste"]])
    print("\nMelhor modelo:", nome_melhor)
    print("CSV salvo em:", caminho_csv)
    print("Matriz de confusão salva em:", caminho_matriz)


if __name__ == "__main__":
    main()

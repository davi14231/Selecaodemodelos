import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon, loguniform, randint, uniform
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_ITER = 25
CV = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def carregar_dados():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names


def separar_treino_teste(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def limpar_parametros(valor):
    if isinstance(valor, dict):
        return {chave: limpar_parametros(item) for chave, item in valor.items()}
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def avaliar_modelo(nome, modelo, parametros, X_treino, X_teste, y_treino, y_teste, n_iter=N_ITER):
    if parametros is None:
        scores_cv = cross_val_score(modelo, X_treino, y_treino, cv=CV, scoring="accuracy")
        melhor_cv = scores_cv.mean()
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
        melhor_cv = busca.best_score_
        melhor_modelo = busca.best_estimator_
        melhores_parametros = limpar_parametros(busca.best_params_)

    previsoes = melhor_modelo.predict(X_teste)

    resultado = {
        "modelo": nome,
        "acuracia_cv": accuracy_score_cv(melhor_cv),
        "acuracia_teste": accuracy_score(y_teste, previsoes),
        "precisao_teste": precision_score(y_teste, previsoes, average="macro", zero_division=0),
        "recall_teste": recall_score(y_teste, previsoes, average="macro", zero_division=0),
        "melhores_parametros": str(melhores_parametros),
        "matriz_confusao": confusion_matrix(y_teste, previsoes),
    }

    return melhor_modelo, resultado


def accuracy_score_cv(valor):
    return float(valor)


def criar_modelos():
    modelos = {
        "K-NN": {
            "modelo": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", KNeighborsClassifier()),
            ]),
            "parametros": {
                "modelo__n_neighbors": randint(1, 50),
            },
            "n_iter": N_ITER,
        },
        "Naive Bayes": {
            "modelo": GaussianNB(),
            "parametros": None,
            "n_iter": N_ITER,
        },
        "Regressão Logística": {
            "modelo": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "parametros": {
                "modelo__C": uniform(loc=0.001, scale=10**8),
            },
            "n_iter": N_ITER,
        },
        "Árvore de Decisão": {
            "modelo": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "parametros": {
                "min_samples_split": uniform(loc=0.01, scale=0.99),
                "min_samples_leaf": uniform(loc=0.01, scale=0.99),
                "max_leaf_nodes": randint(2, 1000),
                "max_depth": randint(1, 1000),
            },
            "n_iter": N_ITER,
        },
        "Rede Neural": {
            "modelo": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", MLPClassifier(
                    solver="sgd",
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                )),
            ]),
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
            "modelo": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", SVC(random_state=RANDOM_STATE)),
            ]),
            "parametros": {
                "modelo__C": loguniform(1e-5, 1e5),
                "modelo__kernel": ["poly", "rbf", "sigmoid"],
                "modelo__degree": randint(2, 5),
                "modelo__gamma": loguniform(1e-5, 1e5),
            },
            "n_iter": N_ITER,
        },
    }

    return modelos


def plotar_fronteira_decisao(modelo, nome_modelo, X_treino, X_teste, y_treino, y_teste, nomes_atributos, nomes_classes):
    features_2d = [2, 3]

    X2_treino = X_treino[:, features_2d]
    X2_teste = X_teste[:, features_2d]

    modelo_2d = clone(modelo)
    modelo_2d.fit(X2_treino, y_treino)

    x_min, x_max = X2_treino[:, 0].min() - 0.5, X2_treino[:, 0].max() + 0.5
    y_min, y_max = X2_treino[:, 1].min() - 0.5, X2_treino[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grade = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo_2d.predict(grade).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.Set2)

    pontos = plt.scatter(
        X2_teste[:, 0],
        X2_teste[:, 1],
        c=y_teste,
        cmap=plt.cm.Set2,
        edgecolor="black",
    )

    plt.xlabel(nomes_atributos[features_2d[0]])
    plt.ylabel(nomes_atributos[features_2d[1]])
    plt.title(f"Fronteira de decisão - {nome_modelo}")

    handles, _ = pontos.legend_elements()
    plt.legend(handles, nomes_classes, title="Classes")

    RESULTS_DIR.mkdir(exist_ok=True)
    caminho = RESULTS_DIR / "fronteira_decisao.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()

    return caminho


def salvar_resultados(resultados):
    tabela = pd.DataFrame(resultados)
    tabela = tabela.drop(columns=["matriz_confusao"])
    tabela = tabela.sort_values(
        by=["acuracia_teste", "precisao_teste", "recall_teste", "acuracia_cv"],
        ascending=False,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    caminho = RESULTS_DIR / "resultados_modelos.csv"
    tabela.to_csv(caminho, index=False)

    return tabela, caminho


def main():
    X, y, nomes_atributos, nomes_classes = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(X, y)

    print("Treino:", X_treino.shape)
    print("Teste:", X_teste.shape)
    print("Classes:", nomes_classes)

    modelos = criar_modelos()
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
    melhor_modelo = melhores_modelos[nome_melhor]

    caminho_grafico = plotar_fronteira_decisao(
        modelo=melhor_modelo,
        nome_modelo=nome_melhor,
        X_treino=X_treino,
        X_teste=X_teste,
        y_treino=y_treino,
        y_teste=y_teste,
        nomes_atributos=nomes_atributos,
        nomes_classes=nomes_classes,
    )

    print("\nResultados finais:")
    print(tabela[["modelo", "acuracia_cv", "acuracia_teste", "precisao_teste", "recall_teste"]])
    print("\nMelhor modelo:", nome_melhor)
    print("CSV salvo em:", caminho_csv)
    print("Gráfico salvo em:", caminho_grafico)


if __name__ == "__main__":
    main()


from time import time
from datetime import timedelta
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier as model

from src.pipeline import Pipeline

def media(lista):
    """Calcula a média de uma lista de números"""
    asw = sum(lista) / len(lista)
    if np.isnan(asw):
        return 0
    return asw

def mediana(lista):
    """Calcula a mediana de uma lista de números"""
    sorted_lista = sorted(lista)
    n = len(sorted_lista)
    meio = n // 2
    asw = sorted_lista[meio]
    try:
        if np.isnan(asw):
            return 0
    except:
        return asw
    return asw

def moda(lista):
    """Calcula a moda de uma lista de números"""
    counter = Counter(lista)
    max_freq = max(counter.values())
    asw = [item for item, freq in counter.items() if freq == max_freq][0]
    return asw

def set_zero(lista):
    return 0

if __name__ == "__main__":
    
    start = time()

    main_train = pd.read_csv("data/trabalho1/conjunto_de_treinamento.csv")
    to_answer = pd.read_csv("data/trabalho1/conjunto_de_teste.csv")

    # Dividido em proporção 0.81, 0.09, 0.10 em treino, teste e validação

    train, validacao = train_test_split(main_train, test_size=0.1)

    pipeline = Pipeline(scaler=RobustScaler(),
                        model=model(max_depth=16),
                        fill_categorical_strategy=mediana,
                        fill_numerical_strategy=set_zero,
                        exclude_collumns=["grau_instrucao"],
                        y_column="inadimplente",
                        identifier="id_solicitante")

    treated_data, answer = pipeline.process(train, validacao, see_action=True)
    tempo = str(timedelta(seconds=time()-start))
    # answer = answer.astype(int)
    resp = accuracy_score(validacao["inadimplente"], answer["inadimplente"])
    print(resp, tempo)

    #pipeline.view_config()
    treated_data, answer = pipeline.process(main_train, to_answer, see_action=True)
    answer = answer.astype(int)
    answer.to_csv("data/trabalho1/results/na_mao.csv", index=False)

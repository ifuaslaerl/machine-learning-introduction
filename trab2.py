from random import seed, sample, shuffle
from time import time
from datetime import timedelta
from math import inf
from collections import Counter
import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

from src.models_settings import options_regressors, random_random_forest_regressor
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

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def get_pipelines() -> typing.List[Pipeline]:

    scalers = [RobustScaler(), None]
    fill_categorical_strategys = [mediana, moda, set_zero]
    fill_numerical_strategys = [set_zero, media, mediana, moda]
    exclude_columns = ["diferenciais", "bairro", "s_festas"]

    models = options_regressors()

    objs = []

    for model in models:
        for scaler in scalers:
            for fill_categorical_strategy in fill_categorical_strategys:
                for fill_numerical_strategy in fill_numerical_strategys:
                    for kndex in range(len(exclude_columns)+1):

                        obj = Pipeline(scaler=scaler,
                            fill_categorical_strategy=fill_categorical_strategy,
                            fill_numerical_strategy=fill_numerical_strategy,
                            exclude_collumns=exclude_columns[:kndex],
                            model=model,
                            identifier="Id",
                            y_column="preco")

                        #obj.view_config()
                        objs.append(obj)

    return objs

def get_pipelines2():

    scalers = [RobustScaler(), None]
    fill_categorical_strategys = [mediana, set_zero]
    fill_numerical_strategys = [set_zero, media, mediana]
    exclude_columns = ["diferenciais", "playground", "bairro", "estacionamento"]

    models = options_regressors()

    objs = []

    for model in models:
        for scaler in scalers:
            for fill_categorical_strategy in fill_categorical_strategys:
                for fill_numerical_strategy in fill_numerical_strategys:
                    for kndex in range(len(exclude_columns)+1):

                        obj = Pipeline(scaler=scaler,
                            fill_categorical_strategy=fill_categorical_strategy,
                            fill_numerical_strategy=fill_numerical_strategy,
                            exclude_collumns=exclude_columns[:kndex+1],
                            model=model,
                            identifier="Id",
                            y_column="preco")

                        #obj.view_config()
                        objs.append(obj)

    return objs

if __name__ == "__main__":
    
    start = time()

    #seed(201058)

    main_train = pd.read_csv("data/trabalho2/conjunto_de_treinamento.csv")
    to_answer = pd.read_csv("data/trabalho2/conjunto_de_teste.csv")

    # Dividido em proporção 0.81, 0.09, 0.10 em treino, teste e validação

    train1, validacao = train_test_split(main_train, test_size=0.1)
    train, test = train_test_split(train1, test_size=0.1)

    objs = get_pipelines2()
    
    shuffle(objs)
    
    objs = sample(objs,1000)
    
    shuffle(objs)

    print(f"primeira parte {len(objs)} serão testados.")

    results = []

    for index, obj in enumerate(objs):
        #obj.view_config()
        treated_data, answer = obj.process(train, test)
        
        # colunas = [coluna for coluna in treated_data.columns if coluna != "Id"]
        
        # treated_data = treated_data[colunas]
        
        # matriz = treated_data.corr()
        # plt.imshow(matriz)
        # plt.colorbar()
        # plt.show()
        
        # menor = inf
        # for jndex, element in enumerate(matriz["preco"]):
        #     if element < menor:
        #         menor = element
        #         sla = jndex
        # print(treated_data.columns[sla], menor)
        
        precisao = rmspe(test["preco"], answer["preco"])
        tempo = str(timedelta(seconds=time()-start))
        print(index, precisao, tempo)
        results.append((precisao, obj))

    results.sort(key= lambda x: x[0], reverse=False)
    results = results[:max(round(len(results)*0.1),1)]
    
    print(f"segunda parte {len(results)} serão testados.")

    result_final = []
    for index, (porcentagem , model) in enumerate(results):
        treated_data, answer = model.process(train, validacao)
        precisao_ = rmspe(validacao["preco"], answer["preco"])

        
        #model.view_config()
        tempo = str(timedelta(seconds=time()-start))
        print(index, precisao_, porcentagem, tempo)
        result_final.append((precisao_, model))

    result_final.sort(key= lambda x: x[0], reverse=False)    
    result_final = result_final[:max(round(len(results)*0.1),1)]

    print(f"{len(result_final)} final options to choose")

    for index, (porcentagem, model) in enumerate(result_final):
        model.view_config()
        tempo = str(timedelta(seconds=time()-start))
        print(index, porcentagem, tempo)
        treated_data, answer = model.process(main_train, to_answer)
        answer["Id"] = answer["Id"].astype(int)
        answer.to_csv(f"data/trabalho2/results/{index}.csv", index=False)

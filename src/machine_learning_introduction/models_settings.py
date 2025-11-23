""" Módulo feito para fit e test dos dados. """

# Bibliotecas usadas
from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def random_forest_regressor(N=2000):
    options = []
    n_estimators = []
    max_features = ["sqrt", "log2", None, 1.0]
    max_death = []
    
    i = 1
    while i<N:
        n_estimators.append(i)
        max_death.append(i)
        i*=2
    
    for n in n_estimators:
        for m in max_death:
            for mf in max_features:
                options.append(RandomForestRegressor(n_estimators=n, max_depth=m, max_features=mf))
    
    return options

def knn_regressor(N=2000):
    options = []
    n_neighbours = []
    weights = ["uniform", "distance"]
    metric = ["manhattan", "nan_euclidean", "euclidean", "cosine"]
    
    i = 1
    while i<N:
        n_neighbours.append(i)
        i*=3

    for n in n_neighbours:
        for w in weights:
            for m in metric:
                options.append(KNeighborsRegressor(n_neighbors=n, weights=w, metric=m))

    return options

def options_regressors():
    
    options = [LinearRegression(fit_intercept=True), LinearRegression(fit_intercept=False)]
    
    options += random_forest_regressor()
    options += knn_regressor()

    return options

def knn_classifier(N=2000):
    options = []
    n_neighbours = []
    weights = ["uniform", "distance"]
    metric = ["manhattan", "nan_euclidean", "euclidean", "cosine"]
    
    i = 1
    while i<N:
        n_neighbours.append(i)
        i*=3

    for n in n_neighbours:
        for w in weights:
            for m in metric:
                options.append(KNeighborsClassifier(n_neighbors=n, weights=w, metric=m))

    return options

def random_forest_classifier(N=2000):
    options = []
    n_estimators = []
    max_death = []
    
    i = 1
    while i<N:
        n_estimators.append(i)
        max_death.append(i)
        i*=2
    
    for n in n_estimators:
        for m in max_death:
            options.append(RandomForestClassifier(n_estimators=n, max_depth=m))
    
    return options

def options_classifiers():
    options = [LogisticRegressionCV(fit_intercept=True), LogisticRegressionCV(fit_intercept=False)]
    options += knn_classifier()
    options += random_forest_classifier()

    return options

def random_random_forest_regressor(n, a, b):
    options = []
    
    for i in range(n):
        options.append(RandomForestRegressor(n_estimators=randint(1,a), max_depth=randint(1,b)))
        
    return options
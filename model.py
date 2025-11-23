import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class ModelPipeline:
    def __init__(self, model: BaseEstimator):
        self.model = model
    
    def fit(self, X_train, y_train):
        """Ajusta o modelo ao conjunto de treino"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo no conjunto de teste"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, self.model

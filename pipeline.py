import typing
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Pipeline:
    def __init__(self, scaler: TransformerMixin, fill_categorical_strategy: typing.Callable[[pd.Series], float], 
                fill_numerical_strategy: typing.Callable[[pd.Series], float],
                select_collumns: typing.List[str],
                model: ClassifierMixin,
                y_column: str,
                test_size: float):
        self.scaler = scaler
        self.fill_categorical_strategy = fill_categorical_strategy
        self.fill_numerical_strategy = fill_numerical_strategy
        self.selected_columns = select_collumns
        self.model = model
        self.y_column = y_column
        self.test_size = test_size
    
    def fill_categorical(self, df: pd.DataFrame):
        """Preenche valores categóricos com estratégia personalizada"""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(self.fill_categorical_strategy(df[col]))
        return df

    def encode_categorical(self, df: pd.DataFrame):
        """Converte variáveis categóricas para dummies apenas para as colunas categóricas."""
        colunas_categoricas = df.select_dtypes(include=['object']).columns
        df_categorico = df[colunas_categoricas]
        
        df_encoded = pd.get_dummies(df_categorico, drop_first=True)
        
        df = df.drop(columns=colunas_categoricas).join(df_encoded)
        
        return df
    
    def fill_numerical(self, df: pd.DataFrame):
        """Preenche valores numéricos com estratégia personalizada"""
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(self.fill_numerical_strategy(df[col]))
        return df
    
    def normalize(self, df: pd.DataFrame):
        """Normaliza os dados numéricos"""
        numerical_cols = df.select_dtypes(include=[np.number, int, float, bool]).columns
        assert len(numerical_cols) == len(df.columns), f"Dados não numéricos presentes {len(numerical_cols)}, {len(df.columns)}"
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        return df

    def fit(self, X_train, y_train):
        """Ajusta o modelo ao conjunto de treino"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Avalia o modelo no conjunto de teste"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, self.model

    def process(self, train: pd.DataFrame, validate: pd.DataFrame):
        """Processa dados de treino e teste"""
        
        y_train = train[self.y_column]
        
        if self.selected_columns:
            train = train[self.selected_columns]
            validate = validate[self.selected_columns]
        
        train = self.fill_categorical(train)
        validate = self.fill_categorical(validate)
        
        print("categorical data filled")
        
        train = self.encode_categorical(train)
        validate = self.encode_categorical(validate)
        
        print("data encoded")
        
        train = self.fill_numerical(train)
        validate = self.fill_numerical(validate)
        
        print("numerical data filled")
        
        train, validate = train.align(validate, join="inner", axis=1)  # Garantir que colunas sejam iguais
        
        print("columns are equal")
        
        train = self.normalize(train)
        validate = self.normalize(validate)
        
        print("data normalized")
        
        # Gera matriz de correlação do conjunto de treino
        correlation_matrix = train.corr()
        
        print("correlation_matrix made")
        
        train, test, y_train, y_test = train_test_split(
            train, y_train, test_size=self.test_size, shuffle=True)
        
        self.model.fit(train, y_train)
        pred = self.model.predict(test)
        precisao = accuracy_score(pred, y_test)

        pred = self.model.predict(validate)
        
        # Garantir controle dos Id's
        
        return train, test, validate, correlation_matrix

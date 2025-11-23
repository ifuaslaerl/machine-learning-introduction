""" Module providing loading and pre-processing functions. """

import typing
from bisect import bisect_left
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

class DataManager():
    """ Class representing dataloader. """

    def __init__(self, main_path: str):
        self.data = pd.read_csv(main_path)

    def normalize(self, columns: typing.List[str], \
                    normalization: sklearn.base.TransformerMixin) -> None:
        """ Normalize data based on a function """

        self.data[columns] = normalization.fit_transform(self.data[columns])

    def find_index_by_column_value(self, value: any, column: str) -> int:
        """ Get index line of determined id. """

        index = bisect_left(self.data[column], value)

        if value != self.data[column][index]:
            raise ValueError(f"{value} not found in {column}.")

        return index

    def transform_column(self, column: str, function: typing.Callable[[any],int]) -> None:
        """ Transform column based on personalized function. """

        self.data[column] = self.data[column].apply(lambda item: function(item))

    def set_column(self, column: str, data: any) -> None:
        """ Set column to data. """
        self.data[column] = data

    def get_column(self, column: str) -> np.array:
        """ Get column from data. """
        return self.data[column]

    def filter_columns(self, type_of: type) -> typing.List[str]:
        """ Overview data by columns. """

        columns = []
        for column in self.data.columns:
            if self.data[column].dtype == type_of:
                columns.append(column)
        return columns

    def view_columns(self, columns: typing.List[str] = None) -> None:
        """ View data of columns. """

        for column in (columns if columns else self.data.columns):
            print(f"{column}: {self.data[column].dtype}")

    def correlation_matrix(self, columns: typing.List[str], figsize=(10,8),\
                            cmap: str = "coolwarm") -> None:
        """ Gets correlations between columns. """

        if not self.data[columns].dtypes.apply(lambda dtype: dtype in [float, int]).all():
            raise ValueError("Not all data is numerical.")

        corr_matrix = np.corrcoef(self.data[columns].T)

        plt.figure(figsize=figsize)
        plt.title("Correlation Matrix")
        sns.heatmap(corr_matrix, annot=False, cmap=cmap, xticklabels=columns, yticklabels=columns)
        plt.show()

if __name__ == "__main__":

    loader = DataManager("data/trabalho1/conjunto_de_treinamento.csv")

    loader.view_columns(loader.filter_columns(object))

    loader.transform_column("forma_envio_solicitacao")

    """
        forma_envio_solicitacao: object
        sexo: object
        estado_onde_nasceu: object
        estado_onde_reside: object
        possui_telefone_residencial: object
        codigo_area_telefone_residencial: object
        possui_telefone_celular: object
        vinculo_formal_com_empresa: object
        estado_onde_trabalha: object
        possui_telefone_trabalho: object
        codigo_area_telefone_trabalho: object
    """

    columns_ = loader.filter_columns(int)
    columns_ += loader.filter_columns(float)

    loader.normalize(columns_, sklearn.preprocessing.RobustScaler())

    #loader.view_columns(columns_)

    loader.correlation_matrix(columns_)

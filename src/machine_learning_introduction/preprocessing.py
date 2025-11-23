import typing
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.base
import matplotlib.pyplot as plt
import seaborn as sns
from bisect import bisect_left

# --- Strategies ---

def strategy_mean(lista):
    """Calculates mean, safe for NaN."""
    asw = sum(lista) / len(lista)
    if np.isnan(asw):
        return 0
    return asw

def strategy_median(lista):
    """Calculates median, safe for NaN."""
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

def strategy_mode(lista):
    """Calculates mode."""
    counter = Counter(lista)
    if not counter:
        return 0
    max_freq = max(counter.values())
    asw = [item for item, freq in counter.items() if freq == max_freq][0]
    return asw

def strategy_zero(lista):
    return 0

# --- Legacy DataManager (Refactored from loader.py) ---

class DataManager:
    def __init__(self, main_path: str):
        self.data = pd.read_csv(main_path)

    def normalize(self, columns: typing.List[str], normalization: sklearn.base.TransformerMixin) -> None:
        self.data[columns] = normalization.fit_transform(self.data[columns])

    def correlation_matrix(self, columns: typing.List[str], figsize=(10,8), cmap: str = "coolwarm") -> None:
        if not self.data[columns].dtypes.apply(lambda dtype: dtype in [float, int]).all():
            raise ValueError("Not all data is numerical.")

        corr_matrix = np.corrcoef(self.data[columns].T)
        plt.figure(figsize=figsize)
        plt.title("Correlation Matrix")
        sns.heatmap(corr_matrix, annot=False, cmap=cmap, xticklabels=columns, yticklabels=columns)
        plt.show()

import typing
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.base
import matplotlib.pyplot as plt
import seaborn as sns

# --- Strategies ---

def strategy_mean(lista):
    """Calculates mean, safe for NaN."""
    return lista.mean()

def strategy_median(lista):
    """Calculates median using optimized pandas method."""
    return lista.median()

def strategy_mode(lista):
    """Calculates mode."""
    # Pandas mode() returns a Series (could be multiple modes), take the first
    modes = lista.mode()
    if modes.empty:
        return 0
    return modes[0]

def strategy_zero(lista):
    return 0

# --- Legacy DataManager (Unchanged) ---
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

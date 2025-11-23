import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import numpy as np

# --- Path Management ---
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_data_path(trabalho: int, filename: str) -> Path:
    """Returns the absolute path for a data file based on the assignment ID."""
    subdir = f"trabalho{trabalho}"
    return DATA_DIR / subdir / filename

# --- Task Configuration ---

@dataclass
class TaskConfig:
    id: int
    y_column: str
    identifier: str
    exclude_columns: List[str]
    default_metric: str  # 'accuracy' or 'rmspe'

TASKS = {
    1: TaskConfig(
        id=1,
        y_column="inadimplente",
        identifier="id_solicitante",
        exclude_columns=["idade", "possui_telefone_residencial", "codigo_area_telefone_residencial", "meses_na_residencia"],
        default_metric="accuracy"
    ),
    2: TaskConfig(
        id=2,
        y_column="preco",
        identifier="Id",
        exclude_columns=["diferenciais", "bairro", "s_festas", "playground", "estacionamento"],
        default_metric="rmspe"
    )
}

# Machine Learning Introduction

A comprehensive machine learning pipeline designed for automated credit scoring (classification) and housing price prediction (regression). This project includes tools for model optimization, inference, and data processing.

## Installation

To install the package and its dependencies, run the following command in the project root:

```bash
pip install .
```

For development (editable mode):

```bash
pip install -e .
```

## Usage

The project installs two command-line interface (CLI) tools: `ml-optimize` and `ml-predict`.

### 1. Model Optimization (`ml-optimize`)

Runs a tournament-style optimization to find the best pipeline configuration (Scaler + Imputation Strategy + Model + Hyperparameters) for a specific task.

**Arguments:**
* `--task`: The task ID (1 for Credit Scoring, 2 for Housing Prices).
* `--samples`: (Optional) Number of random pipeline configurations to test (default: 500).

**Example:**

```bash
# Run optimization for Task 1 (Credit Scoring) with 1000 candidates
ml-optimize --task 1 --samples 1000

# Run optimization for Task 2 (Housing Prices)
ml-optimize --task 2
```

### 2. Inference (`ml-predict`)

Runs inference using a manually configured model on the test dataset and generates a submission file.

**Arguments:**
* `--task`: The task ID (1 for Credit Scoring, 2 for Housing Prices).

**Example:**

```bash
# Generate predictions for Task 1
ml-predict --task 1
```

## Supported Tasks

The pipeline supports two specific assignments defined in `src/machine_learning_introduction/config.py`:

* **Task 1: Credit Scoring**
    * **Type:** Classification
    * **Target:** `inadimplente`
    * **Metric:** Accuracy

* **Task 2: Housing Prices**
    * **Type:** Regression
    * **Target:** `preco`
    * **Metric:** RMSPE (Root Mean Squared Percentage Error)

## Project Structure

```text
.
├── data/                   # Data directories (trabalho1, trabalho2)
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # CLI scripts (run_inference.py, run_optimization.py)
├── src/                    # Source code
│   └── machine_learning_introduction/
│       ├── config.py       # Task configurations
│       ├── models_settings.py # Model hyperparameters/factories
│       ├── pipeline.py     # Core pipeline logic
│       └── preprocessing.py # Imputation strategies
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Project documentation
```

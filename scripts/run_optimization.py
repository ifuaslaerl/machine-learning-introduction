import sys
from pathlib import Path
from time import time
from datetime import timedelta
from random import shuffle, sample
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from machine_learning_introduction import (
    Pipeline, options_classifiers, options_regressors, 
    get_data_path, TASKS
)
from machine_learning_introduction.preprocessing import (
    strategy_mean, strategy_median, strategy_mode, strategy_zero
)

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

@click.command()
@click.option('--task', type=int, required=True, help='Task ID (1 for Credit, 2 for Housing)')
@click.option('--samples', default=500, help='Number of random pipelines to test')
def main(task, samples):
    """Runs a tournament optimization to find the best model/pipeline."""
    
    config = TASKS.get(task)
    if not config:
        click.echo(f"Error: Task {task} not found.")
        return

    click.echo(f"Starting Optimization for Task {task} ({config.default_metric})")
    start = time()

    # Load Data
    train_path = get_data_path(task, "conjunto_de_treinamento.csv")
    test_path = get_data_path(task, "conjunto_de_teste.csv")
    
    main_train = pd.read_csv(train_path)
    to_answer = pd.read_csv(test_path)

    # Splits
    train1, validation = train_test_split(main_train, test_size=0.1, random_state=42)
    train, test = train_test_split(train1, test_size=0.1, random_state=42)

    # Setup Pipeline Grid
    scalers = [RobustScaler(), None]
    fill_cat = [strategy_median, strategy_zero] if task == 1 else [strategy_median, strategy_mode, strategy_zero]
    fill_num = [strategy_zero, strategy_median] if task == 1 else [strategy_zero, strategy_mean, strategy_median]
    
    models = options_classifiers() if task == 1 else options_regressors()
    
    pipelines = []
    
    # Generate combinations
    for model in models:
        for scaler in scalers:
            for cat_strat in fill_cat:
                for num_strat in fill_num:
                    # Feature selection variation
                    for k in range(len(config.exclude_columns) + 1):
                        obj = Pipeline(
                            scaler=scaler,
                            fill_categorical_strategy=cat_strat,
                            fill_numerical_strategy=num_strat,
                            exclude_collumns=config.exclude_columns[:k],
                            model=model,
                            identifier=config.identifier,
                            y_column=config.y_column
                        )
                        pipelines.append(obj)

    # Random Sampling
    shuffle(pipelines)
    selected_pipelines = sample(pipelines, min(samples, len(pipelines)))
    click.echo(f"Testing {len(selected_pipelines)} candidates...")

    # Phase 1: Test
    results = []
    for i, obj in enumerate(selected_pipelines):
        try:
            _, answer = obj.process(train.copy(), test.copy())
            
            if config.default_metric == 'accuracy':
                score = accuracy_score(test[config.y_column], answer[config.y_column])
                reverse_sort = True
            else:
                score = rmspe(test[config.y_column], answer[config.y_column])
                reverse_sort = False # Lower is better for error
                
            results.append((score, obj))
            if i % 50 == 0:
                click.echo(f"Processed {i}/{len(selected_pipelines)}")
        except Exception as e:
            continue

    # Sort results
    results.sort(key=lambda x: x[0], reverse=reverse_sort)
    top_tier = results[:max(round(len(results) * 0.1), 1)]
    
    click.echo(f"Refining top {len(top_tier)} candidates on Validation set...")

    # Phase 2: Validation
    final_results = []
    for score_test, model in top_tier:
        try:
            _, answer = model.process(train.copy(), validation.copy())
            
            if config.default_metric == 'accuracy':
                score_val = accuracy_score(validation[config.y_column], answer[config.y_column])
            else:
                score_val = rmspe(validation[config.y_column], answer[config.y_column])
                
            final_results.append((score_val, model))
        except:
            continue

    final_results.sort(key=lambda x: x[0], reverse=reverse_sort)
    
    # Output
    best_score, best_model = final_results[0]
    click.echo(f"\nBest Score ({config.default_metric}): {best_score}")
    best_model.view_config()
    
    click.echo("Generating submission file...")
    _, final_answer = best_model.process(main_train, to_answer)
    
    output_path = get_data_path(task, "submission_optimized.csv")
    final_answer.to_csv(output_path, index=False)
    click.echo(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

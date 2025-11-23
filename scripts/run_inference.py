import sys
from pathlib import Path
from time import time
from datetime import timedelta
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from machine_learning_introduction import Pipeline, get_data_path, TASKS
from machine_learning_introduction.preprocessing import strategy_median, strategy_zero

@click.command()
@click.option('--task', type=int, required=True, help='Task ID (1 for Credit, 2 for Housing)')
def main(task):
    """Runs a single manual model configuration for the task."""
    
    config = TASKS.get(task)
    if not config:
        click.echo("Invalid Task ID")
        return

    click.echo(f"Running Inference for Task {task}...")
    start = time()

    train_path = get_data_path(task, "conjunto_de_treinamento.csv")
    test_path = get_data_path(task, "conjunto_de_teste.csv")
    
    main_train = pd.read_csv(train_path)
    to_answer = pd.read_csv(test_path)

    # Split for validation
    train, validation = train_test_split(main_train, test_size=0.1, random_state=42)

    # Configure Model manually based on original main1.py / main2.py
    if task == 1:
        model = RandomForestClassifier(max_depth=16)
    else:
        model = RandomForestRegressor(max_depth=256, max_features=None, n_estimators=8)

    pipeline = Pipeline(
        scaler=RobustScaler(),
        model=model,
        fill_categorical_strategy=strategy_median,
        fill_numerical_strategy=strategy_zero,
        exclude_collumns=config.exclude_columns,
        y_column=config.y_column,
        identifier=config.identifier
    )

    # Run on validation
    _, answer = pipeline.process(train, validation, see_action=True)
    
    # Run on final test set
    _, final_answer = pipeline.process(main_train, to_answer, see_action=False)
    
    output_path = get_data_path(task, "submission_manual.csv")
    final_answer.to_csv(output_path, index=False)
    
    click.echo(f"Finished in {timedelta(seconds=time()-start)}")
    click.echo(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

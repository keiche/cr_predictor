#!/usr/bin/env python3
"""Challenge rating prediction using supervised machine learning"""

# Standard libraries
from typing import Union

# PyPI libraries
import click
import colorama
import logging
import numpy as np
import pandas as pd
from tabulate import tabulate

# Custom libraries
from cr_model.objects import CRModel
from cr_model.utils import round_to_cr

logger = logging.getLogger(__name__)


def color_diff(diff: Union[float, int]) -> str:
    """
    Color the difference
        GREEN = exact
        YELLOW = close
        RED = far
    :param diff: difference
    :return: colored result
    """
    if diff == 0:
        return colorama.Fore.GREEN + str(diff) + colorama.Fore.RESET
    if abs(diff) <= 2:
        return colorama.Fore.YELLOW + str(diff) + colorama.Fore.RESET
    return colorama.Fore.RED + str(diff) + colorama.Fore.RESET


@click.group()
@click.option("-d", "--debug", is_flag=True, help="Debug output")
def cli(debug) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )


@cli.command("validate")
@click.argument("featureset", type=click.Path(exists=True))
@click.option(
    "-s",
    "--split",
    type=int,
    default=10,
    show_default=True,
    help="Number of cross-validation splits",
)
@click.option(
    "-t",
    "--testsize",
    type=float,
    default=0.3,
    show_default=True,
    help="Percent of training set to use for testing (decimal from 0-1)",
)
def validate(featureset, split, testsize) -> None:
    """Cross-validation scores for a feature set"""
    # Fit
    crm = CRModel()
    crm.train_model(pd.read_csv(featureset))

    # Cross-validate
    cv_scores = crm.cross_validate_scores(splits=split, test_size=testsize)

    # Create table to prep for tabulate output
    stats = []
    for c, score in enumerate(cv_scores):
        stats.append([c, round((score * 100), 2)])

    print(tabulate(stats, ["test count", "score"], tablefmt="pretty"))
    print(f"{round((cv_scores.mean() * 100), 2)}% accuracy with a standard deviation of {round(cv_scores.std(), 4)}")


@cli.command("features")
@click.argument("featureset", type=click.Path(exists=True))
def feature_importances(featureset) -> None:
    """Evaluate the importances of features"""
    # Reading from the raw dataset and fitting the model to get the feature set names dynamically
    df = pd.read_csv(featureset)
    crm = CRModel()
    crm.train_model(df)

    headers = list(df.columns.values)

    # Create feature importance table
    table = []
    for i, v in enumerate(crm.feature_importance()):
        table.append([i, headers[i + 1], round(v, 4)])

    print(tabulate(table, ["num", "name", "importance"], tablefmt="pretty"))


@cli.command("train")
@click.argument("featureset", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=str,
    default="cr_model.joblib",
    show_default=True,
    help="Output model name",
)
def train(featureset, output) -> None:
    """Fit and save a classification model"""
    # Load the feature set
    crm = CRModel()

    # Fit
    crm.train_model(pd.read_csv(featureset))

    # Save the model
    crm.save_model(output)
    logger.info(f"Saved model to {output}")


@cli.command("predict")
@click.argument("model", type=click.Path(exists=True))
@click.argument("prediction", type=click.File("r"))
def predict(model, prediction) -> None:
    """Predict the CR using a classification model"""
    # Load the model
    crm = CRModel()
    crm.load_model(model)

    # Go through each creature
    results = []
    for i, line in enumerate(prediction):
        # Split the CSV input
        cells = line.split(",")

        # Skip the header row if it exists
        if i == 0 and not cells[1].isdigit():
            continue

        # Predict
        p = crm.predict(cells[1:])
        results.append([cells[0], str(round_to_cr(p))])
    print(tabulate(results, headers=["name", "predicted cr"], tablefmt="pretty"))


@cli.command("predict_known")
@click.argument("model", type=click.Path(exists=True))
@click.argument("prediction", type=click.File("r"))
def predict_known(model, prediction) -> None:
    """Predict the CR with known outcomes
    Used to show granular test output for a trained model"""
    crm = CRModel()
    crm.load_model(model)

    diffs = []
    results = []
    for line in prediction:
        # Split csv
        cells = line.split(",")
        if cells[0] == "name":
            continue

        # Actual
        actual = round_to_cr(float(cells[-1]))

        # Prediction
        p = crm.predict([int(e) for e in cells[1:-1]])

        # Difference between actual and prediction
        diff = abs(float(actual) - round_to_cr(p))
        diffs.append(diff)

        # Store results
        results.append([cells[0], str(actual), str(round_to_cr(p)), color_diff(diff)])
    print(
        tabulate(
            results,
            headers=["name", "actual", "predicted", "difference"],
            tablefmt="pretty",
        )
    )

    # Final stats
    print("\n" + "Difference Stats")
    total_headers = ["stat", "value"]
    total_stats = [
        ["avg", color_diff(round(float(sum(diffs) / len(diffs)), 2))],
        ["median", color_diff(float(np.median(diffs)))],
        ["max", color_diff(max(diffs, key=abs))],
        ["min", color_diff(min(diffs, key=abs))],
        ["std", color_diff(round(np.std(diffs, axis=0), 2))],
    ]
    print("\n" + tabulate(total_stats, total_headers, tablefmt="pretty"))


if __name__ == "__main__":
    cli()

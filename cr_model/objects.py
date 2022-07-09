"""CR model class"""

# Standard libraries
import logging
from typing import Union

# PyPI libraries
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils import check_array

# Custom libraries
from cr_model.utils import split_dataset

logger = logging.getLogger(__name__)


class CRModel:
    def __init__(self) -> None:
        """
        Challenge rating classification model
        """
        # Classification model
        self.clf = None
        # Feature set (2D array)
        self.x = None
        # Outcome (1D array)
        self.y = None

    def train_model(self, feature_set: Union[pd.DataFrame, tuple]) -> None:
        """
        Train the regression model
        :param feature_set: Entire or pre-split feature set
        """
        # Split into the feature set and outcomes
        if isinstance(feature_set, pd.DataFrame):
            self.x, self.y = split_dataset(feature_set)
        elif isinstance(feature_set, tuple):
            self.x, self.y = feature_set
        else:
            raise TypeError(
                "dataset needs to be a pandas dataframe or a tuple of x/y "
                "from sklearn.model_selection.train_test_split"
            )

        # Remove the 'name' column from the feature set before using it
        self.x = check_array(np.delete(self.x, 0, 1))

        # Choose the regression algorithm
        regr = ExtraTreesRegressor()

        # Create the regression model
        logger.debug("Fitting the regression model")
        self.clf = regr.fit(self.x, self.y)

    def predict(self, attributes: list) -> float:
        """
        Use the ML model to predict the challenge rating
        :param attributes: Ordered list of attributes
        :return: predicted challenge rating
        """
        return self.clf.predict([attributes])[0]

    def cross_validate_scores(self, splits=5, test_size=0.3) -> list:
        """
        Cross-validation using ShuffleSplit to get a more randomized set of test data per fold
        https://scikit-learn.org/stable/modules/cross_validation.html
        :param splits: Number of cross-validation folds
        :param test_size: Percent of the total set that should be used for testing (decimal from 0-1)
        :return:
        """
        logger.debug("Validating: n_splits=%s, test_size=%s", splits, test_size)
        cv_split = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=0)
        return cross_val_score(self.clf, self.x, self.y, cv=cv_split)

    def feature_importance(self) -> dict:
        """
        Output feature importance
        :return:
        """
        return self.clf.feature_importances_

    def save_model(self, filename: str) -> None:
        """
        Save classification model
        :param filename: file name
        """
        dump(self.clf, filename)

    def load_model(self, filename: str) -> None:
        """
        Load classification model
        :param filename: file name
        """
        logger.debug("Loading model %s", filename)
        self.clf = load(filename)

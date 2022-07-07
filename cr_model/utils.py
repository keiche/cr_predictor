"""CR model utilities"""

# Standard libraries
from fractions import Fraction
from typing import Union

# PyPI libraries
import numpy as np
import pandas as pd


def round_to_cr(cr: float) -> Union[int, Fraction]:
    """
    Round to a valid challenge rating
    :param cr: Raw challenge rating
    :return: Closest possible challenge rating
    """
    if cr < 1:
        under_one = np.asarray([0, 0.125, 0.25, 0.5, 1])
        idx_min = (np.abs(under_one - cr)).argmin()
        final_cr = under_one[idx_min]
        return final_cr if cr >= 1 else Fraction(final_cr)
    return int(cr)


def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Split dataset into the feature set (x) and outcomes (y)
    This function does not remove the first column (labels)
    :param df: 2D array of feature set data including the outcomes
    :return: feature set (2D array), outcomes (1D array)
    """
    x = df.iloc[:, :-1].values
    y = df["cr"].values
    return x, y

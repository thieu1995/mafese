#!/usr/bin/env python
# Created by "Thieu" at 22:10, 31/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from mafese.selector import Selector
from mafese.utils import validator


class UnsupervisedSelector(Selector):
    """
    Defines a UnsupervisedSelector class that hold all Unsupervised learning methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    method: str, default = "MCL"
        We support:
            - "VAR": Variance Threshold method, remove all features whose variance is lower than the threshold.
            - "MCL": Multicollinearity method, remove features with high multicollinearity.

    threshold: int or float, default=10
        If method is `VAR`, the threshold should set a small number (i.g, 0.01, 0.1)
        If method is `MCL`, the threshold should set around 10.

    Attributes
    ----------
    support_values: np.ndarray
        The values of each feature (variance or variance inflation factor)

    Examples
    --------
    The following example shows how to retrieve the most informative features in the FilterSelector FS method

    >>> import pandas as pd
    >>> from mafese.filter import FilterSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features=5)
    >>> # find all relevant features
    >>> feat_selector.fit(X, y)
    >>> # check selected features - True (or 1) is selected, False (or 0) is not selected
    >>> print(feat_selector.selected_feature_masks)
    array([ True, True, True, False, False, True, False, False, False, True])
    >>> print(feat_selector.selected_feature_solution)
    array([ 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    >>> # check the index of selected features
    >>> print(feat_selector.selected_feature_indexes)
    array([ 0, 1, 2, 5, 9])
    >>> # call transform() on X to filter it down to selected features
    >>> X_filtered = feat_selector.transform(X)
    """

    SUPPORTED_METHODS = {
        "VAR": "variance_threshold",
        "MCL": "multicollinearity"
    }

    def __init__(self, problem="classification", method="MCL", threshold=10):
        super().__init__(problem)
        self.method = self.set_method(method)
        self.threshold = threshold
        self.support_values = None

    def set_method(self, method=None):
        if type(method) is str:
            return validator.check_str("method", method, list(self.SUPPORTED_METHODS.keys()))
        else:
            raise TypeError(f"Your method needs to be a string.")

    def transform(self, X):
        return X[:, self.selected_feature_indexes]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def get_vif(X):
        cc = np.corrcoef(X, rowvar=False)
        vif = np.linalg.inv(cc)
        return vif.diagonal()

    def fit(self, X, y=None):
        if self.method == "VAR":
            vt = VarianceThreshold(self.threshold).fit(X)
            sfm = vt.get_support().copy()
            self.support_values = vt.variances_
        else:
            vif = self.get_vif(X)
            self.support_values = vif
            sfm = vif <= self.threshold
        self.selected_feature_masks = sfm
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

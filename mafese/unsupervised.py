#!/usr/bin/env python
# Created by "Thieu" at 22:10, 31/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
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
            - "VAR": Variance Threshold method, remove all features whose variance smaller than a threshold.
            - "MAD": Mean Absolute Difference, remove all features whose mad value smaller than a threshold.
            - "DR": Dispersion Ratio, remove all features whose dr value smaller than a threshold.
            - "MCL": Multicollinearity method, remove features whose Variance Inflation Factor (VIF) value higher than a threshold.

    threshold: int or float, default=10
        If method is `VAR`, the threshold should set a small number (i.g, 0.01, 0.1)
        If method is `MAD`, the threshold is depended on MAD values of features
        If method is `DR`, the threshold is depended on DR values of features
        If method is `MCL`, the threshold should set around 10.

    Attributes
    ----------
    support_values: np.ndarray
        The values of each feature

    Examples
    --------
    The following example shows how to retrieve the most informative features in the UnsupervisedSelector FS method

    >>> import pandas as pd
    >>> from mafese.unsupervised import UnsupervisedSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = UnsupervisedSelector(problem='classification', method='DR', n_features=5)
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

    SUPPORT = {
        "VAR": "Variance Threshold",
        "DR": "Dispersion Ratio",
        "MAD": "Mean Absolute Difference",
        "MCL": "Multicollinearity",
    }

    def __init__(self, problem="classification", method="MCL", n_features=None, threshold=10):
        super().__init__(problem)
        self.method = self._set_method(method)
        self.n_features = n_features
        self.threshold = threshold
        self.support_values = None

    def _set_method(self, method=None):
        if type(method) is str:
            return validator.check_str("method", method, list(self.SUPPORT.keys()))
        else:
            raise TypeError(f"Your method needs to be a string.")

    def transform(self, X):
        return X[:, self.selected_feature_indexes]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def get_var(X):
        """
        Calculate the Variance values of data X
        """
        var = np.var(X, axis=0)
        return var

    @staticmethod
    def get_mcl(X):
        """
        Calculate the Variance Inflation Factor (VIF) values of data X
        """
        cc = np.corrcoef(X, rowvar=False)
        vif = np.linalg.inv(cc)
        return vif.diagonal()

    @staticmethod
    def get_mad(X):
        """
        Calculate the Mean Absolute Difference values of data X
        """
        mad = np.sum(np.abs(X - np.mean(X, axis=0)), axis=0) / X.shape[0]
        return mad

    @staticmethod
    def get_dr(X):
        """
        Calculate the Dispersion Ratio values of data X
        """
        X = X + 1
        am = np.mean(X, axis=0)
        gm = np.power(np.prod(X, axis=0), 1. / X.shape[0])
        return am / gm

    def fit(self, X, y=None):
        if self.n_features is None:
            res = getattr(self, f"get_{self.method.lower()}")(X)
            self.support_values = res
            if self.method == "MCL":
                idx = np.where(res <= self.threshold)[0]
            else:
                idx = np.where(res >= self.threshold)[0]
        else:
            self.n_features = validator.check_int("n_features", self.n_features, [1, X.shape[1]])
            res = getattr(self, f"get_{self.method.lower()}")(X)
            self.support_values = res
            if self.method == "MCL":
                idx = np.argpartition(res, self.n_features)[:self.n_features]
            else:
                idx = np.argpartition(res, -self.n_features)[-self.n_features:]
        if len(idx) == 0:
            print(f"N selected features set: {self.n_features}, threshold: {self.threshold}, supported values: {self.support_values}")
        self.selected_feature_indexes = np.sort(idx)
        self.selected_feature_masks = np.in1d(np.arange(len(res)), idx)
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)

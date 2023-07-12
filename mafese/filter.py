#!/usr/bin/env python
# Created by "Thieu" at 22:14, 24/05/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mafese.selector import Selector
from mafese.utils import validator, correlation


class FilterSelector(Selector):
    """
    Defines a FilterSelector class that hold all filter methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    method: str, default = "ANOVA"
        If the problem = "classification", FilterSelector's support method can be one of this value:

            - "CHI": Chi-Squared statistic
            - "ANOVA": ANOVA F-score
            - "MI": Mutual information
            - "KENDALL":  Kendall Tau correlation
            - "SPEARMAN": Spearman’s Rho correlation
            - "POINT": Point-biserial correlation
            - "RELIEF": Original Relief method

        If the problem = "regression", FilterSelector's support method can be one of this value:

            - "PEARSON": Pearson correlation
            - "ANOVA": ANOVA F-score
            - "MI": Mutual information
            - "KENDALL": Kendall Tau correlation
            - "SPEARMAN": Spearman’s Rho correlation
            - "POINT": Point-biserial correlation
            - "RELIEF": Original Relief method

    n_features: int or float, default=3
        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    n_neighbors : int, default=5, Optional
        Number of neighbors to use for computing feature importance scores of Relief-based family

    n_bins : int, default=10, Optional
        Number of bins to use for discretizing the target variable of Relief-based family in regression problems.

    Attributes
    ----------

    n_features: int
        The number of selected features.

    supported_methods: dict
        Key: is the support method name
        Value: is the support method function

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

    SUPPORT = {
        "classification": {"CHI": "chi2", "ANOVA": "f_classif", "MI": "mutual_info_classif",
                           "KENDALL": "kendall_func", "SPEARMAN": "spearman_func", "POINT": "point_func",
                           "RELIEF": "relief_func", },
        "regression": {"PEARSON": "r_regression", "ANOVA": "f_regression", "MI": "mutual_info_regression",
                       "KENDALL": "kendall_func", "SPEARMAN": "spearman_func", "POINT": "point_func",
                       "RELIEF": "relief_func", }
    }

    def __init__(self, problem="classification", method="ANOVA", n_features=3, n_neighbors=5, n_bins=10):
        super().__init__(problem)
        self.supported_methods = self.SUPPORT[self.problem]
        self.method = self._set_method(method)
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.n_bins = n_bins
        self.relief_flag = False

    def _set_method(self, method=None):
        if type(method) is str:
            method_name = validator.check_str("method", method, list(self.supported_methods.keys()))
            if method_name in ["relief_func", ]:
                self.relief_flag = True
            return getattr(correlation, self.supported_methods[method_name])
        else:
            raise TypeError(f"Your method needs to be a string.")

    def fit(self, X, y=None):
        if self.relief_flag:
            importance_scores = self.method(X, y, n_neighbors=self.n_neighbors, n_bins=self.n_bins)
        else:
            importance_scores = self.method(X, y)
        self.selected_feature_masks = correlation.select_bests(importance_scores, n_features=self.n_features)
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

    def transform(self, X):
        return X[:, self.selected_feature_indexes]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

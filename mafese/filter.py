#!/usr/bin/env python
# Created by "Thieu" at 22:14, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils import correlation


class Filter(Selector):
    """
    Defines a Filter class that hold all filter methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    method: str, default = "ANOVA"
        If the problem = "classification", Filter's support method can be one of this value:

            - "CHI": Chi-Squared statistic
            - "ANOVA": ANOVA F-score
            - "MI": Mutual information
            - "KENDALL":  Kendall Tau correlation
            - "SPEARMAN": Spearman’s Rho correlation
            - "POINT": Point-biserial correlation

        If the problem = "regression", Filter's support method can be one of this value:

            - "PEARSON": Pearson correlation
            - "ANOVA": ANOVA F-score
            - "MI": Mutual information
            - "KENDALL": Kendall Tau correlation
            - "SPEARMAN": Spearman’s Rho correlation
            - "POINT": Point-biserial correlation

    n_features: int or float, default=3
        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    Attributes
    ----------

    n_features: int
        The number of selected features.

    supported_methods: dict
        Key: is the support method name
        Value: is the support method function

    Examples
    --------
    import pandas as pd
    from mafese.filter import Filter

    # load X and y
    # NOTE mafese accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()

    # define mafese feature selection method
    feat_selector = Filter(problem='classification', method='SPEARMAN', n_features=5)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check selected features - True (or 1) is selected, False (or 0) is not selected
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)

    # check the index of selected features
    print(feat_selector.selected_feature_indexes)

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    """

    def __init__(self, problem="classification", method="ANOVA", n_features=3):
        super().__init__(problem)
        if self.problem == "classification":
            self.supported_methods = {"CHI": "chi2", "ANOVA": "f_classif", "MI": "mutual_info_classif",
                                      "KENDALL": "kendall_func", "SPEARMAN": "spearman_func", "POINT": "point_func"}
        else:
            self.supported_methods = {"PEARSON": "r_regression", "ANOVA": "f_regression", "MI": "mutual_info_regression",
                                      "KENDALL": "kendall_func", "SPEARMAN": "spearman_func", "POINT": "point_func"}
        self.method = self.set_method(method)
        self.set_selector(n_features)

    def set_method(self, method=None):
        if type(method) is str:
            method_name = validator.check_str("method", method, list(self.supported_methods.keys()))
            return getattr(correlation, self.supported_methods[method_name])
        else:
            raise TypeError(f"Your method needs to be a string.")

    def set_selector(self, n_features):
        self.n_features = n_features
        if type(n_features) is int:
            self.selector = correlation.SelectKBest(score_func=self.method, k=n_features)
        elif type(n_features) is float and 0 < n_features < 1:
            self.selector = correlation.SelectPercentile(score_func=self.method, percentile=self.n_features*100)
        else:
            raise TypeError(f"Type of n_features parameter is int or float. If int, 1 <= n_features <= max_features. If float, 0 < n_features < 1")

    def fit(self, X, y=None):
        self.selector = self.selector.fit(X, y)
        self.selected_feature_masks = self.selector._get_support_mask().copy()
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

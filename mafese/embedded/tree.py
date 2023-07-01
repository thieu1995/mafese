#!/usr/bin/env python
# Created by "Thieu" at 16:43, 31/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.feature_selection import SelectFromModel
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils.estimator import get_tree_based_estimator


class TreeSelector(Selector):
    """
    Defines a TreeSelector class that hold all Tree-based Feature Selection methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    estimator: str, default = 'tree'
        We are currently support:
            - rf: random forest
            - adaboost: AdaBoost
            - xgb: Gradient Boosting
            - tree: Extra Trees

    estimator_paras: None or dict, default = None
        The parameters of the estimator, please see the official document of scikit-learn to selected estimator.
        If None, we use the best parameter for selected estimator

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose absolute importance value is greater or equal
        are kept while the others are discarded. If "median" (resp. "mean"), then the ``threshold`` value is the median
        (resp. the mean) of the feature importances. A scaling factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below ``threshold`` in the case where the
        ``coef_`` attribute of the estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed by using the output of `max_feaures(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.

    Examples
    --------
    The following example shows how to retrieve the most informative features in the Tree-based FS method

    >>> import pandas as pd
    >>> from mafese.embedded.tree import TreeSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = TreeSelector(problem="classification", estimator="tree")
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

    SUPPORTED = ["rf", "adaboost", "xgb", "tree"]

    def __init__(self, problem="classification", estimator="tree", estimator_paras=None, threshold=None, norm_order=1, max_features=None):
        super().__init__(problem)
        self.estimator = self._set_estimator(estimator, estimator_paras)
        self.estimator_paras = estimator_paras
        self.threshold = threshold
        self.norm_order = norm_order
        self.max_features = max_features
        self.selector = SelectFromModel(estimator=self.estimator, threshold=self.threshold, prefit=False,
                                        norm_order=self.norm_order, max_features=self.max_features)

    def _set_estimator(self, estimator=None, paras=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORTED)
            return get_tree_based_estimator(self.problem, estimator_name, paras)
        else:
            raise TypeError("Estimator should be a string.")

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        self.selected_feature_masks = self.selector.get_support()
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

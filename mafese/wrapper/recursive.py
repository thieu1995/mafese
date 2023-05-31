#!/usr/bin/env python
# Created by "Thieu" at 10:14, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.feature_selection import RFE
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils.estimator import get_classifier_for_recursive, get_regressor_for_recursive


class RecursiveSelector(Selector):
    """
    Defines a RecursiveSelector class that hold all RecursiveSelector Feature Selection methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    estimator: str or Estimator instance (from scikit-learn or custom)
        If estimator is str, we are currently support:
            - rf: random forest
            - svm: support vector machine with kernel = 'linear'

        If estimator is Estimator instance: you need to make sure it is has a ``fit`` method that provides
        information about feature importance (e.g. `coef_`, `feature_importances_`).

    n_features : int or float, default=3
        The number of features to select. If `None`, half of the features are selected.
        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to  select.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`  or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of :class:`~sklearn.compose.TransformedTargetRegressor` or
        `named_steps.clf.feature_importances_` in case of class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should return importance for each feature.

    Examples
    --------
    The following example shows how to retrieve the most informative features in the RecursiveSelector FS method

    >>> import pandas as pd
    >>> from mafese.wrapper.recursive import RecursiveSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=5)
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

    SUPPORTED_ESTIMATORS = ["rf", "svm"]

    def __init__(self, problem="classification", estimator="knn", n_features=3, step=1, verbose=0, importance_getter="auto"):
        super().__init__(problem)
        self.estimator = self.set_estimator(estimator)
        self.n_features = n_features
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.selector = RFE(estimator=self.estimator, n_features_to_select=self.n_features,
                            step=self.step, verbose=self.verbose, importance_getter=self.importance_getter)

    def set_estimator(self, estimator=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORTED_ESTIMATORS)
            if self.problem == "classification":
                return get_classifier_for_recursive(estimator_name)
            else:
                return get_regressor_for_recursive(estimator_name)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and \
                (callable(estimator.fit) and callable(estimator.predict)):
            return estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")

    def fit(self, X, y=None):
        self.selector = self.selector.fit(X, y)
        self.selected_feature_masks = self.selector.support_.copy()
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

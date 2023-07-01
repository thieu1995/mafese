#!/usr/bin/env python
# Created by "Thieu" at 10:41, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils.estimator import get_general_estimator


class SequentialSelector(Selector):
    """
    Defines a SequentialSelector class that hold all Forward or Backward Feature Selection methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    estimator: str or Estimator instance (from scikit-learn or custom)
        If estimator is str, we are currently support:
            - knn: k-nearest neighbors
            - svm: support vector machine
            - rf: random forest
            - adaboost: AdaBoost
            - xgb: Gradient Boosting
            - tree: Extra Trees
            - ann: Artificial Neural Network (Multi-Layer Perceptron)

        If estimator is Estimator instance: you need to make sure it is has a ``fit`` method that provides
        information about feature importance (e.g. `coef_`, `feature_importances_`).

    estimator_paras: None or dict, default = None
        The parameters of the estimator, please see the official document of scikit-learn to selected estimator.
        If None, we use the default parameter for selected estimator

    n_features : int or float, default=3
        The number of features to select. If `None`, half of the features are selected.
        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to  select.

    direction : {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.

    tol : float, default=None
        If the score is not incremented by at least `tol` between two consecutive feature additions or removals,
        stop adding or removing. `tol` can be negative when removing features using `direction="backward"`.
        It can be useful to reduce the number of features at the cost of a small decrease in the score.
        `tol` is enabled only when `n_features` is `"auto"`.

    scoring : str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable to evaluate the predictions on the test set.
        NOTE that when using a custom scorer, it should return a single value. If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is either binary or multiclass,
        :class:`StratifiedKFold` is used. In all other cases, :class:`KFold` is used. These splitters are
        instantiated with `shuffle=False` so the splits will be the same across calls.

    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to add or remove, the cross-validation
        procedure is parallel over the folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    Examples
    --------
    The following example shows how to retrieve the most informative features in the Sequential-based (forward, backward) FS method

    >>> import pandas as pd
    >>> from mafese.wrapper.sequential import SequentialSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = SequentialSelector(problem="classification", estimator="knn", n_features=5, direction="forward")
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

    SUPPORT = ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"]

    def __init__(self, problem="classification", estimator="knn", estimator_paras=None, n_features=3,
                 direction="forward", tol=None, scoring=None, cv=5, n_jobs=None):
        super().__init__(problem)
        self.estimator = self._set_estimator(estimator, estimator_paras)
        self.estimator_paras = estimator_paras
        self.n_features = n_features
        self.direction = direction
        self.tol = tol
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.selector = SFS(estimator=self.estimator, n_features_to_select=self.n_features, direction=self.direction)

    def _set_estimator(self, estimator=None, paras=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORT)
            return get_general_estimator(self.problem, estimator_name, paras)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and (callable(estimator.fit) and callable(estimator.predict)):
            return estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")

    def fit(self, X, y=None):
        self.selector = self.selector.fit(X, y)
        self.selected_feature_masks = self.selector.support_.copy()
        self.selected_feature_solution = np.array(self.selected_feature_masks, dtype=int)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

#!/usr/bin/env python
# Created by "Thieu" at 10:43, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC
from mafese.utils import validator


class Selector(ABC):
    """
    Defines an abstract class for Feature Selector.
    """
    name = "Feature Selector"
    SUPPORTED_PROBLEMS = ["classification", "regression"]

    def __init__(self, problem="classification"):
        self.problem = self.set_problem(problem)
        self.selector = None
        self.estimator = None
        self.paras = {}
        self.selected_feature_indexes = []
        self.selected_feature_masks = []
        self.selected_feature_solution = []
        self.epsilon = 1e-8
        self.w = 1e8

    def set_problem(self, problem):
        if type(problem) is not str:
            raise TypeError(f"problem should be string, and is 'classification' or 'regression'.")
        else:
            return validator.check_str("problem", problem, self.SUPPORTED_PROBLEMS)

    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self.selector.fit(X, y)

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        return self.selector.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.selector.fit_transform(X, y, **fit_params)

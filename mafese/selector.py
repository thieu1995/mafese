#!/usr/bin/env python
# Created by "Thieu" at 10:43, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC
from mafese.utils import validator
from mafese.utils.estimator import get_general_estimator
from mafese.utils.evaluator import get_metrics, get_all_classification_metrics, get_all_regression_metrics
from mafese.utils.data_loader import Data


class Selector(ABC):
    """
    Defines an abstract class for Feature Selector.
    """
    name = "Feature Selector"
    SUPPORTED_PROBLEMS = ["classification", "regression"]
    SUPPORTED_ESTIMATORS = ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"]
    SUPPORTED_REGRESSION_METRICS = list(get_all_regression_metrics().keys())
    SUPPORTED_CLASSIFICATION_METRICS = list(get_all_classification_metrics().keys())

    def __init__(self, problem="classification"):
        self.problem = self._set_problem(problem)
        self.selector = None
        self.estimator = None
        self.paras = {}
        self.selected_feature_indexes = []
        self.selected_feature_masks = []
        self.selected_feature_solution = []
        self.epsilon = 1e-8
        self.w = 1e8

    def _set_problem(self, problem):
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

    def evaluate(self, estimator=None, estimator_paras=None, data=None, metrics=None):
        """
        Evaluate the new dataset. We will re-train the estimator with training set
        and  return the metrics of both training and testing set

        Parameters
        ----------
        estimator : str or Estimator instance (from scikit-learn or custom)
            If estimator is str, we are currently support:
                - knn: k-nearest neighbors
                - svm: support vector machine
                - rf: random forest
                - adaboost: AdaBoost
                - xgb: Gradient Boosting
                - tree: Extra Trees
                - ann: Artificial Neural Network (Multi-Layer Perceptron)

            If estimator is Estimator instance: you need to make sure that it has `fit` and `predict` methods

        estimator_paras: None or dict, default = None
            The parameters of the estimator, please see the official document of scikit-learn to selected estimator.
            If None, we use the default parameter for selected estimator

        data : Data, an instance of Data class. It must have training and testing set

        metrics : tuple, list, default = None
            Depend on the regression or classification you are trying to tackle. The supported metrics can be found at:
            https://github.com/thieu1995/permetrics

        Returns
        -------
        metrics_results: dict.
            The metrics for both training and testing set.
        """

        if estimator is None:
            if self.estimator is None:
                raise ValueError("You need to set estimator to evaluate the data.")
            est_ = self.estimator
        elif type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORTED_ESTIMATORS)
            est_ = get_general_estimator(self.problem, estimator_name, estimator_paras)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and (callable(estimator.fit) and callable(estimator.predict)):
            est_ = estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")
        if (metrics is None) or (type(metrics) not in (tuple, list)):
            raise ValueError("You need to pass a tuple/list of performance metrics. See the supported metrics at https://github.com/thieu1995/permetrics")
        if isinstance(data, Data):
            X_train = self.transform(data.X_train)
            X_test = self.transform(data.X_test)
            est_.fit(X_train, data.y_train)
            y_train_pred = est_.predict(X_train)
            y_test_pred = est_.predict(X_test)
            train_result = get_metrics(self.problem, data.y_train, y_train_pred, metrics=metrics, testcase="train")
            test_result = get_metrics(self.problem, data.y_test, y_test_pred, metrics=metrics, testcase="test")
            return {**train_result, **test_result}
        else:
            raise ValueError("'data' should be an instance of Data class.")

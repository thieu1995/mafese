#!/usr/bin/env python
# Created by "Thieu" at 16:22, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils.estimator import get_classifier, get_regressor
from mafese.utils.mealpy_util import get_optimizer_by_name, get_all_optimizers, FeatureSelectionProblem, Optimizer
from mafese.utils import transfer
from mafese.utils.data_loader import Data
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric


class MhaSelector(Selector):
    """
    Defines a MhaSelector class that hold all Metaheuristic-based Feature Selection methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    estimator: str or Estimator instance (from scikit-learn or custom)
        If estimator is str, we are currently support:
            - knn: K-Nearest Neighbors
            - rf: random forest
            - svm: support vector machine

        If estimator is Estimator instance: you need to make sure it is has a ``fit`` method that provides
        information about feature importance (e.g. `coef_`, `feature_importances_`).

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    transfer_func : str or callable function, default="vstf_01"
        The transfer function used to convert solution from float to integer. Current supported list:
            - v-shape transfer function: "vstf_01", "vstf_02", "vstf_03", "vstf_04"
            - s-shape transfer function: "sstf_01", "sstf_02", "sstf_03", "sstf_04"

        If `callable` function, make sure it return a list/tuple/np.ndarray values.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

        - If problem is classification, `None` will be replaced by AS (Accuracy score).
        - If problem is regression, `None` will be replaced by MSE (Mean squared error).

    Examples
    --------
    The following example shows how to retrieve the most informative features in the MhaSelector FS method

    >>> import pandas as pd
    >>> from mafese.wrapper.mha import MhaSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = MhaSelector(problem="classification", estimator="rf", optimizer="BaseGA")
    >>> # find all relevant features - 5 features should be selected
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

    SUPPORTED_ESTIMATORS = ["knn", "rf", "svm"]
    SUPPORTED_TRANSFER_FUNCS = ["vstf_01", "vstf_02", "vstf_03", "vstf_04", "sstf_01", "sstf_02", "sstf_03", "sstf_04", "rtf"]
    SUPPORTED_REG_METRICS = {"MAE": "min", "MSE": "min", "RMSE": "min", "MRE": "min", "MAPE": "min", "MASE": "min",
                             "NSE": "max", "NNSE": "max", "WI": "max", "PCC": "max", "R2s": "max", "R2": "max", "AR2": "max",
                             "CI": "max", "KGE": "max", "VAF": "max", "A10": "max", "A20": "max"}
    SUPPORTED_CLS_METRICS = {"AS": "max", "PS": "max", "NPV": "max", "RS": "max", "F1S": "max", "F2S": "max",
                             "FBS": "max", "SS": "max", "MCC": "max", "JSI": "max", "CKS": "max", "ROC-AUC": "max"}
    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())

    def __init__(self, problem="classification", estimator="knn", optimizer="BaseGA", optimizer_paras=None,
                 transfer_func="vstf_01", obj_name=None):
        super().__init__(problem)
        self.estimator = self.set_estimator(estimator)
        self.optimizer = self.set_optimizer(optimizer, optimizer_paras)
        self.transfer_func_ = self.set_transfer_func(transfer_func)
        self.obj_name = obj_name

    def set_estimator(self, estimator=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORTED_ESTIMATORS)
            if self.problem == "classification":
                return get_classifier(estimator_name)
            else:
                return get_regressor(estimator_name)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and \
                (callable(estimator.fit) and callable(estimator.predict)):
            return estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")

    def set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=100, pop_size=30)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def set_transfer_func(self, transfer_func="vstf_01"):
        if transfer_func is None:
            return getattr(transfer, "vstf_01")
        elif type(transfer_func) is str:
            transfer_func = validator.check_str("transfer_func", transfer_func, self.SUPPORTED_TRANSFER_FUNCS)
            return getattr(transfer, transfer_func)
        elif callable(transfer_func):
            return transfer_func
        else:
            raise TypeError(f"transfer_func needs to be a callable function or a string with valid value belongs to {self.SUPPORTED_TRANSFER_FUNCS}")

    def set_metric(self, metric_name=None, list_supported_metrics=None):
        if type(metric_name) is str:
            return validator.check_str("obj_name", metric_name, list_supported_metrics)
        else:
            raise ValueError(f"obj_name should be a string and belongs to {list_supported_metrics}")

    def fit(self, X, y=None, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        fit_weights : list, tuple or np.ndarray, default = (0.9, 0.1)
            The first weight is for objective value and the second weight is for the number of features

        verbose : int, default = True
            Controls verbosity of output.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) to do the tasks (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.data = Data(X.copy(), y.copy())
        self.data.split_train_test(test_size=0.25)
        lb = [-8, ] * X.shape[1]
        ub = [8, ] * X.shape[1]
        if self.problem == "classification":
            if len(np.unique(y)) == 2:
                self.obj_paras = {"average": "micro"}
            else:
                self.obj_paras = {"average": "weighted"}
            if self.obj_name is None:
                self.obj_name = "AS"
            else:
                self.obj_name = self.set_metric(self.obj_name, self.SUPPORTED_CLS_METRICS)
            minmax = self.SUPPORTED_CLS_METRICS[self.obj_name]
            metric_class = ClassificationMetric
        else:
            self.obj_paras = {"decimal": 4}
            if self.obj_name is None:
                self.obj_name = "MSE"
            else:
                self.obj_name = self.set_metric(self.obj_name, self.SUPPORTED_REG_METRICS)
            minmax = self.SUPPORTED_REG_METRICS[self.obj_name]
            metric_class = RegressionMetric
        fit_sign = -1 if minmax == "max" else 1
        log_to = "console" if verbose else "None"
        prob = FeatureSelectionProblem(lb, ub, minmax, data=self.data,
                                       estimator=self.estimator, transfer_func=self.transfer_func_, obj_name=self.obj_name,
                                       metric_class=metric_class, fit_weights=fit_weights, fit_sign=fit_sign, log_to=log_to,
                                       obj_weights=(1.0, 0.), obj_paras=self.obj_paras)
        best_position, best_fitness = self.optimizer.solve(prob, mode=mode, n_workers=n_workers, termination=termination)
        self.selected_feature_solution = np.array(best_position, dtype=int)
        self.selected_feature_masks = np.where(self.selected_feature_solution == 0, False, True)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

    def transform(self, X):
        return X[:, self.selected_feature_indexes]

    def fit_transform(self, X, y=None, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        self.fit(X, y, fit_weights, verbose, mode, n_workers, termination)
        return self.transform(X)

    def get_best_obj_and_fit(self):
        return {
            "obj": self.optimizer.solution[1][1][1],
            "fit": self.optimizer.solution[1][1][0]
        }

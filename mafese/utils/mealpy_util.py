#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import Problem


class FeatureSelectionProblem(Problem):
    """
    A class to define a feature selection optimization problem.

    Attributes
    ----------
    data : object
        An object containing training and testing datasets.
    estimator : object
        A machine learning model with `fit` and `predict` methods.
    metric_class : object
        A class used to evaluate the performance of the model.
    obj_name : str
        The name of the objective metric to optimize.
    obj_paras : dict
        Parameters for the objective metric.
    fit_weights : tuple
        Weights for combining the objective value and feature selection ratio.
    fit_sign : int
        Sign to determine the direction of optimization (e.g., 1 for maximization, -1 for minimization).

    Methods
    -------
    obj_func(solution)
        Computes the fitness, objective value, and number of selected features for a given solution.
    """
    def __init__(self, bounds=None, minmax=None, data=None, estimator=None, metric_class=None,
                 obj_name=None, obj_paras=None, fit_weights=(0.9, 0.1), fit_sign=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.data = data
        self.estimator = estimator
        self.metric_class = metric_class
        self.obj_name = obj_name
        self.obj_paras = obj_paras
        self.obj_weights = (1, 0, 0)
        self.fit_weights = fit_weights
        self.fit_sign = fit_sign

    def obj_func(self, solution):
        """
        Computes the fitness, objective value, and number of selected features for a given solution.

        Parameters
        ----------
        solution : array-like
            The solution representing selected features.

        Returns
        -------
        list
            A list containing the fitness value, objective value, and number of selected features.
        """
        x = self.decode_solution(solution)["my_var"]    # Please don't change this line, it is used to decode the solution
        cols = np.flatnonzero(x)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        evaluator = self.metric_class(self.data.y_test, y_valid_pred)
        obj = evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name]
        fitness = self.fit_weights[0]*obj + self.fit_weights[1]*self.fit_sign*(np.sum(x) / self.n_dims)
        return [fitness, obj, np.sum(x)]    # Return fitness, objective value, and number of selected features

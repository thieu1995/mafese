#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy import *


class FeatureSelectionProblem(Problem):
    def __init__(self, minmax=None, data=None, estimator=None,
                 obj_name=None, obj_weights=(0.9, 0.1), obj_paras=None, transfer_func="vstf_01", **kwargs):
        bounds = TransferBinaryVar(n_vars=data.X.shape[1], tf_func=transfer_func, lb=-8, ub=8, all_zeros=False, name="my_var")
        super().__init__(bounds, minmax, **kwargs)
        self.data = data
        self.estimator = estimator
        self.obj_name = obj_name
        self.metric_class = metric_class
        self.obj_weights = obj_weights
        self.obj_sign = obj_sign
        self.obj_paras = obj_paras

    def obj_func(self, solution):
        x = self.decode_solution(solution)["my_var"]
        cols = np.flatnonzero(x)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        evaluator = self.metric_class(self.data.y_test, y_valid_pred)
        obj = evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name]
        fitness = self.obj_weights[0]*obj + self.obj_sign * self.obj_weights[1]*(np.sum(solution)/self.n_dims)
        return [fitness, obj]


        return np.sum(x ** 2)

## Now, we define an algorithm, and pass an instance of our *Squared* class as the problem argument.
bound = TransferBinaryVar(n_vars=20, tf_func="vstf_01", lb=-8, ub=8, all_zeros=False, name="my_var")
model = BBO.OriginalBBO(epoch=100, pop_size=20)
g_best = model.solve(problem)




class FeatureSelectionProblem(Problem):
    def __init__(self, lb, ub, minmax, data=None, estimator=None,
                 transfer_func=None, obj_name=None,
                 metric_class=None, fit_weights=(0.9, 0.1), fit_sign=1, obj_paras=None,
                 name="Feature Selection Problem", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.estimator = estimator
        self.transfer_func = transfer_func
        self.obj_name = obj_name
        self.metric_class = metric_class
        self.fit_weights = fit_weights
        self.fit_sign = fit_sign
        self.obj_paras = obj_paras
        self.name = name
        super().__init__(lb, ub, minmax, **kwargs)

    def amend_position(self, position=None, lb=None, ub=None):
        position = self.transfer_func(position)
        cons = np.random.uniform(0, 1, len(position))
        x = np.where(cons < position, 1, 0)
        if np.sum(x) == 0:
            x[np.random.randint(0, len(x))] = 1
        return x

    def fit_func(self, solution):
        cols = np.flatnonzero(solution)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        evaluator = self.metric_class(self.data.y_test, y_valid_pred)
        obj = evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name]
        fitness = self.fit_weights[0]*obj + self.fit_sign * self.fit_weights[1]*(np.sum(solution)/self.n_dims)
        return [fitness, obj]

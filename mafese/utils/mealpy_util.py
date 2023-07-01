#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy import *
import sys, inspect


EXCLUDE_MODULES = ["__builtins__", "current_module", "inspect", "sys"]


def get_all_optimizers():
    cls = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.ismodule(obj) and (name not in EXCLUDE_MODULES):
            for cls_name, cls_obj in inspect.getmembers(obj):
                if inspect.isclass(cls_obj) and issubclass(cls_obj, Optimizer):
                    cls[cls_name] = cls_obj
    del cls['Optimizer']
    return cls


def get_optimizer_by_name(name):
    try:
        cls = get_all_optimizers()[name]
        return cls
    except KeyError:
        print(f"Mafese doesn't support optimizer named: {name}.\n"
              f"Please see the supported Optimizer name from here: https://mealpy.readthedocs.io/en/latest/pages/support.html#classification-table")


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

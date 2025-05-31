#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy import *


class FeatureSelectionProblem(Problem):
    def __init__(self, minmax=None, data=None, estimator=None, metric_class=None,
                 obj_name=None, obj_paras=None, fit_weights=(0.9, 0.1), fit_sign=None,
                 transfer_func="vstf_01", **kwargs):
        bounds = TransferBinaryVar(n_vars=data.X.shape[1], tf_func=transfer_func, lb=-8, ub=8, all_zeros=False, name="my_var")
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
        x = self.decode_solution(solution)["my_var"]
        cols = np.flatnonzero(x)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        evaluator = self.metric_class(self.data.y_test, y_valid_pred)
        obj = evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name]
        fitness = self.fit_weights[0]*obj + self.fit_weights[1]*self.fit_sign*(np.sum(x) / self.n_dims)
        return [fitness, obj, np.sum(x)]

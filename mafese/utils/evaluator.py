#!/usr/bin/env python
# Created by "Thieu" at 06:52, 01/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import validation_curve
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric


def get_metrics(problem, y_true, y_pred, metrics=None, testcase="test"):
    if problem == "regression":
        evaluator = RegressionMetric(y_true, y_pred)
        paras = [{"decimal": 4}, ] * len(metrics)
    else:
        evaluator = ClassificationMetric(y_true, y_pred)
        paras = [{"average": "weighted"}, ] * len(metrics)
    if type(metrics) is dict:
        result = evaluator.get_metrics_by_dict(metrics)
    elif type(metrics) in (tuple, list):
        result = evaluator.get_metrics_by_list_names(metrics, paras)
    else:
        raise ValueError("metrics parameter should be a list or dict")
    final_result = {}
    for key, value in result.items():
        if testcase is None or testcase == "":
            final_result[f"{key}"] = value
        else:
            final_result[f"{key}_{testcase}"] = value
    return final_result

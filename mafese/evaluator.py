#!/usr/bin/env python
# Created by "Thieu" at 07:52, 27/05/2023 ----------%                                                                               
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


def __get_metrics(selector, estimator, X_train, y_train, X_test, y_test, metrics=None):

    def __get_results(problem, y_true, y_pred, testcase="train"):
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
            final_result[f"{key}_{testcase}"] = value
        return final_result

    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    train_result = __get_results(selector.problem, y_train, y_train_pred, testcase="train")
    test_result = __get_results(selector.problem, y_test, y_test_pred, testcase="test")
    return {**train_result, **test_result}


def evaluate(selector, estimator=None, data=None, metrics=None):
    X_train = selector.transform(data.X_train)
    X_test = selector.transform(data.X_test)
    y_train, y_test = data.y_train, data.y_test
    if estimator is None:
        if selector.estimator is None:
            raise ValueError("Your selector doesn't has any Estimator. You need to set up the estimator.")
        elif data is None:
            raise ValueError("You need to pass an Data object. Remember to set up your own X_train, X_test, y_train, y_test in Data object.")
        elif metrics is None:
            raise ValueError("You need to pass a list of performance metrics. See the supported metrics at https://github.com/thieu1995/permetrics")
        else:
            return __get_metrics(selector, selector.estimator, X_train, y_train, X_test, y_test, metrics)
    else:
        if data is None:
            raise ValueError("You need to pass an Data object. Remember to set up your own X_train, X_test, y_train, y_test in Data object.")
        elif metrics is None:
            raise ValueError("You need to pass a list of performance metrics. See the supported metrics at https://github.com/thieu1995/permetrics")
        else:
            return __get_metrics(selector, estimator, X_train, y_train, X_test, y_test, metrics)

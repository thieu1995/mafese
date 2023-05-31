#!/usr/bin/env python
# Created by "Thieu" at 15:43, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression


def get_classifier(name):
    name = name.lower()
    if name == 'knn':
        return KNeighborsClassifier()
    elif name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC()
    else:
        raise ValueError(f"Currently, MAFESE doesn't support: {name} classifier. You can define your own scikit-learn model.")


def get_regressor(name):
    name = name.lower()
    if name == "knn":
        return KNeighborsRegressor()
    elif name == "rf":
        return RandomForestRegressor()
    elif name == "svm":
        return SVR()
    else:
        raise ValueError(f"Currently, we don't support: {name} regressor. You can define your own scikit-learn model.")


def get_classifier_for_recursive(name):
    name = name.lower()
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC(kernel="linear")
    else:
        raise ValueError(f"Currently, MAFESE doesn't support: {name} classifier. You can define your own scikit-learn model.")


def get_regressor_for_recursive(name):
    name = name.lower()
    if name == "rf":
        return RandomForestRegressor()
    elif name == "svm":
        return SVR(kernel="linear")
    else:
        raise ValueError(f"Currently, we don't support: {name} regressor. You can define your own scikit-learn model.")


def get_lasso_based_estimator(problem, name, paras=None):
    if paras is None:
        paras = {}
    name = name.lower()
    if problem == "regression":
        if name == "lasso":
            return Lasso(**paras)
        else:
            raise ValueError(f"For Regression problem, lasso-based method only supports 'lasso' estimator.")
    else:
        if name == "lasso":
            return Lasso(**paras)
        elif name == "lr":
            est = LogisticRegression(**paras)
            est.set_params(**{"penalty": "l1", "solver": "liblinear"})
            return est
        elif name == "svm":
            return LinearSVC(**paras)
        else:
            raise ValueError(f"For Classification problem, lasso-based method supports: 'lasso', 'lr', and 'svm' estimator.")


def get_tree_based_estimator(problem, name, paras=None):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

    if paras is None:
        paras = {}
    name = name.lower()
    if problem == "regression":
        if name == "rf":
            return RandomForestRegressor(**paras)
        elif name == "adaboost":
            return AdaBoostRegressor(**paras)
        elif name == "xgb":
            return GradientBoostingRegressor(**paras)
        elif name == "tree":
            return ExtraTreesRegressor(**paras)
        else:
            raise ValueError(f"For Regression problem, we don't support {name} estimator.")
    else:
        if name == "rf":
            return RandomForestClassifier(**paras)
        elif name == "adaboost":
            return AdaBoostClassifier(**paras)
        elif name == "xgb":
            return GradientBoostingClassifier(**paras)
        elif name == "tree":
            return ExtraTreesClassifier(**paras)
        else:
            raise ValueError(f"For Classification problem, we don't support {name} estimator.")

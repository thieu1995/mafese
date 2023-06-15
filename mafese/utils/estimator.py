#!/usr/bin/env python
# Created by "Thieu" at 15:43, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression, BayesianRidge, Perceptron
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


def get_general_estimator(problem, name, paras=None):
    if paras is None:
        paras = {}
    name = name.lower()
    if problem == "regression":
        if name == "knn":
            return KNeighborsRegressor(**paras)
        elif name == "svm":
            return SVR(**paras)
        elif name == "rf":
            return RandomForestRegressor(**paras)
        elif name == "adaboost":
            return AdaBoostRegressor(**paras)
        elif name == "xgb":
            return GradientBoostingRegressor(**paras)
        elif name == "tree":
            return ExtraTreesRegressor(**paras)
        elif name == "ann":
            return MLPRegressor(**paras)
        else:
            raise ValueError(f"For Regression problem, we don't support: {name} regressor as string. \n"
                             f"You can define your own scikit-learn model.")
    else:
        if name == 'knn':
            return KNeighborsClassifier(**paras)
        elif name == 'svm':
            return SVC(**paras)
        elif name == 'rf':
            return RandomForestClassifier(**paras)
        elif name == "adaboost":
            return AdaBoostClassifier(**paras)
        elif name == "xgb":
            return GradientBoostingClassifier(**paras)
        elif name == "tree":
            return ExtraTreesClassifier(**paras)
        elif name == "ann":
            return MLPClassifier(**paras)
        else:
            raise ValueError(f"For Classification problem, we don't support: {name} classifier as string. \n"
                             f"You can define your own scikit-learn model.")


def get_recursive_estimator(problem, name, paras=None):
    if paras is None:
        paras = {}
    name = name.lower()
    if problem == "regression":
        if name == "rf":
            return RandomForestRegressor(**paras)
        elif name == "svm":
            return SVR(**paras, kernel="linear")
        elif name == "adaboost":
            return AdaBoostRegressor(**paras)
        elif name == "xgb":
            return GradientBoostingRegressor(**paras)
        elif name == "tree":
            return ExtraTreesRegressor(**paras)
        else:
            raise ValueError(f"For Regression problem, with Recursive method, you can't use: {name} regressor as base estimator.")
    else:
        if name == 'rf':
            return RandomForestClassifier(**paras)
        elif name == 'svm':
            est = SVC(**paras)
            est.set_params(**{"kernel": "linear"})
            return est
        elif name == "adaboost":
            return AdaBoostClassifier(**paras)
        elif name == "xgb":
            return GradientBoostingClassifier(**paras)
        elif name == "tree":
            return ExtraTreesClassifier(**paras)
        else:
            raise ValueError(f"For Classification problem, with Recursive method, you can't use: {name} classifier as base estimator.")


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

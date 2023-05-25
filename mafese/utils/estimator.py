#!/usr/bin/env python
# Created by "Thieu" at 15:43, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

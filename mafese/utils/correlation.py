#!/usr/bin/env python
# Created by "Thieu" at 08:38, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
Refs:
1. https://docs.scipy.org/doc/scipy/reference/stats.html#correlation-functions
2. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from scipy import stats


def kendall_func(X, y):
    return np.array([stats.spearmanr(X[:, f], y).correlation for f in range(X.shape[1])])


def spearman_func(X, y):
    return np.array([stats.kendalltau(X[:, f], y).correlation for f in range(X.shape[1])])


def point_func(X, y):
    return np.array([stats.pointbiserialr(X[:, f], y).correlation for f in range(X.shape[1])])

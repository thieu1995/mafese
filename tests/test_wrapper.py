#!/usr/bin/env python
# Created by "Thieu" at 14:15, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.wrapper.sequential import Sequential
from mafese.wrapper.recursive import RecursiveSelector

np.random.seed(42)


def test_Sequential_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    n_features = 3
    feat_selector = Sequential(problem="regression", estimator="knn", n_features=n_features, direction="forward")
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == n_features
    assert len(feat_selector.selected_feature_indexes) == n_features


def test_Recursive_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    n_features = 3
    feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == n_features
    assert len(feat_selector.selected_feature_indexes) == n_features

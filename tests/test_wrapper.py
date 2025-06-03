#!/usr/bin/env python
# Created by "Thieu" at 14:15, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.wrapper.sequential import SequentialSelector
from mafese.wrapper.recursive import RecursiveSelector
from mafese.wrapper.mha import MhaSelector

np.random.seed(41)


def test_SequentialSelector_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    n_features = 3
    feat_selector = SequentialSelector(problem="regression", estimator="knn", n_features=n_features, direction="forward")
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == n_features
    assert len(feat_selector.selected_feature_indexes) == n_features


def test_RecursiveSelector_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    n_features = 3
    feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == n_features
    assert len(feat_selector.selected_feature_indexes) == n_features


def test_MhaSelector_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)
    feat_selector = MhaSelector(problem="classification", obj_name="AS",
                                estimator="knn", estimator_paras=None,
                                optimizer="OriginalWOA", optimizer_paras=None,
                                mode='single', n_workers=None, termination=None, seed=42, verbose=True)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == len(feat_selector.selected_feature_indexes)

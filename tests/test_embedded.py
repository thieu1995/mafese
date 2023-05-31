#!/usr/bin/env python
# Created by "Thieu" at 17:18, 31/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.embedded.lasso import LassoSelector
from mafese.embedded.tree import TreeSelector

np.random.seed(42)


def test_LassoSelector_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    feat_selector = LassoSelector(problem="classification", estimator="lasso", estimator_paras={"alpha": 0.1})
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == len(feat_selector.selected_feature_indexes)


def test_TreeSelector_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    feat_selector = TreeSelector(problem="classification", estimator="tree")
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == len(feat_selector.selected_feature_indexes)

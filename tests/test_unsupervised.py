#!/usr/bin/env python
# Created by "Thieu" at 09:54, 01/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.unsupervised import UnsupervisedSelector

np.random.seed(42)


def test_UnsupervisedSelector_class():
    X = np.random.rand(10, 6)
    y = np.random.randint(0, 2, size=10)
    n_features = 3
    feat_selector = UnsupervisedSelector(problem="classification", method="MCL", n_features=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    assert X_selected.shape[1] == n_features
    assert len(feat_selector.selected_feature_indexes) == n_features

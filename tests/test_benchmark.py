#!/usr/bin/env python
# Created by "Thieu" at 20:21, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese.engineer import Engineer


def test_Benchmark_class():
    ndim = 10
    bounds = np.array([[-15, ] * ndim, [15, ] * ndim]).T
    problem = Engineer()
    problem._bounds = bounds

    x = np.random.uniform(problem.lb, problem.ub)

    assert len(problem.lb) == len(x)
    assert isinstance(problem.lb, np.ndarray)
    assert type(problem.bounds) == np.ndarray
    assert problem.bounds.shape[0] == ndim

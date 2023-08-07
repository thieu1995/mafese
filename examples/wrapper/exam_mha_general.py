#!/usr/bin/env python
# Created by "Thieu" at 17:29, 07/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import MhaSelector

print(MhaSelector.SUPPORT["regression_objective"])
print(len(MhaSelector.SUPPORT["regression_objective"]))

print(MhaSelector.SUPPORT["classification_objective"])
print(len(MhaSelector.SUPPORT["classification_objective"]))

print(MhaSelector.SUPPORTED_CLASSIFICATION_METRICS)
print(MhaSelector.SUPPORTED_REGRESSION_METRICS)

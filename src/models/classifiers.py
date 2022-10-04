#!/usr/bin/env python
# Created by "Thieu" at 07:48, 04/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classifier(name):
    if name.lower() == 'knn':
        return KNeighborsClassifier()
    elif name.lower() == 'rf':
        return RandomForestClassifier()
    elif name.lower() == 'svm':
        return SVC()
    else:
        raise ValueError(f"Currently, we don't support: {name} classifier. You can define it at src/models/classifiers.py")

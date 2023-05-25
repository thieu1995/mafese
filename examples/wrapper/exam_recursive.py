#!/usr/bin/env python
# Created by "Thieu" at 17:03, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


def get_dataset(data_type="classification"):
    if data_type == "classification":
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        print(X.shape)
        print(f"X: {X[:1]}, y: {y[:1]}")
        return X, y
    else:
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)       # Load the diabetes dataset
        print(X.shape)
        print(f"X: {X[:1]}, y: {y[:1]}")
        return X, y


def get_recursive_from_scikit_learn(data_type, X, y):
    ## Using scikit-learn library
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVC, SVR

    if data_type == "classification":
        estimator = SVC(kernel="linear")
    else:
        estimator = SVR(kernel="linear")
    feat_selector = RFE(estimator=estimator, n_features_to_select=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print("=============Using scikit-learn library===============")
    print(X_selected.shape)
    print(X_selected[:1])


def get_recursive_from_mafese(data_type, X, y):
    ## Using Mafese library
    from mafese.wrapper.recursive import RecursiveSelector

    print("=============Using Mafese library===============")
    feat_selector = RecursiveSelector(problem=data_type, estimator="rf", n_features=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)


data_type = "regression"    # classification
X, y = get_dataset(data_type)
get_recursive_from_scikit_learn(data_type, X, y)
get_recursive_from_mafese(data_type, X, y)

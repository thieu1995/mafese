#!/usr/bin/env python
# Created by "Thieu" at 16:03, 31/05/2023 ----------%                                                                               
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


def get_lasso_from_scikit_learn(data_type, X, y):
    ## Using scikit-learn library
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso
    from sklearn.svm import LinearSVC

    if data_type == "classification":
        estimator = LinearSVC(C=0.01, penalty="l1", dual=False)
    else:
        estimator = Lasso(alpha=0.1)
    feat_selector = SelectFromModel(estimator=estimator)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print("=============Using scikit-learn library===============")
    print(X_selected.shape)
    print(X_selected[:1])


def get_lasso_from_mafese(data_type, X, y):
    ## Using Mafese library
    from mafese.embedded.lasso import LassoSelector

    print("=============Using Mafese library===============")
    if data_type == "regression":
        feat_selector = LassoSelector(problem=data_type, estimator="lasso", estimator_paras={"alpha": 0.1})
    else:
        feat_selector = LassoSelector(problem=data_type, estimator="svm")
    print(feat_selector.SUPPORT)

    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)

    ## Set up evaluating methods
    results = feat_selector.evaluate(estimator="svm", data=(X, y), metrics=["RMSE", "MSE", "MAPE"])
    print(results)


data_type = "regression"    # classification
X, y = get_dataset(data_type)
get_lasso_from_scikit_learn(data_type, X, y)
get_lasso_from_mafese(data_type, X, y)

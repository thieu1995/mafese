#!/usr/bin/env python
# Created by "Thieu" at 22:55, 31/05/2023 ----------%                                                                               
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
        from sklearn.metrics import mean_squared_error, r2_score

        # Load the diabetes dataset
        X, y = load_diabetes(return_X_y=True)
        print(X.shape)
        print(f"X: {X[:1]}, y: {y[:1]}")
        return X, y


def get_unsupervised_from_mafese(data_type, method, X, y):
    ## Using Mafese library
    from mafese.unsupervised import UnsupervisedSelector

    print("=============Using Mafese library===============")
    feat_selector = UnsupervisedSelector(problem=data_type, method=method, threshold=10)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)
    print(feat_selector.support_values)


data_type = "regression"
method = "MCL"      # MCL
X, y = get_dataset(data_type)
get_unsupervised_from_mafese(data_type, method, X, y)

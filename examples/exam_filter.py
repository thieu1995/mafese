#!/usr/bin/env python
# Created by "Thieu" at 21:01, 24/05/2023 ----------%                                                                               
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


def get_filter_from_scikit_learn(data_type, X, y):
    ## Using scikit-learn library
    from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, \
        r_regression, f_regression, mutual_info_regression, SelectKBest

    if data_type == "regression":
        feat_selector = SelectKBest(r_regression, k=3)
    else:
        feat_selector = SelectKBest(f_classif, k=3)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print("=============Using scikit-learn library===============")
    print(X_selected.shape)
    print(X_selected[:1])


def get_filter_from_mafese(data_type, method, X, y):
    ## Using Mafese library
    from mafese.filter import FilterSelector

    print("=============Using Mafese library===============")
    feat_selector = FilterSelector(problem=data_type, method=method, n_features=2)
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)


data_type = "classification"
method = "KENDALL"
X, y = get_dataset(data_type)
get_filter_from_scikit_learn(data_type, X, y)
get_filter_from_mafese(data_type, method, X, y)

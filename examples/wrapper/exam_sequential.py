#!/usr/bin/env python
# Created by "Thieu" at 16:41, 24/05/2023 ----------%                                                                               
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

        X, y = load_diabetes(return_X_y=True)               # Load the diabetes dataset
        print(X.shape)
        print(f"X: {X[:1]}, y: {y[:1]}")
        return X, y


def get_sequential_from_scikit_learn(data_type, X, y):
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    ## Using scikit-learn library
    print("=============Using scikit-learn library===============")
    if data_type == "regression":
        estimator = KNeighborsRegressor(n_neighbors=5)
    else:
        estimator = KNeighborsClassifier(n_neighbors=5)
    feat_selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=3, direction="forward")
    feat_selector.fit(X, y)
    X_selected = feat_selector.transform(X)
    print(X_selected.shape)
    print(X_selected[:1])


def get_sequential_from_mafese(data_type, estimator, X, y):
    from mafese import Data, SequentialSelector

    data = Data(X, y)
    data.split_train_test(test_size=0.2, inplace=True)

    ## Using Mafese library
    print("=============Using Mafese library===============")
    feat_selector = SequentialSelector(problem=data_type, estimator=estimator, n_features=3, direction="forward")
    feat_selector.fit(data.X_train, data.y_train)
    X_selected = feat_selector.transform(data.X_train)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)

    ## Set up evaluating method
    results = feat_selector.evaluate(estimator="svm", data=data, metrics=["AS", "PS", "RS"])
    print(results)


data_type = "classification"        # classification
X, y = get_dataset(data_type)
get_sequential_from_scikit_learn(data_type, X, y)

estimator = "rf"
get_sequential_from_mafese(data_type, estimator, X, y)

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
    from mafese import Data, UnsupervisedSelector

    data = Data(X, y)
    data.split_train_test(test_size=0.2, inplace=True)

    print("=============Using Mafese library===============")
    feat_selector = UnsupervisedSelector(problem=data_type, method=method, n_features=5, threshold=10)
    print(feat_selector.SUPPORT)
    feat_selector.fit(data.X_train, data.y_train)
    X_selected = feat_selector.transform(data.X_train)
    print(X_selected.shape)
    print(X_selected[:1])
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)
    print(feat_selector.support_values)

    ## Set up evaluating methods
    results = feat_selector.evaluate(estimator="svm", data=data, metrics=["RMSE", "MSE", "MAPE"])
    print(results)


data_type = "regression"
method = "MAD"      # MCL, DR, MAD, VAR
X, y = get_dataset(data_type)
get_unsupervised_from_mafese(data_type, method, X, y)

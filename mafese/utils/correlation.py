#!/usr/bin/env python
# Created by "Thieu" at 08:38, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
Refs:
1. https://docs.scipy.org/doc/scipy/reference/stats.html#correlation-functions
2. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
"""

import numpy as np
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from scipy import stats


def chi2_func(X, y):
    return chi2(X, y)[0]


def f_classification_func(X, y):
    return f_classif(X, y)[0]


def f_regression_func(X, y, center=True, force_finite=True):
    return f_regression(X, y, center=center, force_finite=force_finite)[0]


def kendall_func(X, y):
    return np.array([stats.kendalltau(X[:, f], y).correlation for f in range(X.shape[1])])


def spearman_func(X, y):
    return np.array([stats.spearmanr(X[:, f], y).correlation for f in range(X.shape[1])])


def point_func(X, y):
    return np.array([stats.pointbiserialr(X[:, f], y).correlation for f in range(X.shape[1])])


def select_bests(importance_scores=None, n_features=3):
    """
    Select features according to the k highest scores or percentile of the highest scores.

    Parameters
    ----------
    importance_scores : array-like of shape (n_features,)
        Scores of features.

    n_features : int, float. default=3
        Number of selected features.

        - If `float`, it should be in range of (0, 1). That represent the percentile of the highest scores.
        - If `int`, it should be in range of (1, N-1). N is total number of features in your dataset.

    Returns
    -------
    mask: Number of top features to select.

    """
    max_features = len(importance_scores)
    if type(n_features) in (int, float):
        if type(n_features) is float:
            if 0 < n_features < 1:
                n_features = int(n_features * max_features) + 1
            else:
                raise ValueError("n_features based on percentile should has value in range (0, 1).")
        if n_features < 1 or n_features > max_features:
            raise ValueError(f"n_features should has value in range [1, {max_features}].")
    else:
        raise ValueError("n_features should be int if based on k highest scores, or float if based on percentile of highest scores.")
    mask = np.zeros(importance_scores.shape, dtype=bool)
    mask[np.argsort(importance_scores, kind="mergesort")[-n_features:]] = 1
    return mask


def relief_func(X, y, n_neighbors=5, n_bins=10, problem="classification", normalized=True, **kwargs):
    """
    Performs Relief feature selection on the input dataset X and target variable y.
    Returns a vector of feature importance scores.

    Parameters
    ----------
    X : numpy array
        Input dataset of shape (n_samples, n_features).

    y : numpy array
        Target variable of shape (n_samples,).

    n_neighbors : int, default=5
        Number of neighbors to use for computing feature importance scores.

    n_bins : int, default=10
        Number of bins to use for discretizing the target variable in regression problems.

    problem : str
        The problem of dataset, either regression or classification
        If `regression`, discretize the target variable into n_bins classes

    normalized : bool, default=True
        Normalize feature importance scores by the number of instances in the dataset

    Returns
    -------
    importance score : np.ndarray
        Vector of feature importance scores, with shape (n_features,).
    """
    # Initialize feature importance scores to zero
    importance_scores = np.zeros(X.shape[1])

    # Regression problem: discretize the target variable into n_bins classes
    if problem == "regression":
        y_bins = np.linspace(np.min(y), np.max(y), n_bins)
        y = np.digitize(y, y_bins) - 1

    # Compute distance matrix between instances in the dataset
    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))

    # Loop over instances in the dataset
    for i in range(X.shape[0]):
        # Get the target value of the current instance
        target_value = y[i]

        # Find the indices of the n_neighbors nearest instances with different target labels
        indices = np.argsort(dist_matrix[i])
        neighbors = []
        for j in range(len(indices)):
            if len(neighbors) == n_neighbors:
                break
            if y[indices[j]] != target_value:
                neighbors.append(indices[j])

        # Update feature importance scores based on the distances to the nearest neighbors
        for j in range(X.shape[1]):
            diff = np.abs(X[i, j] - X[neighbors, j])
            importance_scores[j] += np.sum(diff) / n_neighbors

    # Normalize feature importance scores by the number of instances in the dataset
    if normalized:
        importance_scores /= X.shape[0]
    return importance_scores


def relief_f_func(X, y, n_neighbors=5, n_bins=10, problem="classification", normalized=True, **kwargs):
    """
    Performs Relief-F feature selection on the input dataset X and target variable y.
    Returns a vector of feature importance scores

    Parameters
    ----------
    X : numpy array
        Input dataset of shape (n_samples, n_features).

    y : numpy array
        Target variable of shape (n_samples,).

    n_neighbors : int, default=5
        Number of neighbors to use for computing feature importance scores.

    n_bins : int, default=10
        Number of bins to use for discretizing the target variable in regression problems.

    problem : str
        The problem of dataset, either regression or classification
        If `regression`, discretize the target variable into n_bins classes

    normalized : bool, default=True
        Normalize feature importance scores by the number of instances in the dataset

    Returns
    -------
    importance score : np.ndarray
        Vector of feature importance scores, with shape (n_features,).
    """
    # Initialize feature importance scores to zero for each class
    n_features = X.shape[1]
    # Regression problem: discretize the target variable into n_bins classes
    if problem == "regression":
        y_bins = np.linspace(np.min(y), np.max(y), n_bins)
        y = np.digitize(y, y_bins) - 1
    n_classes = len(np.unique(y))
    importance_scores = np.zeros((n_features, n_classes))

    # Compute distance matrix between instances in the dataset
    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))

    # Loop over instances in the dataset
    for i in range(X.shape[0]):
        # Get the target value of the current instance
        target_value = y[i]

        # Find the indices of the n_neighbors nearest instances with different target labels
        indices = np.argsort(dist_matrix[i])
        neighbors_diff = []
        neighbors_same = []
        for j in range(len(indices)):
            if len(neighbors_diff) == n_neighbors and len(neighbors_same) == n_neighbors:
                break
            if y[indices[j]] != target_value:
                neighbors_diff.append(indices[j])
            else:
                neighbors_same.append(indices[j])

        # Update feature importance scores based on the distances to the nearest neighbors
        for j in range(X.shape[1]):
            diff = np.abs(X[i, j] - X[neighbors_diff, j])
            importance_scores[j, target_value] += np.sum(diff) / n_neighbors
            same = np.abs(X[i, j] - X[neighbors_same, j])
            importance_scores[j, target_value] -= np.sum(same) / n_neighbors

    # Combine feature importance scores for each class using a weighted average based on the frequency of each class
    class_freq = np.bincount(y) / y.shape[0]
    importance_scores_weighted = np.dot(importance_scores, class_freq)

    # Normalize feature importance scores by the number of instances in the dataset
    if normalized:
        importance_scores_weighted /= X.shape[0]
    return importance_scores_weighted


def vls_relief_f_func(X, y, n_neighbors=5, n_bins=10, problem="classification", normalized=True, **kwargs):
    """
    Performs Very Large Scale ReliefF feature selection on the input dataset X and target variable y.
    Returns a vector of feature importance scores

    Parameters
    ----------
    X : numpy array
        Input dataset of shape (n_samples, n_features).

    y : numpy array
        Target variable of shape (n_samples,).

    n_neighbors : int, default=5
        Number of neighbors to use for computing feature importance scores.

    n_bins : int, default=10
        Number of bins to use for discretizing the target variable in regression problems.

    problem : str
        The problem of dataset, either regression or classification
        If `regression`, discretize the target variable into n_bins classes

    normalized : bool, default=True
        Normalize feature importance scores by the number of instances in the dataset

    Returns
    -------
    importance score : np.ndarray
        Vector of feature importance scores, with shape (n_features,).
    """
    n_samples, n_features = X.shape
    # Regression problem: discretize the target variable into n_bins classes
    if problem == "regression":
        y_bins = np.linspace(np.min(y), np.max(y), n_bins)
        y = np.digitize(y, y_bins) - 1
    relevance_scores = np.zeros(n_features)
    redundancy_scores = np.zeros(n_features)

    for i in range(n_samples):
        # Randomly select k neighbors
        neighbors = np.random.choice(n_samples, n_neighbors, replace=False)

        for j in range(n_features):
            feature_values = X[neighbors, j]

            # Calculate relevance score
            relevant_diff = np.abs(feature_values - X[i, j])
            relevance_scores[j] += np.sum(relevant_diff * (y[neighbors] != y[i]))

            # Calculate redundancy score
            redundant_diff = np.abs(feature_values - feature_values[:, np.newaxis])
            redundancy_scores[j] += np.sum(redundant_diff)

    # Normalize the scores
    relevance_scores /= (n_samples * n_neighbors)
    redundancy_scores /= (n_samples * n_neighbors * (n_samples - 1))

    # Compute the feature importance by subtracting redundancy from relevance
    feature_importance = relevance_scores - redundancy_scores

    # Normalize feature importance scores by the number of instances in the dataset
    if normalized:
        feature_importance /= X.shape[0]
    return feature_importance

#!/usr/bin/env python
# Created by "Thieu" at 15:35, 01/06/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mafese import get_dataset, MhaSelector, FeatureSelectionProblem


# Define a custom callback objective function
def my_custom_problem(**kwargs):

    class MyProblem(FeatureSelectionProblem):
        # Please check out the FeatureSelectionProblem class in mafese/utils/mealpy_util.py to know which attributes are existing in the object

        def obj_func(self, solution):
            # Decode the solution to get the selected features
            x = self.decode_solution(solution)["my_var"]  # Please don't change this line, it is used to decode the solution
            cols = np.flatnonzero(x)  # Get columns where features are selected

            # In self.data object we have train and test data, we can just use them directly

            # Fit the estimator on the selected features
            self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)

            # Predict on the test set
            y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])

            # Evaluate using a custom metric (e.g., F1 score)
            evaluator = self.metric_class(self.data.y_test, y_valid_pred)
            obj = evaluator.get_metric_by_name("NPV", paras=self.obj_paras)["NPV"]  # Change "NPV" to your desired metric name

            # Calculate fitness
            fitness = self.fit_weights[0] * obj + self.fit_weights[1] * self.fit_sign * (np.sum(x) / self.n_dims)
            return [fitness, obj, int(np.sum(x))]  # Return fitness, objective value, and number of selected features

    return MyProblem(**kwargs)


data = get_dataset("aniso")
data.split_train_test(test_size=0.2)

selector = MhaSelector(problem="classification", obj_name="F1S",
                       estimator="svm", estimator_paras=None,
                       optimizer="BaseGA", optimizer_paras={"epoch": 100, "pop_size": 20, "name": "GA"},
                       mode='single', n_workers=None, termination=None,
                       seed=42, verbose=True)
selector.fit(data.X_train, data.y_train, fs_problem=my_custom_problem)

# Transform test data
X_selected = selector.transform(data.X_test)
print(f"Original Dataset: {data.X_train.shape}")
print(f"Selected dataset: {X_selected.shape}")

# Get some information
print(selector.get_best_information())
print(selector.selected_feature_masks)
print(selector.selected_feature_solution)
print(selector.selected_feature_indexes)

# Predict with new selected features
res1 = selector.evaluate(estimator=None, estimator_paras=None, data=data, metrics=["AS", "PS", "RS"])
# AS: Accuracy score, PS: precision score, RS: recall score
print(res1)



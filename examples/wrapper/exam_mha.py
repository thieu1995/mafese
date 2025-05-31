#!/usr/bin/env python
# Created by "Thieu" at 17:03, 31/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import get_dataset, MhaSelector

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)


selector = MhaSelector(problem="classification", obj_name="F1S",
                       estimator="knn", estimator_paras=None,
                       optimizer="BaseGA", optimizer_paras={"epoch": 100, "pop_size": 20, "name": "GA"},
                       mode='single', n_workers=None, termination=None,
                       seed=None, verbose=True)
selector.fit(data.X_train, data.y_train)

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



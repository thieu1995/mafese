#!/usr/bin/env python
# Created by "Thieu" at 08:57, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import get_dataset, MhaSelector
from sklearn.svm import SVC


data = get_dataset("ecoli")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)            # (361, 279) (91, 279)

feat_selector = MhaSelector(problem="classification", estimator="knn",
                            optimizer="OriginalTLO", optimizer_paras={"epoch": 50, "pop_size": 30},
                            transfer_func="vstf_01", obj_name="AS")
feat_selector.fit(data.X_train, data.y_train, fit_weights=(0.9, 0.1), verbose=True)
X_selected = feat_selector.transform(data.X_train)
print(feat_selector.get_best_obj_and_fit())             # {'obj': 0.83504, 'fit': 0.7424429876435843}
print(f"Original Dataset: {data.X_train.shape}")        # Original Dataset: (361, 279)
print(f"Selected dataset: {X_selected.shape}")          # Selected dataset: (361, 31)

print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)
print(feat_selector.selected_feature_indexes)

results = feat_selector.evaluate(estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
print(results)
# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}

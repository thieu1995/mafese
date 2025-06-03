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

feat_selector = MhaSelector(problem="classification", obj_name="AS",
                            estimator="knn", estimator_paras=None,
                            optimizer="OriginalTLO", optimizer_paras={"epoch": 50, "pop_size": 30},
                            mode='single', n_workers=None, termination=None, seed=None, verbose=True)
feat_selector.fit(data.X_train, data.y_train, fit_weights=(0.9, 0.1))
X_selected = feat_selector.transform(data.X_train)
print(feat_selector.get_best_information())             # {'obj': 0.83504, 'fit': 0.7424429876435843, 'n_columns': 2}
print(f"Original Dataset: {data.X_train.shape}")        # Original Dataset: (361, 279)
print(f"Selected dataset: {X_selected.shape}")          # Selected dataset: (361, 31)

print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)
print(feat_selector.selected_feature_indexes)

results = feat_selector.evaluate(estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
print(results)
# {'AS_train': 0.9310258409445311, 'PS_train': 0.7407825454641211, 'RS_train': 0.7985074626865671, 'AS_test': 0.9167387543252595, 'PS_test': 0.7893032212885155, 'RS_test': 0.8088235294117647}

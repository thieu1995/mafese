#!/usr/bin/env python
# Created by "Thieu" at 08:57, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese.wrapper.mha import MhaSelector
from mafese import get_dataset


data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)

feat_selector = MhaSelector(problem="classification", estimator="knn",
                            optimizer="BaseGA", optimizer_paras=None,
                            transfer_func="vstf_01", obj_name="AS")
feat_selector.fit(data.X_train, data.y_train, fit_weights=(0.9, 0.1), verbose=True)
X_selected = feat_selector.transform(data.X_test)
print(feat_selector.get_best_obj_and_fit())             # {'obj': 0.94139, 'fit': 0.8474085293158468}
print(f"Original Dataset: {data.X_train.shape}")        # Original Dataset: (361, 279)
print(f"Selected dataset: {X_selected.shape}")          # Selected dataset: (91, 20)

print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)
print(feat_selector.selected_feature_indexes)

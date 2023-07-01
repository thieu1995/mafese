#!/usr/bin/env python
# Created by "Thieu" at 07:35, 30/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import get_dataset
from mafese.wrapper.mha import MultiMhaSelector

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)

list_optimizers = ("OriginalWOA", "OriginalGWO", "OriginalTLO", "OriginalGSKA")
list_paras = [{"epoch": 50, "pop_size": 30}, ]*4
feat_selector = MultiMhaSelector(problem="classification", estimator="knn",
                            list_optimizers=list_optimizers, list_optimizer_paras=list_paras,
                            transfer_func="vstf_01", obj_name="AS")

feat_selector.fit(data.X_train, data.y_train, n_trials=3, n_jobs=3, verbose=False)
feat_selector.export_boxplot_figures()
feat_selector.export_convergence_figures()

print(feat_selector.transform(data.X_train, trial=2, model="OriginalGSKA"))

print(feat_selector.evaluate(estimator="svm", data=data, metrics=["AS", "PS", "RS"], trial=2, model="OriginalGWO"))

print(feat_selector.evaluate(estimator="knn", data=data, metrics=["AS", "PS", "RS"], all_models=True))

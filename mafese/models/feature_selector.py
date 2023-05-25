#!/usr/bin/env python
# Created by "Thieu" at 22:28, 03/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem
from src.config import Config


class FeatureSelector(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Feature Selection Problem", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def decode_solution(self, solution):
        pos = solution.astype(int)
        if np.all((pos == 0)):
            pos[np.random.randint(0, len(pos))] = 1

        cols = np.flatnonzero(pos)
        X_train = self.data.X_train[:, cols]
        X_test = self.data.X_test[:, cols]
        return {
            "X_train": X_train,
            "X_test": X_test,
            "position": pos
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        self.data.classifier.fit(structure["X_train"], self.data.y_train)
        return self.data.classifier

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(structure["X_test"])

        ## Save confusion matrix
        if Config.DRAW_CONFUSION_MATRIX:
            plot_confusion_matrix(self.data.classifier, structure["X_test"], self.data.y_test)
            plt.savefig('confusion_matrix.png')
            plt.title('Confusion Matrix')
            plt.show()

        evaluator = ClassificationMetric(self.data.y_test, y_pred, decimal=6)
        dict_paras = {"AS": {"average": Config.AVERAGE_METRIC}, # accuracy_score
                      "PS": {"average": Config.AVERAGE_METRIC}, # precision_score
                      "RS": {"average": Config.AVERAGE_METRIC},  # recall_score
                      "F1S": {"average": Config.AVERAGE_METRIC},  # f1_score
                      }
        return evaluator.get_metrics_by_dict(dict_paras)

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        if Config.PRINT_ALL:
            print(fitness)
        return list(fitness.values())  # Metrics return: [accuracy, precision, recall, f1]

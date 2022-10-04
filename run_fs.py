#!/usr/bin/env python
# Created by "Thieu" at 06:12, 04/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import train_test_split
import numpy as np
from src.models.feature_selector import FeatureSelector
from src.models.optimizers import get_optimizer
from src.models.classifiers import get_classifier
from src.utils.data_util import get_dataset
from src.config import Config


if __name__ == "__main__":

    ## 1. Set up the data object
    data = get_dataset(Config.DATASET_NAME)
    train_X, test_X, train_Y, test_Y = train_test_split(data.features, data.labels,
                                        stratify=data.labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
    data.X_train = train_X
    data.X_test = test_X
    data.y_train = train_Y
    data.y_test = test_Y
    print(f"y_label: {test_Y}")

    ## 2. Set up the classifier
    data.classifier = get_classifier(Config.CLASSIFIER)

    ## 3. Define problem
    n_features = data.features.shape[1]
    n_labels = len(np.unique(data.labels))
    LOWER_BOUND = [0, ] * n_features
    UPPER_BOUND = [1.99, ] * n_features
    problem = FeatureSelector(lb=LOWER_BOUND, ub=UPPER_BOUND, minmax=Config.MIN_MAX_PROBLEM,
                              data=data, name=f"Feature Select for {Config.CLASSIFIER} classifier",
                              save_population=False, log_to="console", obj_weights=Config.OBJ_WEIGHTS)

    ## 4. Define algorithm and run
    model = get_optimizer(Config.OPTIMIZER)(epoch=Config.EPOCH, pop_size=Config.POP_SIZE)
    best_position, best_fitness = model.solve(problem)
    print(f"Best features: {problem.decode_solution(best_position)['position']}, \n"
          f"Best accuracy: {best_fitness}")

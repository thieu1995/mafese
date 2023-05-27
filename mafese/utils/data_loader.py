#!/usr/bin/env python
# Created by "Thieu" at 23:33, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class Data:
    #  structure for the data (not dataset)
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def get_dataset(dataset_name):
    # function to retrieve the data
    dir_root = f"{Path(__file__).parent.parent.__str__()}/data"
    list_paths = Path(dir_root).glob("*.csv")
    list_datasets = [pathfile.name[:-4] for pathfile in list_paths]

    if dataset_name not in list_datasets:
        print(f"MAFESE currently does not have '{dataset_name}' data in its database....")
        display = input("Enter 1 to see the available datasets: ") or 0
        if display:
            for idx, dataset in enumerate(list_datasets):
                print(f"{idx + 1}: {dataset}")
    else:
        df = pd.read_csv(f"{dir_root}/{dataset_name}.csv", header=None)
        data = Data(np.array(df.iloc[:, 0:-1]), np.array(df.iloc[:, -1]))
        print(f"Requested dataset: {dataset_name} found and loaded...")
        return data

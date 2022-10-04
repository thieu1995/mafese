#!/usr/bin/env python
# Created by "Thieu" at 23:33, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import numpy as np
from pathlib import Path


class Data:
    #  structure for the data (not dataset)
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.classifier = None


def get_dataset(dataset_name):
    # function to retrieve the data
    dir_root = f"{Path(__file__).parent.parent.parent.__str__()}/data"
    list_paths = Path(dir_root).glob("*.csv")
    list_datasets = [pathfile.name[:-4] for pathfile in list_paths]

    if dataset_name not in list_datasets:
        print(f"MHA-FS currently does not have {dataset_name} in its database....")
        display = input("Enter 1 to see the available datasets: ") or 0
        if display:
            for idx, dataset in enumerate(list_datasets):
                print(f"{idx + 1}: {dataset}")
        exit(0)
    else:
        df = pd.read_csv(f"{dir_root}/{dataset_name}.csv", header=None)
        data = Data(features=np.array(df.iloc[:, 0:-1]), labels=np.array(df.iloc[:, -1]))
        print(f"Requested dataset: {dataset_name} found and loaded...")
        # print(df.head(10))
        return data

#
# if __name__ == '__main__':
#     data = get_dataset('WaveformEW1')


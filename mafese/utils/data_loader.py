#!/usr/bin/env python
# Created by "Thieu" at 23:33, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from mafese.utils.encoder import LabelEncoder
from mafese.utils.scaler import DataTransformer


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray
        The labels of your data
    """

    SUPPORT = {
        "scaler": list(DataTransformer.SUPPORTED_SCALERS.keys())
    }

    def __init__(self, X=None, y=None, name="Unknown"):
        self.X = X
        self.y = y
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    @staticmethod
    def scale(X, scaling_methods=('standard', ), list_dict_paras=None):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if X.ndim >= 3:
            raise TypeError(f"Invalid X data type. It should be array-like with shape (n samples, m features)")
        scaler = DataTransformer(scaling_methods=scaling_methods, list_dict_paras=list_dict_paras)
        data = scaler.fit_transform(X)
        return data, scaler

    @staticmethod
    def encode_label(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise TypeError(f"Invalid y data type. It should be a vector / array-like with shape (n samples,)")
        scaler = LabelEncoder()
        data = scaler.fit_transform(y)
        return data, scaler

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return self


def get_dataset(dataset_name):
    """
    Helper function to retrieve the data

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    data: Data
        The instance of Data class, that hold X and y variables.
    """
    dir_root = f"{Path(__file__).parent.parent.__str__()}/data"
    list_path_reg = Path(f"{dir_root}/reg").glob("*.csv")
    list_path_cls = Path(f"{dir_root}/cls").glob("*.csv")
    reg_list = [pf.name[:-4] for pf in list_path_reg]
    cls_list = [pf.name[:-4] for pf in list_path_cls]
    list_datasets = reg_list + cls_list

    if dataset_name not in list_datasets:
        print(f"MAFESE currently does not have '{dataset_name}' data in its database....")
        display = input("Enter 1 to see the available datasets: ") or 0
        if display:
            print("+ For classification problem. We support datasets:")
            for idx, dataset in enumerate(cls_list):
                print(f"\t{idx + 1}: {dataset}")
            print("+ For regression problem. We support datasets:")
            for idx, dataset in enumerate(reg_list):
                print(f"\t{idx + 1}: {dataset}")
    else:
        if dataset_name in reg_list:
            df = pd.read_csv(f"{dir_root}/reg/{dataset_name}.csv", header=None)
            data_type = "REGRESSION"
        else:
            df = pd.read_csv(f"{dir_root}/cls/{dataset_name}.csv", header=None)
            data_type = "CLASSIFICATION"
        data = Data(np.array(df.iloc[:, 0:-1]), np.array(df.iloc[:, -1]))
        print(f"Requested {data_type} dataset: {dataset_name} found and loaded!")
        return data

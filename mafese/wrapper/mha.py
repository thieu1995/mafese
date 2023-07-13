#!/usr/bin/env python
# Created by "Thieu" at 16:22, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import concurrent.futures as parallel
from functools import partial
import os
from mafese.selector import Selector
from mafese.utils import validator
from mafese.utils.estimator import get_general_estimator
from mafese.utils.mealpy_util import get_optimizer_by_name, get_all_optimizers, FeatureSelectionProblem, Optimizer
from mafese.utils import transfer
from mafese.utils.data_loader import Data
from mafese.utils.evaluator import get_metrics
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


class MhaSelector(Selector):
    """
    Defines a MhaSelector class that hold all Metaheuristic-based Feature Selection methods for feature selection problems

    Parameters
    ----------
    problem: str, default = "classification"
        The problem you are trying to solve (or type of dataset), "classification" or "regression"

    estimator: str or Estimator instance (from scikit-learn or custom)
        If estimator is str, we are currently support:
            - knn: k-nearest neighbors
            - svm: support vector machine
            - rf: random forest
            - adaboost: AdaBoost
            - xgb: Gradient Boosting
            - tree: Extra Trees
            - ann: Artificial Neural Network (Multi-Layer Perceptron)

        If estimator is Estimator instance: you need to make sure that it has `fit` and `predict` methods

    estimator_paras: None or dict, default = None
        The parameters of the estimator, please see the official document of scikit-learn to selected estimator.
        If None, we use the default parameter for selected estimator

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    transfer_func : str or callable function, default="vstf_01"
        The transfer function used to convert solution from float to integer. Current supported list:
            - v-shape transfer function: "vstf_01", "vstf_02", "vstf_03", "vstf_04"
            - s-shape transfer function: "sstf_01", "sstf_02", "sstf_03", "sstf_04"

        If `callable` function, make sure it return a list/tuple/np.ndarray values.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

        - If problem is classification, `None` will be replaced by AS (Accuracy score).
        - If problem is regression, `None` will be replaced by MSE (Mean squared error).

    Examples
    --------
    The following example shows how to retrieve the most informative features in the MhaSelector FS method

    >>> import pandas as pd
    >>> from mafese.wrapper.mha import MhaSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define mafese feature selection method
    >>> feat_selector = MhaSelector(problem="classification", estimator="rf", optimizer="BaseGA")
    >>> # find all relevant features - 5 features should be selected
    >>> feat_selector.fit(X, y)
    >>> # check selected features - True (or 1) is selected, False (or 0) is not selected
    >>> print(feat_selector.selected_feature_masks)
    array([ True, True, True, False, False, True, False, False, False, True])
    >>> print(feat_selector.selected_feature_solution)
    array([ 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    >>> # check the index of selected features
    >>> print(feat_selector.selected_feature_indexes)
    array([ 0, 1, 2, 5, 9])
    >>> # call transform() on X to filter it down to selected features
    >>> X_filtered = feat_selector.transform(X)
    """

    SUPPORT = {
        "estimator": ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"],
        "transfer_func": ["vstf_01", "vstf_02", "vstf_03", "vstf_04", "sstf_01", "sstf_02", "sstf_03", "sstf_04"],
        "regression_objective": {"MAE": "min", "MSE": "min", "RMSE": "min", "MRE": "min", "MAPE": "min", "MASE": "min",
                             "NSE": "max", "NNSE": "max", "WI": "max", "PCC": "max", "R2s": "max", "R2": "max", "AR2": "max",
                             "CI": "max", "KGE": "max", "VAF": "max", "A10": "max", "A20": "max"},
        "classification_objective": {"AS": "max", "PS": "max", "NPV": "max", "RS": "max", "F1S": "max", "F2S": "max",
                             "FBS": "max", "SS": "max", "MCC": "max", "JSI": "max", "CKS": "max", "ROC-AUC": "max"},
        "optimizer": list(get_all_optimizers().keys())
    }

    def __init__(self, problem="classification", estimator="knn", estimator_paras=None,
                 optimizer="BaseGA", optimizer_paras=None, transfer_func="vstf_01", obj_name=None):
        super().__init__(problem)
        self.estimator = self._set_estimator(estimator, estimator_paras)
        self.optimizer_paras = estimator_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.transfer_func_ = self._set_transfer_func(transfer_func)
        self.obj_name = obj_name

    def _set_estimator(self, estimator=None, paras=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORT["estimator"])
            return get_general_estimator(self.problem, estimator_name, paras)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and \
                (callable(estimator.fit) and callable(estimator.predict)):
            return estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=100, pop_size=30)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def _set_transfer_func(self, transfer_func="vstf_01"):
        if transfer_func is None:
            return getattr(transfer, "vstf_01")
        elif type(transfer_func) is str:
            transfer_func = validator.check_str("transfer_func", transfer_func, self.SUPPORT["transfer_func"])
            return getattr(transfer, transfer_func)
        elif callable(transfer_func):
            return transfer_func
        else:
            raise TypeError(f"transfer_func needs to be a callable function or a string with valid value belongs to {self.SUPPORT['transfer_func']}")

    def _set_metric(self, metric_name=None, list_supported_metrics=None):
        if type(metric_name) is str:
            return validator.check_str("obj_name", metric_name, list_supported_metrics)
        else:
            raise ValueError(f"obj_name should be a string and belongs to {list_supported_metrics}")

    def fit(self, X, y=None, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        fit_weights : list, tuple or np.ndarray, default = (0.9, 0.1)
            The first weight is for objective value and the second weight is for the number of features

        verbose : int, default = True
            Controls verbosity of output.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) to do the tasks (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.data = Data(X.copy(), y.copy())
        self.data.split_train_test(test_size=0.25)
        lb = [-8, ] * X.shape[1]
        ub = [8, ] * X.shape[1]
        if self.problem == "classification":
            if len(np.unique(y)) == 2:
                self.obj_paras = {"average": "micro"}
            else:
                self.obj_paras = {"average": "weighted"}
            if self.obj_name is None:
                self.obj_name = "AS"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["classification_objective"])
            minmax = self.SUPPORT["classification_objective"][self.obj_name]
            metric_class = ClassificationMetric
        else:
            self.obj_paras = {"decimal": 4}
            if self.obj_name is None:
                self.obj_name = "MSE"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["regression_objective"])
            minmax = self.SUPPORT["regression_objective"][self.obj_name]
            metric_class = RegressionMetric
        fit_sign = -1 if minmax == "max" else 1
        log_to = "console" if verbose else "None"
        prob = FeatureSelectionProblem(lb, ub, minmax, data=self.data,
                                       estimator=self.estimator, transfer_func=self.transfer_func_, obj_name=self.obj_name,
                                       metric_class=metric_class, fit_weights=fit_weights, fit_sign=fit_sign, log_to=log_to,
                                       obj_weights=(1.0, 0.), obj_paras=self.obj_paras)
        best_position, best_fitness = self.optimizer.solve(prob, mode=mode, n_workers=n_workers, termination=termination)
        self.selected_feature_solution = np.array(best_position, dtype=int)
        self.selected_feature_masks = np.where(self.selected_feature_solution == 0, False, True)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

    def transform(self, X):
        return X[:, self.selected_feature_indexes]

    def fit_transform(self, X, y=None, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        self.fit(X, y, fit_weights, verbose, mode, n_workers, termination)
        return self.transform(X)

    def get_best_obj_and_fit(self):
        return {
            "obj": self.optimizer.solution[1][1][1],
            "fit": self.optimizer.solution[1][1][0]
        }


class MultiMhaSelector(Selector):

    SUPPORT = {
        "estimator": ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"],
        "transfer_func": ["vstf_01", "vstf_02", "vstf_03", "vstf_04", "sstf_01", "sstf_02", "sstf_03", "sstf_04"],
        "regression_objective": {"MAE": "min", "MSE": "min", "RMSE": "min", "MRE": "min", "MAPE": "min", "MASE": "min",
                             "NSE": "max", "NNSE": "max", "WI": "max", "PCC": "max", "R2s": "max", "R2": "max", "AR2": "max",
                             "CI": "max", "KGE": "max", "VAF": "max", "A10": "max", "A20": "max"},
        "classification_objective": {"AS": "max", "PS": "max", "NPV": "max", "RS": "max", "F1S": "max", "F2S": "max",
                             "FBS": "max", "SS": "max", "MCC": "max", "JSI": "max", "CKS": "max", "ROC-AUC": "max"},
        "optimizer": list(get_all_optimizers().keys())
    }

    def __init__(self, problem="classification", estimator="knn", estimator_paras=None,
                 list_optimizers=("BaseGA",), list_optimizer_paras=None, transfer_func="vstf_01", obj_name=None):
        super().__init__(problem)
        self.estimator, self.estimator_paras = self._set_estimator(estimator, estimator_paras)
        self.list_optimizers, self.list_optimizer_paras = self._set_optimizers(list_optimizers, list_optimizer_paras)
        self.transfer_func = self._set_transfer_func(transfer_func)
        self.obj_name = obj_name

    def _set_estimator(self, estimator=None, paras=None):
        if type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORT['estimator'])
            return get_general_estimator(self.problem, estimator_name, paras), paras
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and (callable(estimator.fit) and callable(estimator.predict)):
            return estimator, paras
        else:
            raise NotImplementedError(f"Your defined estimator needs to implement at least 'fit' and 'predict' functions.")

    def _set_optimizers(self, list_optimizers=None, list_paras=None):
        if type(list_optimizers) not in (list, tuple):
            raise ValueError("list_optimizers should be a list or tuple.")
        else:
            if list_paras is None or type(list_paras) not in (list, tuple):
                list_paras = [{}, ] * len(list_optimizers)
            elif len(list_paras) != len(list_optimizers):
                raise ValueError("list_optimizer_paras should be a list with the same length as list_optimizers")
            list_opts = []
            for idx, opt in enumerate(list_optimizers):
                if type(opt) is str:
                    opt_class = get_optimizer_by_name(opt)
                    if type(list_paras[idx]) is dict:
                        list_opts.append(opt_class(**list_paras[idx]))
                    else:
                        list_opts.append(opt_class(epoch=100, pop_size=30))
                elif isinstance(opt, Optimizer):
                    if type(list_paras[idx]) is dict:
                        opt.set_parameters(list_paras[idx])
                    list_opts.append(opt)
                else:
                    raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")
        return list_opts, list_paras

    def _set_transfer_func(self, transfer_func="vstf_01"):
        if transfer_func is None:
            transfer_func = "vstf_01"
        if type(transfer_func) is str:
            transfer_func = validator.check_str("transfer_func", transfer_func, self.SUPPORT['transfer_func'])
            return getattr(transfer, transfer_func)
        elif callable(transfer_func):
            return transfer_func
        else:
            raise TypeError(f"transfer_func needs to be a callable function or a string with valid value belongs to {self.SUPPORT['transfer_func']}")

    def _set_metric(self, metric_name=None, list_supported_metrics=None):
        if type(metric_name) is str:
            return validator.check_str("obj_name", metric_name, list_supported_metrics)
        else:
            raise ValueError(f"obj_name should be a string and belongs to {list_supported_metrics}")

    def __run__(self, id_trial, model, problem, mode="single", n_workers=2, termination=None):
        _, best_fitness = model.solve(problem, mode=mode, n_workers=n_workers, termination=termination)
        return {
            "id_trial": id_trial,
            "best_fitness": best_fitness,
            "best_solution": model.solution,
            "convergence": model.history.list_global_best_fit
        }

    def fit(self, X, y=None, n_trials=2, n_jobs=2, save_path="history", save_results=True,
            verbose=True, fit_weights=(0.9, 0.1), mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        n_trials : int. Number of repetitions

        n_jobs : int, None. Number of processes will be used to speed up the computation (<=1 or None: sequential, >=2: parallel)

        save_path : str. The path to the folder that hold results

        save_results : bool. Save the global best fitness and loss (convergence/fitness) during generations to csv file (default: True)

        fit_weights : list, tuple or np.ndarray, default = (0.9, 0.1)
            The first weight is for objective value and the second weight is for the number of features

        verbose : int, default = True
            Controls verbosity of output.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) used in Optimizer (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.data = Data(X.copy(), y.copy())
        self.data.split_train_test(test_size=0.25)
        lb = [-8, ] * X.shape[1]
        ub = [8, ] * X.shape[1]
        if self.problem == "classification":
            if len(np.unique(y)) == 2:
                self.obj_paras = {"average": "micro"}
            else:
                self.obj_paras = {"average": "weighted"}
            if self.obj_name is None:
                self.obj_name = "AS"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT['classification_objective'])
            minmax = self.SUPPORT['classification_objective'][self.obj_name]
            metric_class = ClassificationMetric
        else:
            self.obj_paras = {"decimal": 4}
            if self.obj_name is None:
                self.obj_name = "MSE"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT['regression_objective'])
            minmax = self.SUPPORT['regression_objective'][self.obj_name]
            metric_class = RegressionMetric
        fit_sign = -1 if minmax == "max" else 1
        log_to = "console" if verbose else "None"
        prob = FeatureSelectionProblem(lb, ub, minmax, data=self.data,
                                       estimator=self.estimator, transfer_func=self.transfer_func, obj_name=self.obj_name,
                                       metric_class=metric_class, fit_weights=fit_weights, fit_sign=fit_sign, log_to=log_to,
                                       obj_weights=(1.0, 0.), obj_paras=self.obj_paras)
        list_problems = [deepcopy(prob) for _ in range(len(self.list_optimizers))]

        self.n_trials = validator.check_int("n_trials", n_trials, [1, 100000])
        if (type(n_jobs) is int) and (n_jobs >= 1):
            n_jobs = validator.check_int("n_jobs", n_jobs, [2, min(61, os.cpu_count() - 1)])
        else:
            n_jobs = None
        ## Check parent directories
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.save_results = save_results
        best_fit_results = []
        convergence_results = {}
        solution_results = {}
        position_results = {}
        self.list_selected_feature_solutions = {}
        self.list_selected_feature_masks = {}
        self.list_selected_feature_indexes = {}
        trial_list = list(range(1, self.n_trials + 1))
        for idx in trial_list:
            convergence_results[f"trial{idx}"] = {}
            solution_results[f"trial{idx}"] = {}
            position_results[f"trial{idx}"] = {}
            self.list_selected_feature_solutions[f"trial{idx}"] = {}
            self.list_selected_feature_masks[f"trial{idx}"] = {}
            self.list_selected_feature_indexes[f"trial{idx}"] = {}
        for id_model, model in enumerate(self.list_optimizers):
            if n_jobs is None:
                with parallel.ProcessPoolExecutor(n_workers) as executor:
                    list_results = executor.map(partial(self.__run__, model=model, problem=list_problems[id_model],
                                                        mode=mode, n_workers=n_workers, termination=termination), trial_list)
                    for result in list_results:
                        best_fit_results.append({"model": model.get_name(), "trial": result['id_trial'], "best_fitness": result['best_fitness']})
                        convergence_results[f"trial{result['id_trial']}"][model.get_name()] = result['convergence']
                        solution_results[f"trial{result['id_trial']}"][model.get_name()] = result['best_solution']
                        if verbose:
                            print(f"Solving problem: {list_problems[id_model].get_name()} using algorithm: {model.get_name()}, "
                                  f"on the: {result['id_trial']} trial, with best fitness: {result['best_fitness']}")
            else:
                for idx in trial_list:
                    result = self.__run__(idx, model=model, problem=list_problems[id_model], mode=mode, n_workers=n_workers, termination=termination)
                    best_fit_results.append({"model": model.get_name(), "trial": result['id_trial'], "best_fitness": result['best_fitness']})
                    convergence_results[f"trial{result['id_trial']}"][model.get_name()] = result['convergence']
                    solution_results[f"trial{result['id_trial']}"][model.get_name()] = result['best_solution']
                    if verbose:
                        print(f"Solving problem: {list_problems[id_model].get_name()} using algorithm: {model.get_name()}, "
                              f"on the: {result['id_trial']} trial, with best fitness: {result['best_fitness']}")

        for trial_, models_ in solution_results.items():
            for name_, sol in models_.items():
                # Get the position, fitness, and objective
                position_results[trial_][name_] = np.concatenate((sol[0], [sol[1][0], sol[1][1][1]]), axis=None)
                self.list_selected_feature_solutions[trial_][name_] = np.array(sol[0], dtype=int)
                self.list_selected_feature_masks[trial_][name_] = np.where(sol[0] == 0, False, True)
                self.list_selected_feature_indexes[trial_][name_] = np.where(sol[0] == 1)[0]
            indexes = [f"x{idx}" for idx in range(1, len(sol[0]) + 1)] + ["fitness", "objective"]
            df0 = pd.DataFrame(position_results[trial_])
            df0["idx"] = indexes
            df0.set_index("idx", inplace=True)
            df0.T.to_csv(f"{save_path}/solution-{trial_}.csv", index_label="Model")

        if save_results:
            df1 = pd.DataFrame(best_fit_results)
            df1.to_csv(f"{save_path}/best_fitness.csv", index=False)
            for idx in trial_list:
                df2 = pd.DataFrame(convergence_results[f"trial{idx}"])
                df2.to_csv(f"{save_path}/convergence-trial{idx}.csv", index=False)

    def export_boxplot_figures(self, xlabel="Model", ylabel="Global best fitness value",
                               title="Boxplot of comparison models", show_legend=True, show_mean_only=False, exts=(".png", ".pdf")):
        if xlabel is None:
            xlabel = ""
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        boxmean = True if show_mean_only else "sd"
        if not self.save_results:
            raise ValueError("You didn't set save_results parameter to True when you call fit() function.")
        df = pd.read_csv(f"{self.save_path}/best_fitness.csv")
        fig = px.box(df, x="model", y="best_fitness", color="model", labels={"model": xlabel, "best_fitness": ylabel})
        fig.update_traces(boxmean=boxmean) # boxmean=True if want to show mean only
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=show_legend,
            title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
        for idx, ext in enumerate(exts):
            fig.write_image(f"{self.save_path}/boxplot{ext}")

    def export_convergence_figures(self, xlabel="Epoch", ylabel="Fitness value",
                               title="Convergence chart of comparison models", exts=(".png", ".pdf")):
        if xlabel is None:
            xlabel = ""
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        if not self.save_results:
            raise ValueError("You didn't set save_results parameter to True when you call fit() function.")
        for trial in range(1, self.n_trials+1):
            df = pd.read_csv(f"{self.save_path}/convergence-trial{trial}.csv")
            df = df.reset_index()
            # Melt the DataFrame to convert it from wide to long format
            df_long = pd.melt(df, id_vars='index', var_name='Column', value_name='Value')
            # Define the line chart using Plotly Express
            fig = px.line(df_long, x='index', y='Value', color='Column', labels={'index': xlabel, 'Value': ylabel, 'Column': 'Model'})
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=True,
                title={'text': title, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
            for idx, ext in enumerate(exts):
                fig.write_image(f"{self.save_path}/convergence-trial{trial}{ext}")

    def transform(self, X, trial=1, model="BaseGA", all_models=False):
        if all_models:
            dict_X = {}
            for trial_, models_ in self.list_selected_feature_indexes.items():
                dict_X[trial_] = {}
                for name_, indexes in models_.items():
                    dict_X[trial_][name_] = deepcopy(X[:, indexes])
            return dict_X
        else:
            if type(trial) is int and 1 <= trial <= self.n_trials:
                list_models = [model.get_name() for model in self.list_optimizers]
                if type(model) is str and model in list_models:
                    return X[:, self.list_selected_feature_indexes[f"trial{trial}"][model]]
                else:
                    raise ValueError(f"model: {model} is not trained yet.")
            else:
                raise ValueError("trial index should be >= 1 and <= n_trials.")

    def fit_transform(self, X, y=None, n_trials=2, n_jobs=2, save_path="history", save_results=True,
                      verbose=True, fit_weights=(0.9, 0.1), mode='single', n_workers=None, termination=None):
        self.fit(X, y, n_trials, n_jobs, save_path, save_results, verbose, fit_weights, mode, n_workers, termination)
        return self.transform(X, trial=1, model="BaseGa", all_models=False)

    def evaluate(self, estimator=None, estimator_paras=None, data=None, metrics=None, save_path="history", verbose=False):
        """
        Evaluate the new dataset. We will re-train the estimator with training set
        and return the metrics of both training and testing set

        Parameters
        ----------
        estimator : str or Estimator instance (from scikit-learn or custom)
            If estimator is str, we are currently support:
                - knn: k-nearest neighbors
                - svm: support vector machine
                - rf: random forest
                - adaboost: AdaBoost
                - xgb: Gradient Boosting
                - tree: Extra Trees
                - ann: Artificial Neural Network (Multi-Layer Perceptron)

            If estimator is Estimator instance: you need to make sure that it has `fit` and `predict` methods

        estimator_paras: None or dict, default = None
            The parameters of the estimator, please see the official document of scikit-learn to selected estimator.
            If None, we use the default parameter for selected estimator

        data : Data, an instance of Data class. It must have training and testing set

        metrics : tuple, list, default = None
            Depend on the regression or classification you are trying to tackle. The supported metrics can be found at:
            https://github.com/thieu1995/permetrics

        save_path : str, default="history"
            The path to save the file

        verbose : bool, default=False
            Print the results to console or not.

        Returns
        -------
        metrics_results: dict.
            The metrics for both training and testing set.
        """

        if estimator is None:
            if self.estimator is None:
                raise ValueError("You need to set estimator to evaluate the data.")
            est_ = self.estimator
        elif type(estimator) is str:
            estimator_name = validator.check_str("estimator", estimator, self.SUPPORTED_ESTIMATORS)
            est_ = get_general_estimator(self.problem, estimator_name, estimator_paras)
        elif (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')) and (callable(estimator.fit) and callable(estimator.predict)):
            est_ = estimator
        else:
            raise NotImplementedError(f"Your estimator needs to implement at least 'fit' and 'predict' functions.")
        if (metrics is None) or (type(metrics) not in (tuple, list)):
            raise ValueError("You need to pass a tuple/list of performance metrics. See the supported metrics at https://github.com/thieu1995/permetrics")
        if isinstance(data, Data):
            list_results = []
            dict_X_train = self.transform(data.X_train, all_models=True)
            dict_X_test = self.transform(data.X_test, all_models=True)
            for trial_, model_ in dict_X_train.items():
                for name_, X_train_selected in model_.items():
                    est_.fit(X_train_selected, data.y_train)
                    y_train_pred = est_.predict(X_train_selected)
                    y_test_pred = est_.predict(dict_X_test[trial_][name_])
                    train_result = get_metrics(self.problem, data.y_train, y_train_pred, metrics=metrics, testcase="train")
                    test_result = get_metrics(self.problem, data.y_test, y_test_pred, metrics=metrics, testcase="test")
                    list_results.append({"model": name_, "trial": trial_, **train_result, **test_result})
            if verbose:
                print(list_results)
            df = pd.DataFrame(list_results)
            df.to_csv(f"{save_path}/evaluation_results.csv", index_label="Index")
        else:
            raise ValueError("'data' should be an instance of Data class.")

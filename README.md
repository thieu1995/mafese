
<p align="center">
<img style="max-width:100%;" 
src="https://thieu1995.github.io/post/2023-08/mafese-02.png" 
alt="MAFESE"/>
</p>

---

[![GitHub release](https://img.shields.io/badge/release-0.2.0-yellow.svg)](https://github.com/thieu1995/mafese/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/mafese) 
[![PyPI version](https://badge.fury.io/py/mafese.svg)](https://badge.fury.io/py/mafese)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mafese.svg)
![PyPI - Status](https://img.shields.io/pypi/status/mafese.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mafese.svg)
[![Downloads](https://static.pepy.tech/badge/mafese)](https://pepy.tech/project/mafese)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/mafese.svg)
[![Documentation Status](https://readthedocs.org/projects/mafese/badge/?version=latest)](https://mafese.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/mafese.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/545209353.svg)](https://doi.org/10.5281/zenodo.7969042)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MAFESE (Metaheuristic Algorithms for FEature SElection) is the biggest python library for feature selection (FS) 
problem using meta-heuristic algorithms.

* **🆓 Free software:** GNU General Public License (GPL) V3 license
* **🔄 Total Wrapper-based (Metaheuristic Algorithms):** > 200 methods
* **📊 Total Filter-based (Statistical-based):** > 15 methods
* **🌳 Total Embedded-based (Tree and Lasso):** > 10 methods
* **🔍 Total Unsupervised-based:** ≥ 4 methods
* **📂 Total datasets:** ≥ 30 (47 classifications and 7 regressions)
* **📈 Total performance metrics:** ≥ 61 (45 regressions and 16 classifications)
* **⚙️ Total objective functions (as fitness functions):** ≥ 61 (45 regressions and 16 classifications)
* **📖 Documentation:** [https://mafese.readthedocs.io/en/latest/](https://mafese.readthedocs.io/en/latest/)
* **🐍 Python versions:** ≥ 3.7.x
* **📦 Dependencies:** `numpy`, `scipy`, `scikit-learn`, `pandas`, `mealpy`, `permetrics`, `plotly`, `kaleido`


## Citation Request

Please include these citations if you plan to use this incredible library:


```bibtex

@article{van2024feature,
  title={Feature selection using metaheuristics made easy: Open source MAFESE library in Python},
  author={Van Thieu, Nguyen and Nguyen, Ngoc Hung and Heidari, Ali Asghar},
  journal={Future Generation Computer Systems},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.future.2024.06.006},
  url={https://doi.org/10.1016/j.future.2024.06.006},
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

```

# Usage

## Goals

- **Our library provides all state-of-the-art feature selection methods**:
  + Unsupervised-based FS
  + Filter-based FS
  + Embedded-based FS
    + Regularization (Lasso-based)
    + Tree-based methods
  + Wrapper-based FS
    + Sequential-based: forward and backward
    + Recursive-based
    + MHA-based: Metaheuristic Algorithms

## Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/mafese):
```sh 
$ pip install mafese
```

After installation, you can import MAFESE and check its installed version:

```sh
$ python
>>> import mafese
>>> mafese.__version__
```



<details><summary><h2>Lib's structure</h2></summary>

```code
docs
examples
mafese
    data/
        cls/
            aggregation.csv
            Arrhythmia.csv
            ...
        reg/
            boston-housing.csv
            diabetes.csv
            ...
    wrapper/
        mha.py
        recursive.py
        sequential.py
    embedded/
        lasso.py
        tree.py
    filter.py
    unsupervised.py
    utils/
        correlation.py
        data_loader.py
        encoder.py
        estimator.py
        mealpy_util.py
        transfer.py
        validator.py
    __init__.py
    selector.py
README.md
setup.py
```

</details>


## Examples

Let's go through some examples.

### 1. First, load dataset. You can use the available datasets from Mafese:

```python
# Load available dataset from MAFESE
from mafese import get_dataset

# Try unknown data
get_dataset("unknown")
# Enter: 1      -> This wil list all of avaialble dataset

data = get_dataset("Arrhythmia")
```

* Or you can load your own dataset 

```python
import pandas as pd
from mafese import Data

# load X and y
# NOTE mafese accepts numpy arrays only, hence the .values attribute
dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)
```

### 2. Next, prepare your dataset


#### 2.1 Split dataset into train and test set

```python
data.split_train_test(test_size=0.2, inplace=True)
print(data.X_train[:2].shape)
print(data.y_train[:2].shape)
```

#### 2.2 Feature Scaling

```python
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)   # This is for classification problem only
data.y_test = scaler_y.transform(data.y_test)
```

### 3. Next, choose the Selector that you want to use by first import them:

```python
## First way, we recommended 
from mafese import UnsupervisedSelector, FilterSelector, LassoSelector, TreeSelector
from mafese import SequentialSelector, RecursiveSelector, MhaSelector, MultiMhaSelector

## Second way
from mafese.unsupervised import UnsupervisedSelector
from mafese.filter import FilterSelector
from mafese.embedded.lasso import LassoSelector
from mafese.embedded.tree import TreeSelector
from mafese.wrapper.sequential import SequentialSelector
from mafese.wrapper.recursive import RecursiveSelector
from mafese.wrapper.mha import MhaSelector, MultiMhaSelector
```

### 4. Next, create an instance of Selector class you want to use:

```python
feat_selector = UnsupervisedSelector(problem='classification', method='DR', n_features=5)

feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features=5)

feat_selector = LassoSelector(problem="classification", estimator="lasso", estimator_paras={"alpha": 0.1})

feat_selector = TreeSelector(problem="classification", estimator="tree")

feat_selector = SequentialSelector(problem="classification", estimator="knn", n_features=3, direction="forward")

feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=5)

feat_selector = MhaSelector(problem="classification", estimator="knn",
                            optimizer="BaseGA", optimizer_paras=None,
                            transfer_func="vstf_01", obj_name="AS")

list_optimizers = ("OriginalWOA", "OriginalGWO", "OriginalTLO", "OriginalGSKA")
list_paras = [{"epoch": 10, "pop_size": 30}, ]*4
feat_selector = MultiMhaSelector(problem="classification", estimator="knn",
                            list_optimizers=list_optimizers, list_optimizer_paras=list_paras,
                            transfer_func="vstf_01", obj_name="AS")
```

### 5. Fit the model to X_train and y_train

```python
feat_selector.fit(data.X_train, data.y_train)
```

### 6. Get the information

```python
# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)
```

### 7. Call transform() on the X that you want to filter it down to selected features

```python
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)
```

### 8.You can build your own evaluating method or use our method.

If you use our method, don't transform the data.

#### 8.1 You can use difference estimator than the one used in feature selection process 
```python
feat_selector.evaluate(estimator="svm", data=data, metrics=["AS", "PS", "RS"])

## Here, we pass the data that was loaded above. So it contains both train and test set. So, the results will look 
like this: 
{'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}
```

#### 8.2 You can use the same estimator in feature selection process 
```python
X_test, y_test = data.X_test, data.y_test
feat_selector.evaluate(estimator=None, data=data, metrics=["AS", "PS", "RS"])
```

For more usage examples please look at [examples](/examples) folder.


# Support

## Some popular questions

1. Where do I find the supported metrics like above ["AS", "PS", "RS"]. What is that?

You can find it here: https://github.com/thieu1995/permetrics or use this 

```python
from mafese import MhaSelector 

print(MhaSelector.SUPPORTED_REGRESSION_METRICS)
print(MhaSelector.SUPPORTED_CLASSIFICATION_METRICS)
```

2. How do I know my Selector support which estimator? which methods?

```python
print(feat_selector.SUPPORT) 
```
Or you better read the document from: https://mafese.readthedocs.io/en/latest/

3. I got this type of error. How to solve it?

```python
raise ValueError("Existed at least one new label in y_pred.")
ValueError: Existed at least one new label in y_pred.
```

> This occurs only when you are working on a classification problem with a small dataset that has many classes. For 
  instance, the "Zoo" dataset contains only 101 samples, but it has 7 classes. If you split the dataset into a 
  training and testing set with a ratio of around 80% - 20%, there is a chance that one or more classes may appear 
  in the testing set but not in the training set. As a result, when you calculate the performance metrics, you may 
  encounter this error. You cannot predict or assign new data to a new label because you have no knowledge about the 
  new label. There are several solutions to this problem.


+ 1st: Use the SMOTE method to address imbalanced data and ensure that all classes have the same number of samples.

```python
from imblearn.over_sampling import SMOTE
import pandas as pd
from mafese import Data

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]

X_new, y_new = SMOTE().fit_resample(X, y)
data = Data(X_new, y_new)
```

+ 2nd: Use different random_state numbers in split_train_test() function.
```python
import pandas as pd 
from mafese import Data 

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)
data.split_train_test(test_size=0.2, random_state=10)   # Try different random_state value 
```


<details><summary><h2>Official Links</h2></summary>

* Official source code repository: https://github.com/thieu1995/mafese
* Official document: https://mafese.readthedocs.io/
* Download releases: https://pypi.org/project/mafese/
* Issue tracker: https://github.com/thieu1995/mafese/issues
* Notable changes log: https://github.com/thieu1995/mafese/blob/master/ChangeLog.md
* Examples with different mealpy version: https://github.com/thieu1995/mafese/blob/master/examples.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/aiir-team

</details>



<details><summary><h2>Related Documents</h2></summary>

1. https://neptune.ai/blog/feature-selection-methods
2. https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/
3. https://github.com/LBBSoft/FeatureSelect
4. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2754-0
5. https://github.com/scikit-learn-contrib/boruta_py
6. https://elki-project.github.io/
7. https://sci2s.ugr.es/keel/index.php
8. https://archive.ics.uci.edu/datasets
9. https://python-charts.com/distribution/box-plot-plotly/
10. https://plotly.com/python/box-plots/?_ga=2.50659434.2126348639.1688086416-114197406.1688086416#box-plot-styling-mean--standard-deviation

</details>

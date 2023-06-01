
<p align="center"><img style="height:300px;" src=".github/img/logo.png" alt="MAFESE" title="MAFESE"/></p>

---


[![GitHub release](https://img.shields.io/badge/release-0.1.4-yellow.svg)](https://github.com/thieu1995/mafese/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/mafese) 
[![PyPI version](https://badge.fury.io/py/mafese.svg)](https://badge.fury.io/py/mafese)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mafese.svg)
![PyPI - Status](https://img.shields.io/pypi/status/mafese.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mafese.svg)
[![Downloads](https://pepy.tech/badge/mafese)](https://pepy.tech/project/mafese)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/mafese.svg)
[![Documentation Status](https://readthedocs.org/projects/mafese/badge/?version=latest)](https://mafese.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/mafese.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/545209353.svg)](https://doi.org/10.5281/zenodo.7969042)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MAFESE (Metaheuristic Algorithms for FEature SElection) is the largest python library focused on feature selection 
using meta-heuristic algorithms. 

* **Free software:** GNU General Public License (GPL) V3 license
* **Total Wrapper-based (Metaheuristic Algorithms)**: > 180 methods
* **Total Filter-based (Statistical-based)**: > 12 methods
* **Total Embedded-based (Tree and Lasso)**: > 10 methods
* **Total Unsupervised-based**: >= 4 methods
* **Total classification dataset**: >= 30 datasets
* **Total regression dataset**: >= 3 datasets
* **Total performance metrics (as fitness)**: > 30 metrics
* **Documentation:** https://mafese.readthedocs.io/en/latest/
* **Python versions:** 3.7.x, 3.8.x, 3.9.x, 3.10.x, 3.11.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics


# Installation

### Install with pip

Install the [current PyPI release](https://pypi.python.org/pypi/mafese):
```sh 
$ pip install mafese==0.1.4
```

### Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/mafese.git
$ cd mafese
$ python setup.py install
```


### Lib's structure

```code 
docs
examples
mafese
    data/
        Arrhythmia.csv
        BreastCancer.csv
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
    evaluator.py
README.md
setup.py
```


# Usage

After installation, you can import MAFESE as any other Python module:

```sh
$ python
>>> import mafese
>>> mafese.__version__
```

Let's go through some examples.



### Examples

First, you need to load your dataset, or you can load own available datasets:

```python 
# Load available dataset from MAFESE
from mafese import get_dataset

# Try unknown data
get_dataset("unknown")
# Enter: 1

data = get_dataset("Arrhythmia")
```


```python
# Load your own dataset 
import pandas as pd
from mafese import Data

# load X and y
# NOTE mafese accepts numpy arrays only, hence the .values attribute
dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)
```

Next, split dataset into train and test set

```python 
data.split_train_test(test_size=0.2, inplace=True)
print(data.X_train[:2].shape)
print(data.y_train[:2].shape)
```


Next, how to use Recursive wrapper-based method:

```python
from mafese.wrapper.recursive import RecursiveSelector

# define mafese feature selection method
feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=5)

# find all relevant features - 5 features should be selected
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)
```


Or, how to use Sequential (backward or forward) wrapper-based method:

```python
from mafese.wrapper.sequential import SequentialSelector

# define mafese feature selection method
feat_selector = SequentialSelector(problem="classification", estimator="knn", n_features=3, direction="forward")

# find all relevant features - 5 features should be selected
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)
```


Or, how to use Filter-based feature selection with different correlation methods:

```python
from mafese.filter import FilterSelector

# define mafese feature selection method
feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features=5)

# find all relevant features - 5 features should be selected
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)
```


Or, use Metaheuristic-based feature selection with different metaheuristic algorithms:

```python
from mafese.wrapper.mha import MhaSelector
from mafese import get_dataset
from mafese import evaluator
from sklearn.svm import SVC

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)
print(data.X_train.shape, data.X_test.shape)            # (361, 279) (91, 279)

# define mafese feature selection method
feat_selector = MhaSelector(problem="classification", estimator="knn",
                            optimizer="BaseGA", optimizer_paras=None,
                            transfer_func="vstf_01", obj_name="AS")
# find all relevant features
feat_selector.fit(data.X_train, data.y_train, fit_weights=(0.9, 0.1), verbose=True)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)

# Evaluate final dataset with different estimator with multiple performance metrics
results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
print(results)
# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}
```


Or, use Lasso-based feature selection with different estimator:

```python
from mafese.embedded.lasso import LassoSelector
from mafese import get_dataset
from mafese import evaluator
from sklearn.svm import SVC


data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)
print(data.X_train.shape, data.X_test.shape)            # (361, 279) (91, 279)

# define mafese feature selection method
feat_selector = LassoSelector(problem="classification", estimator="lasso", estimator_paras={"alpha": 0.1})
# find all relevant features
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)

# Evaluate final dataset with different estimator with multiple performance metrics
results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
print(results)
# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}
```


Or, use Tree-based feature selection with different estimator:

```python
from mafese.embedded.tree import TreeSelector
from mafese import get_dataset
from mafese import evaluator
from sklearn.svm import SVC


data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)
print(data.X_train.shape, data.X_test.shape)            # (361, 279) (91, 279)

# define mafese feature selection method
feat_selector = TreeSelector(problem="classification", estimator="tree")
# find all relevant features
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)

# Evaluate final dataset with different estimator with multiple performance metrics
results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
print(results)
# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}
```



For more usage examples please look at [examples](/examples) folder.

### Shortcut 
To call the class

```code 
from mafese import Data, get_dataset
from mafese import FilterSelector
from mafese import SequentialSelector, RecursiveSelector, MhaSelector
from mafese import LassoSelector, TreeSelector
```


# Get helps (questions, problems)

* Official source code repo: https://github.com/thieu1995/mafese
* Official document: https://mafese.readthedocs.io/
* Download releases: https://pypi.org/project/mafese/
* Issue tracker: https://github.com/thieu1995/mafese/issues
* Notable changes log: https://github.com/thieu1995/mafese/blob/master/ChangeLog.md
* Examples with different meapy version: https://github.com/thieu1995/mafese/blob/master/examples.md

* This project also related to our another projects which are "meta-heuristics", "neural-network", and "optimization" 
  check it here
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/aiir-team


**Want to have an instant assistant? Join our telegram community at [link](https://t.me/+fRVCJGuGJg1mNDg1)**
We share lots of information, questions, and answers there. You will get more support and knowledge there.



# References 

If you are using mafese in your project, we would appreciate citations:

```code 
@software{nguyen_van_thieu_2023_7969043,
  author       = {Nguyen Van Thieu},
  title        = {MAFESE: Metaheuristic Algorithm for Feature Selection - An Open Source Python Library},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7969042},
  url          = {https://github.com/thieu1995/mafese}
}
```



```code 
1. https://neptune.ai/blog/feature-selection-methods
2. https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/
3. https://github.com/LBBSoft/FeatureSelect
4. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2754-0
5. https://github.com/scikit-learn-contrib/boruta_py
```


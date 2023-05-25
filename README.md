
<p align="center"><img style="height:300px;" src=".github/img/logo.png" alt="MAFESE" title="MAFESE"/></p>

---


[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/mafese/releases)
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
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/thieu1995/mafese.svg)](http://isitmaintained.com/project/thieu1995/mafese "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/thieu1995/mafese.svg)](http://isitmaintained.com/project/thieu1995/mafese "Percentage of issues still open")
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/mafese.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/545209353.svg)](https://doi.org/10.5281/zenodo.7969042)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MAFESE (Metaheuristic Algorithms for FEature SElection) is the largest python library focused on feature selection 
using meta-heuristic algorithms. 

* **Free software:** GNU General Public License (GPL) V3 license
* **Total Wrapper-based (Metaheuristic Algorithms)**: > 170 methods 
* **Total Filter-based (Statistical-based)**: > 6 methods
* **Total classification dataset**: > 20 datasets
* **Total estimator methods**: > 3 methods
* **Total performance metrics (as fitness)**: > 10 metrics
* **Documentation:** https://mafese.readthedocs.io/en/latest/
* **Python versions:** 3.7.x, 3.8.x, 3.9.x, 3.10.x, 3.11.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, matplotlib, mealpy, permetrics



# Installation

### Install with pip

Install the [current PyPI release](https://pypi.python.org/pypi/mafese):
```sh 
$ pip install mafese==0.1.0
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
    wrapper
        recursive.py
        sequential.py
    filter.py
    utils
        correlation.py
        encoder.py
        estimator.py
        validator.py
    __init__.py
    selector.py
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
from mafese.wrapper.recursive import Recursive

# define mafese feature selection method
feat_selector = Recursive(problem="classification", estimator="rf", n_features=5)

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


Or, how to use Sequential wrapper-based method:

```python
from mafese.wrapper.sequential import Sequential

# define mafese feature selection method
feat_selector = Sequential(problem="classification", estimator="knn", n_features=3, direction="forward")

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
from mafese.filter import Filter

# define mafese feature selection method
feat_selector = Filter(problem='classification', method='SPEARMAN', n_features=5)

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


For more usage examples please look at [examples](/examples) folder.



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



## References 


```code 
1. https://neptune.ai/blog/feature-selection-methods
https://github.com/LBBSoft/FeatureSelect
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2754-0

https://github.com/scikit-learn-contrib/boruta_py
```


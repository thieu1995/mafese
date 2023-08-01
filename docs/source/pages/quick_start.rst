============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/mafese />`_::

   $ pip install mafese==0.1.8


* Install directly from source code::

   $ git clone https://github.com/thieu1995/mafese.git
   $ cd mafese
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/mafese


After installation, you can import MAFESE as any other Python module::

   $ python
   >>> import mafese
   >>> mafese.__version__


===============
Lib's structure
===============

Current's structure::

   docs
   examples
   mafese
      data/
         cls/
         ...csv
         reg/
         ...csv
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

========
Examples
========
Let's go through some examples.

First, you need to load your dataset, or you can load own available datasets::

	# Load available dataset from MAFESE
	from mafese import get_dataset

	# Try unknown data
	get_dataset("unknown")
	# Enter: 1

	data = get_dataset("Arrhythmia")


Load your own dataset if you want::

	import pandas as pd
	from mafese import Data

	# load X and y
	# NOTE mafese accepts numpy arrays only, hence the .values attribute
	dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
	X, y = dataset[:, 0:-1], dataset[:, -1]
	data = Data(X, y)


Next, split dataset into train and test set::

	data.split_train_test(test_size=0.2, inplace=True)
	print(data.X_train[:2].shape)
	print(data.y_train[:2].shape)


Next, how to use Recursive wrapper-based method::

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


Or, how to use Sequential (backward or forward) wrapper-based method::

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


Or, how to use Filter-based feature selection with different correlation methods::

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



Or, use Metaheuristic-based feature selection with different metaheuristic algorithms::

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
	results = feat_selector.evaluate(estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
	print(results)
	# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}


Or, use Lasso-based feature selection with different estimator::

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
	results = feat_selector.evaluate(estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
	print(results)
	# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}



Or, use Tree-based feature selection with different estimator::

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
	results = feat_selector.evaluate(estimator=SVC(), data=data, metrics=["AS", "PS", "RS"])
	print(results)
	# {'AS_train': 0.77176, 'PS_train': 0.54177, 'RS_train': 0.6205, 'AS_test': 0.72636, 'PS_test': 0.34628, 'RS_test': 0.52747}




For more usage examples please look at [examples](/examples) folder.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
